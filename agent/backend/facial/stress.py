"""Facial stress evidence from webcam frames (WorkStress3D FacialStressCNN).

The checkpoint is the deployment model from
``WorkStress3D_Facial_Stress_Detection_FINAL.ipynb``: a 4-block grayscale CNN
trained on 83 48x48 faces labelled non-stressful (0) / stressful (1). Its
5-fold cross-validation is weak -- accuracy 0.63 +/- 0.16, ROC-AUC 0.68,
specificity 0.53 -- and the faces carry no subject IDs, so even that is likely
optimistic. Two consequences shape this module:

* **Never score a frame without a face.** The CNN reports ~0.95 stress for a
  flat gray image, so an empty or badly lit frame would read as strong stress.
  Each frame goes through the notebook's Haar-cascade face detector first and
  is dropped when no face is found.
* **One frame is not evidence.** The turn's result is the mean over every frame
  that contained a face, and the turn only reports at all once
  ``min_face_frames`` such frames arrived. Below that it is "no reading", the
  same as a wearable that has not calibrated.

How far the result may sway a reply is ``signal_influence.face_weight`` in
config.yaml, not anything here. Frames are decoded in memory and never written
to disk or kept after the turn.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

LOGGER = logging.getLogger(__name__)

IMAGE_SIZE = 48


class FacialStressCNN(nn.Module):
    """Exactly the notebook's architecture, so the state dict loads strictly."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass(frozen=True)
class FacialStressResult:
    """One turn's facial evidence. ``available`` is False for "no reading"."""

    available: bool
    is_stressed: bool
    probability: float | None
    frames_received: int
    faces_detected: int
    min_face_frames: int
    frame_probabilities: tuple[float, ...] = field(default=())
    reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "available": self.available,
            "is_stressed": self.is_stressed,
            "probability": self.probability,
            "frames_received": self.frames_received,
            "faces_detected": self.faces_detected,
            "min_face_frames": self.min_face_frames,
            "reason": self.reason,
        }


def unavailable(reason: str, *, frames_received: int = 0, min_face_frames: int = 0) -> FacialStressResult:
    return FacialStressResult(
        available=False,
        is_stressed=False,
        probability=None,
        frames_received=frames_received,
        faces_detected=0,
        min_face_frames=min_face_frames,
        reason=reason,
    )


def aggregate_frame_probabilities(
    probabilities: Sequence[float],
    *,
    frames_received: int,
    min_face_frames: int,
    threshold: float,
) -> FacialStressResult:
    """Mean over the frames that held a face, or "no reading" below the floor."""

    kept = tuple(float(p) for p in probabilities)
    if len(kept) < max(1, min_face_frames):
        reason = (
            "no face was visible to the camera"
            if not kept
            else f"a face was visible in only {len(kept)} of the "
            f"{min_face_frames} frames needed"
        )
        return FacialStressResult(
            available=False,
            is_stressed=False,
            probability=None,
            frames_received=frames_received,
            faces_detected=len(kept),
            min_face_frames=min_face_frames,
            frame_probabilities=kept,
            reason=reason,
        )
    probability = float(np.mean(kept))
    return FacialStressResult(
        available=True,
        is_stressed=probability >= threshold,
        probability=probability,
        frames_received=frames_received,
        faces_detected=len(kept),
        min_face_frames=min_face_frames,
        frame_probabilities=kept,
    )


class FacialStressAnalyzer:
    """Haar-cascade face crop + FacialStressCNN, loaded once and kept on CPU."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        threshold: float = 0.5,
        min_face_frames: int = 3,
        max_frames: int = 40,
        min_face_size: int = 60,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = torch.device(device)
        self.threshold = float(threshold)
        self.min_face_frames = max(1, int(min_face_frames))
        self.max_frames = max(1, int(max_frames))
        self.min_face_size = max(20, int(min_face_size))
        self._model: FacialStressCNN | None = None
        self._cascade = None
        self._mean = 0.5
        self._std = 0.5
        self._lock = threading.Lock()
        # OpenCV's cascade is not documented as thread-safe, and the preview
        # check can arrive while a voice turn is being scored.
        self._run_lock = threading.Lock()
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Facial stress threshold must be between 0 and 1")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._cascade is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError(
                    "Facial stress detection requires opencv-python-headless; "
                    "install backend/requirements.txt"
                ) from exc
            if not self.checkpoint.is_file():
                raise FileNotFoundError(
                    f"Facial stress checkpoint not found: {self.checkpoint}"
                )
            checkpoint = torch.load(
                self.checkpoint, map_location="cpu", weights_only=False
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model = FacialStressCNN()
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device).eval()
            self._mean = float((checkpoint.get("normalization_mean") or [0.5])[0])
            self._std = float((checkpoint.get("normalization_std") or [0.5])[0])
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                raise RuntimeError(f"Could not load the face detector: {cascade_path}")
            self._model = model
            self._cascade = cascade

    # -- per-frame steps ---------------------------------------------------

    @staticmethod
    def _decode_gray(image_bytes: bytes) -> np.ndarray | None:
        import cv2

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

    def _largest_face(self, gray: np.ndarray) -> np.ndarray | None:
        """The notebook's detector settings; the largest face wins."""

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        return gray[y : y + h, x : x + w]

    def _to_tensor(self, face_gray: np.ndarray) -> torch.Tensor:
        # torchvision's Resize on a PIL image is PIL's bilinear resize, so this
        # matches the notebook's eval_transform without needing torchvision.
        from PIL import Image

        resized = Image.fromarray(face_gray).convert("L").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - self._mean) / self._std
        return torch.from_numpy(array)[None, None]

    def detect_face(self, image_bytes: bytes) -> bool:
        """Whether a frame holds a usable face -- the camera preview's hint."""

        self.load()
        gray = self._decode_gray(image_bytes)
        if gray is None:
            return False
        with self._run_lock:
            return self._largest_face(gray) is not None

    def predict_frames(self, frames: Sequence[bytes]) -> FacialStressResult:
        """Score every frame that holds a face and average them for the turn."""

        frames = list(frames)
        if not frames:
            return unavailable(
                "the camera was off for this turn",
                min_face_frames=self.min_face_frames,
            )
        self.load()
        if len(frames) > self.max_frames:
            # Keep the spread across the whole utterance, not just its start.
            picks = np.linspace(0, len(frames) - 1, self.max_frames).round().astype(int)
            frames = [frames[i] for i in picks]

        probabilities: list[float] = []
        with self._run_lock:
            crops = []
            for image_bytes in frames:
                gray = self._decode_gray(image_bytes)
                if gray is None:
                    continue
                face = self._largest_face(gray)
                if face is not None:
                    crops.append(self._to_tensor(face))
            if crops:
                batch = torch.cat(crops).to(self.device)
                with torch.no_grad():
                    probabilities = (
                        torch.softmax(self._model(batch), dim=1)[:, 1].cpu().tolist()
                    )
        return aggregate_frame_probabilities(
            probabilities,
            frames_received=len(frames),
            min_face_frames=self.min_face_frames,
            threshold=self.threshold,
        )
