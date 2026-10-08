"""HTTP stand-ins for the voice models when they run on a support node.

Each class here is a drop-in replacement for its counterpart in ``c2/voice.py``
with the same constructor role, the same method names and the same return
types, so ``api.py``/``supportive_app.py`` cannot tell which machine a model is
on. The waveform is already decoded to mono float32 at 16 kHz by the caller, so
it crosses the network as raw little-endian float32 -- no re-encode, no codec
dependency on either side.

Only the three voice services and the reply model can be offloaded this way.
C1 reads a serial port on the machine the wearable is plugged into, and C3 plus
the emotion chain need this repository's own model code, so both
stay in-process. See ../DEPLOYMENT.md.
"""

from __future__ import annotations

import numpy as np

from c2.voice import AppraisalPrediction, TranscriptionResult

SAMPLE_RATE = 16_000


class RemoteServiceError(RuntimeError):
    """A support-node call failed. Raised with the node URL for diagnosis."""


class _NodeClient:
    """Shared request plumbing: URL, timeout and the optional shared secret."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 180.0,
        auth_token: str | None = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)
        self.auth_token = auth_token

    @property
    def is_loaded(self) -> bool:
        """True by definition: the weights are the node's problem, not ours."""

        return True

    def load(self) -> None:
        """No-op; kept so a remote service satisfies the same interface."""

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = content_type
        if self.auth_token:
            headers["X-Auth-Token"] = self.auth_token
        return headers

    def _post(self, path: str, **kwargs):
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "Support-node offloading requires requests; "
                "install backend/requirements.txt"
            ) from exc

        url = f"{self.base_url}{path}"
        try:
            response = requests.post(url, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise RemoteServiceError(f"Support node {url} unreachable: {exc}") from exc
        if response.status_code >= 400:
            detail = response.text.strip()[:400]
            raise RemoteServiceError(f"Support node {url} returned {response.status_code}: {detail}")
        return response

    @staticmethod
    def _pcm(audio: np.ndarray) -> bytes:
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("a non-empty mono waveform is required")
        return waveform.astype("<f4", copy=False).tobytes()


class RemoteVoiceAppraisalAnalyzer(_NodeClient):
    """C2 AudEERING ADV + fuzzy appraisal engine, evaluated on the node."""

    def predict(
        self, audio: np.ndarray, *, file_name: str = "voice_input"
    ) -> AppraisalPrediction:
        response = self._post(
            "/v1/c2/appraise",
            data=self._pcm(audio),
            params={"file_name": file_name},
            headers=self._headers("application/octet-stream"),
        )
        payload = response.json()
        return AppraisalPrediction(
            arousal=float(payload["arousal"]),
            dominance=float(payload["dominance"]),
            valence=float(payload["valence"]),
            dominant_state=str(payload["dominant_state"]),
            dominant_weight=float(payload["dominant_weight"]),
            state_strengths={
                str(state): float(value)
                for state, value in payload["state_strengths"].items()
            },
            interpretation_confidence=float(payload["interpretation_confidence"]),
            appraisal_ambiguity=float(payload["appraisal_ambiguity"]),
            reference_similarity=float(payload["reference_similarity"]),
            uncertain=bool(payload["uncertain"]),
            ood=bool(payload["ood"]),
            low_confidence=bool(payload["low_confidence"]),
            voice_stress=bool(payload["voice_stress"]),
            voice_stress_score=float(payload["voice_stress_score"]),
            contributions=list(payload.get("contributions", [])),
        )


class RemoteVoiceTranscriber(_NodeClient):
    """Faster-Whisper transcription performed on the node."""

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        response = self._post(
            "/v1/stt/transcribe",
            data=self._pcm(audio),
            headers=self._headers("application/octet-stream"),
        )
        payload = response.json()
        text = str(payload.get("text", "")).strip()
        if not text:
            # Same contract as the local transcriber, so api.py's 400 handling
            # is unchanged for a recording that contained no speech.
            raise ValueError("No speech was detected in the audio")
        probability = payload.get("language_probability")
        return TranscriptionResult(
            text=text,
            language=payload.get("language"),
            language_probability=(
                float(probability) if probability is not None else None
            ),
            duration_seconds=float(
                payload.get("duration_seconds", np.asarray(audio).size / SAMPLE_RATE)
            ),
        )


class RemoteKokoroSpeechSynthesizer(_NodeClient):
    """Kokoro synthesis performed on the node; returns the same WAV bytes."""

    def __init__(
        self,
        base_url: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        timeout: float = 180.0,
        auth_token: str | None = None,
    ) -> None:
        super().__init__(base_url, timeout=timeout, auth_token=auth_token)
        # Both optional: omitted values let the node's own config decide, which
        # keeps voice selection in one place when several clients share a node.
        self.voice = voice
        self.speed = speed

    def synthesize(self, text: str) -> bytes:
        normalized = text.strip()
        if not normalized:
            raise ValueError("TTS text cannot be empty")
        request: dict[str, object] = {"text": normalized}
        if self.voice:
            request["voice"] = self.voice
        if self.speed is not None:
            request["speed"] = float(self.speed)
        response = self._post(
            "/v1/tts/synthesize",
            json=request,
            headers=self._headers("application/json"),
        )
        audio = response.content
        if not audio:
            raise RuntimeError("Support node returned no audio")
        return audio


def node_health(
    base_url: str, *, timeout: float = 10.0, auth_token: str | None = None
) -> dict:
    """One-shot reachability probe used by ``/api/health`` and the run scripts."""

    import requests

    url = f"{str(base_url).rstrip('/')}/health"
    headers = {"X-Auth-Token": auth_token} if auth_token else {}
    try:
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
    except Exception as exc:
        return {"url": base_url, "reachable": False, "error": str(exc)}
    payload = response.json()
    payload.update({"url": base_url, "reachable": True})
    return payload
