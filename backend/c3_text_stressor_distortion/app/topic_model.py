"""Live BERTopic theme tagging for diary entries.

Reuses the already-trained Stress header topic model: a saved per-topic
centroid embedding (`topic_embeddings.safetensors`) plus the topic catalog
from its evaluation report. Assigning a new document is just "embed it with
the same sentence-transformer used at training time, then cosine-similarity
against the saved centroids" — no need to reload the full BERTopic/UMAP/HDBSCAN
pipeline. This mirrors `predict_stress_topic` in
streamlit_app/c3_text_stressor_distortion/stressor head/app.py, which already
proved the approach; only the CSV/report loading helpers are reused directly
(via bertopic_frontend.py) rather than duplicated.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from html import unescape
from threading import Lock
from typing import Any

import numpy as np

from .config import (
    BERTOPIC_FRONTEND_PATH,
    DEFAULT_BERTOPIC_EMBEDDING_MODEL,
    STRESS_BERTOPIC_MODEL_DIR,
    STRESS_BERTOPIC_REPORT_DIR,
)

# bertopic_frontend's default (0.45) was tuned on full Reddit-length training
# posts. Short, casual diary sentences sit right at that boundary and can tip
# over it into a nonsense theme (observed: a mundane "quiet evening" entry
# scored 0.452 and matched an unrelated cluster). A silently-omitted tag is
# better than a confidently wrong one, so this raises the bar a bit for
# diary-length text specifically — it does not change the saved model.
MIN_CLEAR_SIMILARITY = 0.48


def _bertopic_frontend():
    module_name = "backend_bertopic_frontend"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, BERTOPIC_FRONTEND_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load bertopic_frontend: {BERTOPIC_FRONTEND_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _clean_text(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"/?r/[A-Za-z0-9_]+|/?u/[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"&#x200B;|\u200b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass
class TopicResult:
    topic_id: int
    theme: str
    similarity: float
    is_clear: bool
    terms: list[str]


class TopicPredictor:
    """Assigns diary text to the nearest saved Stress-header BERTopic theme."""

    def __init__(self):
        self._lock = Lock()
        self._loaded = False
        self._embedding_model = None
        self._topic_embeddings: np.ndarray | None = None
        self._topic_ids: list[int] = []
        self._catalog: dict[str, Any] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            from safetensors.numpy import load_file
            from sentence_transformers import SentenceTransformer

            frontend = _bertopic_frontend()

            required = (
                STRESS_BERTOPIC_MODEL_DIR / "config.json",
                STRESS_BERTOPIC_MODEL_DIR / "topic_embeddings.safetensors",
                STRESS_BERTOPIC_REPORT_DIR / "topic_info.csv",
            )
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    "Missing Stress BERTopic artifacts: " + ", ".join(missing)
                )

            config = json.loads((STRESS_BERTOPIC_MODEL_DIR / "config.json").read_text())
            embedding_model_id = config.get(
                "embedding_model", DEFAULT_BERTOPIC_EMBEDDING_MODEL
            )
            embedding_model = SentenceTransformer(embedding_model_id, device="cpu")

            tensors = load_file(STRESS_BERTOPIC_MODEL_DIR / "topic_embeddings.safetensors")
            topic_embeddings = np.asarray(tensors["topic_embeddings"], dtype=np.float32)
            norms = np.linalg.norm(topic_embeddings, axis=1, keepdims=True)
            topic_embeddings = topic_embeddings / np.clip(norms, 1e-12, None)

            catalog = frontend.load_topic_catalog(STRESS_BERTOPIC_REPORT_DIR)
            topic_ids = frontend.resolve_embedding_topic_ids(
                catalog["topics"], len(topic_embeddings)
            )

            self._embedding_model = embedding_model
            self._topic_embeddings = topic_embeddings
            self._topic_ids = topic_ids
            self._catalog = catalog
            self._loaded = True

    def predict(self, text: str) -> TopicResult:
        self.load()
        assert self._embedding_model is not None
        assert self._topic_embeddings is not None
        assert self._catalog is not None

        frontend = _bertopic_frontend()

        document_embedding = self._embedding_model.encode(
            [_clean_text(text)],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        similarities = np.matmul(self._topic_embeddings, document_embedding)
        ranked = frontend.rank_topic_scores(
            self._topic_ids, similarities.tolist(), top_n=1
        )
        if not ranked:
            raise RuntimeError("No non-outlier topics are available")

        match = frontend.assess_topic_match(ranked, minimum_similarity=MIN_CLEAR_SIMILARITY)
        topic = self._catalog["topics"][ranked[0]["topic_id"]]
        return TopicResult(
            topic_id=topic["topic_id"],
            theme=topic["display_name"],
            similarity=float(ranked[0]["similarity"]),
            is_clear=not match["is_weak"],
            terms=topic["terms"][:5],
        )
