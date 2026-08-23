"""Shared 50-case evaluation support for the two saved BERTopic models.

BERTopic is unsupervised, so these tests validate inference and assignment
contracts instead of pretending that the case categories are trained class
labels.  Clear, weak, and ambiguous outcomes are retained for reporting.
"""

from __future__ import annotations

import csv
import gc
import html
import importlib.util
import math
import os
import re
import sys
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_BERTOPIC_TESTS = os.getenv("RUN_C3_BERTOPIC_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
FRONTEND_HELPER = (
    PROJECT_ROOT
    / "streamlit_app/c3_text_stressor_distortion/stressor head/bertopic_frontend.py"
)
HEADER_PATHS = {
    "stress": {
        "model": PROJECT_ROOT
        / "models/c3_text_stressor_distortion/Stress header/BERTopic",
        "report": PROJECT_ROOT
        / "reports/c3_text_stressor_distortion/Stress header/BERTopic",
        # This is the stricter threshold used by the React diary backend.
        "minimum_similarity": 0.48,
    },
    "cbt": {
        "model": PROJECT_ROOT
        / "models/c3_text_stressor_distortion/CBT header/BERTopic",
        "report": PROJECT_ROOT
        / "reports/c3_text_stressor_distortion/CBT header/BERTopic",
        # This matches the CBT Streamlit research interface.
        "minimum_similarity": 0.45,
    },
}


@dataclass(frozen=True)
class TopicCase:
    case_id: str
    text: str
    category: str
    rationale: str


@dataclass(frozen=True)
class TopicOutput:
    topic_id: int
    theme: str
    similarity: float
    status: str
    is_clear: bool
    is_weak: bool
    is_ambiguous: bool
    runner_up_similarity: float | None
    top_two_margin: float | None
    terms: tuple[str, ...]
    elapsed_ms: float


def load_topic_cases(path: Path) -> tuple[TopicCase, ...]:
    """Load and strictly validate an exactly 50-row topic test corpus."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50:
        raise ValueError(f"{path} must contain exactly 50 cases; found {len(rows)}")

    cases: list[TopicCase] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        missing = {
            field
            for field in ("case_id", "text", "category", "rationale")
            if not (row.get(field) or "").strip()
        }
        if missing:
            raise ValueError(
                f"{path}:{row_number} has blank fields: {', '.join(sorted(missing))}"
            )
        case_id = row["case_id"].strip()
        if case_id in seen_ids:
            raise ValueError(f"{path}:{row_number} duplicates case_id {case_id!r}")
        seen_ids.add(case_id)
        cases.append(
            TopicCase(
                case_id=case_id,
                text=row["text"].strip(),
                category=row["category"].strip(),
                rationale=row["rationale"].strip(),
            )
        )
    return tuple(cases)


def _load_frontend_helper():
    module_name = "c3_bertopic_test_frontend"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, FRONTEND_HELPER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load BERTopic helper: {FRONTEND_HELPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"/?r/[A-Za-z0-9_]+|/?u/[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"&#x200B;|\u200b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class LoadedTopicPredictor:
    """Nearest-centroid predictor shared by the saved Stress and CBT topics."""

    def __init__(self, header: str) -> None:
        if header not in HEADER_PATHS:
            raise ValueError(f"Unknown BERTopic header {header!r}")
        self.header = header
        self.settings = HEADER_PATHS[header]
        self.frontend = _load_frontend_helper()
        self.embedding_model: Any = None
        self.topic_embeddings: np.ndarray | None = None
        self.topic_ids: list[int] = []
        self.catalog: dict[str, Any] | None = None
        self.deployed_predictor: Any = None
        self._load()

    @property
    def available_topic_ids(self) -> tuple[int, ...]:
        assert self.catalog is not None
        return tuple(sorted(topic_id for topic_id in self.catalog["topics"] if topic_id != -1))

    @property
    def minimum_similarity(self) -> float:
        return float(self.settings["minimum_similarity"])

    def _load(self) -> None:
        import json

        import numpy as np
        from safetensors.numpy import load_file
        from sentence_transformers import SentenceTransformer

        if self.header == "stress":
            # Exercise the exact class called by React diary POST /api/diary,
            # including its text cleaning and stricter clarity threshold.
            project = str(PROJECT_ROOT)
            if project not in sys.path:
                sys.path.insert(0, project)
            from backend.c3_text_stressor_distortion.app.topic_model import (
                TopicPredictor,
            )

            self.deployed_predictor = TopicPredictor()
            self.deployed_predictor.load()
            self.embedding_model = self.deployed_predictor._embedding_model
            self.topic_embeddings = self.deployed_predictor._topic_embeddings
            self.topic_ids = list(self.deployed_predictor._topic_ids)
            self.catalog = self.deployed_predictor._catalog
            return

        model_dir = Path(self.settings["model"])
        report_dir = Path(self.settings["report"])
        required = (
            model_dir / "config.json",
            model_dir / "topic_embeddings.safetensors",
            report_dir / "topic_info.csv",
            report_dir / "topic_terms.csv",
            report_dir / "topic_metrics.json",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing BERTopic artifacts: " + ", ".join(missing))

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        model_id = config.get(
            "embedding_model", "sentence-transformers/all-mpnet-base-v2"
        )
        self.embedding_model = SentenceTransformer(model_id, device="cpu")
        tensors = load_file(model_dir / "topic_embeddings.safetensors")
        embeddings = np.asarray(tensors["topic_embeddings"], dtype=np.float32)
        if embeddings.ndim != 2 or not len(embeddings):
            raise ValueError(f"Invalid topic embedding shape: {embeddings.shape}")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.topic_embeddings = embeddings / np.clip(norms, 1e-12, None)

        self.catalog = self.frontend.load_cbt_topic_catalog(report_dir)
        self.topic_ids = self.frontend.resolve_embedding_topic_ids(
            self.catalog["topics"], len(self.topic_embeddings)
        )

    def predict(self, text: str) -> TopicOutput:
        import numpy as np

        if not text or not text.strip():
            raise ValueError("BERTopic input text must not be blank")
        assert self.topic_embeddings is not None
        assert self.catalog is not None

        started = time.perf_counter()
        if self.header == "stress":
            result = self.deployed_predictor.predict(text)
            status = "Clear match" if result.is_clear else "Weak match"
            return TopicOutput(
                topic_id=int(result.topic_id),
                theme=str(result.theme),
                similarity=float(result.similarity),
                status=status,
                is_clear=bool(result.is_clear),
                is_weak=not bool(result.is_clear),
                is_ambiguous=False,
                runner_up_similarity=None,
                top_two_margin=None,
                terms=tuple(str(term) for term in result.terms),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        document_embedding = self.embedding_model.encode(
            [_clean_text(text)],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        similarities = np.matmul(self.topic_embeddings, document_embedding)
        ranked = self.frontend.rank_topic_scores(
            self.topic_ids, similarities.tolist(), top_n=5
        )
        if not ranked:
            raise RuntimeError("No non-outlier BERTopic themes are available")

        assessment = self.frontend.assess_topic_match(
            ranked, minimum_similarity=self.minimum_similarity
        )
        topic = self.catalog["topics"][int(ranked[0]["topic_id"])]
        status = str(assessment["status"])
        return TopicOutput(
            topic_id=int(topic["topic_id"]),
            theme=str(topic["display_name"]),
            similarity=float(ranked[0]["similarity"]),
            status=status,
            is_clear=status == "Clear match",
            is_weak=status == "Weak match",
            is_ambiguous=status == "Ambiguous match",
            runner_up_similarity=(
                float(assessment["runner_up_similarity"])
                if assessment["runner_up_similarity"] is not None
                else None
            ),
            top_two_margin=(
                float(assessment["top_two_margin"])
                if assessment["top_two_margin"] is not None
                else None
            ),
            terms=tuple(str(term) for term in topic["terms"][:5]),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        self.embedding_model = None
        self.topic_embeddings = None
        self.catalog = None
        self.deployed_predictor = None
        gc.collect()


def validate_topic_output(
    output: TopicOutput, available_topic_ids: Sequence[int]
) -> None:
    """Raise AssertionError when an inference result violates the API contract."""
    assert output.topic_id != -1, "nearest-centroid inference returned outlier topic -1"
    assert output.topic_id in available_topic_ids, "topic ID is absent from the catalog"
    assert output.theme.strip(), "topic theme is blank"
    assert output.terms, "topic has no descriptive terms"
    assert math.isfinite(output.similarity), "topic similarity is not finite"
    assert -1.0 <= output.similarity <= 1.0, "cosine similarity is outside [-1, 1]"
    assert output.status in {"Clear match", "Weak match", "Ambiguous match"}
    assert sum((output.is_clear, output.is_weak, output.is_ambiguous)) == 1
    assert output.elapsed_ms >= 0.0


def build_bertopic_test_case(
    *, header: str, cases_path: Path
) -> type[unittest.TestCase]:
    """Create one independently reported BERTopic contract test per CSV row."""
    cases = load_topic_cases(cases_path)

    class BERTopicSemanticTests(unittest.TestCase):
        predictor: LoadedTopicPredictor

        @classmethod
        def setUpClass(cls) -> None:
            if RUN_BERTOPIC_TESTS:
                cls.predictor = LoadedTopicPredictor(header)

        @classmethod
        def tearDownClass(cls) -> None:
            predictor = getattr(cls, "predictor", None)
            if predictor is not None:
                predictor.close()

    BERTopicSemanticTests.__name__ = f"{header.title()}BERTopicTests"
    BERTopicSemanticTests.__qualname__ = BERTopicSemanticTests.__name__
    BERTopicSemanticTests.__doc__ = (
        f"Fifty unsupervised semantic assignment checks for {header} BERTopic."
    )

    def make_test(case: TopicCase) -> Callable[[unittest.TestCase], None]:
        @unittest.skipUnless(
            RUN_BERTOPIC_TESTS,
            "set RUN_C3_BERTOPIC_TESTS=1 (or use run_bertopic_tests.py)",
        )
        def test_method(self: unittest.TestCase) -> None:
            output = self.predictor.predict(case.text)
            try:
                validate_topic_output(output, self.predictor.available_topic_ids)
            except AssertionError as exc:
                self.fail(
                    f"{case.case_id} [{case.category}] {exc}. "
                    f"Rationale: {case.rationale}"
                )

        test_method.__name__ = f"test_{case.case_id.lower()}"
        test_method.__doc__ = f"{case.category}: {case.rationale}"
        return test_method

    for case in cases:
        setattr(BERTopicSemanticTests, f"test_{case.case_id.lower()}", make_test(case))
    return BERTopicSemanticTests
