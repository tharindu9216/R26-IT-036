"""Shared infrastructure for the Stress and CBT model case suites.

The model dependencies and checkpoints are deliberately loaded lazily.  This
keeps ordinary unit-test discovery lightweight while allowing the explicit
``run_model_tests.py`` entry point to exercise the real trained artifacts.
"""

from __future__ import annotations

import csv
import gc
import importlib
import json
import math
import os
import pickle
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_MODEL_TESTS = os.getenv("RUN_C3_MODEL_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MODEL_NAMES = (
    "Logistic Regression",
    "SVM",
    "BERT",
    "MentalBERT",
    "DeBERTa-v3",
    "Ensemble",
)


@dataclass(frozen=True)
class ModelCase:
    case_id: str
    text: str
    expected_label: int
    category: str
    rationale: str


@dataclass(frozen=True)
class ModelOutput:
    predicted_label: int
    positive_probability: float


class LoadedPredictor:
    """Small adapter around either one checkpoint or a deployed ensemble."""

    def __init__(
        self,
        predict_fn: Callable[[str], ModelOutput],
        resources: tuple[Any, ...] = (),
    ) -> None:
        self._predict_fn = predict_fn
        self._resources = resources

    def predict(self, text: str) -> ModelOutput:
        return self._predict_fn(text)

    def close(self) -> None:
        self._predict_fn = lambda _text: ModelOutput(0, 0.0)
        for resource in self._resources:
            if isinstance(resource, LoadedPredictor):
                resource.close()
        self._resources = ()
        gc.collect()
        try:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def load_cases(path: Path) -> tuple[ModelCase, ...]:
    """Load and strictly validate a 50-row semantic test corpus."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 50:
        raise ValueError(f"{path} must contain exactly 50 cases; found {len(rows)}")

    cases: list[ModelCase] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        missing = {
            field
            for field in ("case_id", "text", "expected_label", "category", "rationale")
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
        try:
            expected_label = int(row["expected_label"])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{row_number} expected_label must be 0 or 1"
            ) from exc
        if expected_label not in (0, 1):
            raise ValueError(f"{path}:{row_number} expected_label must be 0 or 1")
        cases.append(
            ModelCase(
                case_id=case_id,
                text=row["text"].strip(),
                expected_label=expected_label,
                category=row["category"].strip(),
                rationale=row["rationale"].strip(),
            )
        )

    label_counts = {label: sum(case.expected_label == label for case in cases) for label in (0, 1)}
    if label_counts != {0: 25, 1: 25}:
        raise ValueError(
            f"{path} must be balanced with 25 cases per label; found {label_counts}"
        )
    return tuple(cases)


def _ensure_project_imports() -> None:
    project = str(PROJECT_ROOT)
    if project not in sys.path:
        sys.path.insert(0, project)


def _single_member_predictor(header: str, model_name: str) -> LoadedPredictor:
    _ensure_project_imports()
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    ensembles = importlib.import_module(
        "backend.c3_text_stressor_distortion.app.ensembles"
    )
    app_config = importlib.import_module(
        "backend.c3_text_stressor_distortion.app.config"
    )

    if header == "stress":
        model_module = importlib.import_module(
            "backend.c3_text_stressor_distortion.app.stress_model"
        )
        spec = app_config.STRESS_ENSEMBLE_MEMBERS[model_name]
        checkpoint_path = spec["checkpoint"]
        runtime_hf_id = spec["runtime_hf_id"]
        max_length = int(spec["max_len"])
        threshold = float(app_config.STRESS_DECISION_THRESHOLD)
        state_dict = ensembles.load_state_dict(checkpoint_path, model_module._resolve_device())
        head_weight = state_dict.get("head_1a.fc1.weight")
        if head_weight is None:
            raise KeyError(f"{checkpoint_path} is missing 'head_1a.fc1.weight'")
        device = model_module._resolve_device()
        model = model_module.DualHeadStressModel(
            runtime_hf_id,
            num_subreddit_labels=app_config.DEFAULT_NUM_SUBREDDITS,
            intermediate=int(head_weight.shape[0]),
        )

        def logits_for(ids: Any, mask: Any) -> Any:
            return model(input_ids=ids, attention_mask=mask)[0]

    elif header == "cbt":
        model_module = importlib.import_module(
            "backend.c3_text_stressor_distortion.app.cbt_model"
        )
        config = json.loads(app_config.CBT_BINARY_CONFIG_PATH.read_text(encoding="utf-8"))
        spec = config["models"][model_name]
        checkpoint_path = app_config.CBT_MODEL_DIR / spec["checkpoint_file"]
        runtime_hf_id = app_config.CBT_RUNTIME_HF_ID_OVERRIDES.get(
            model_name, spec.get("runtime_hf_id", spec["hf_id"])
        )
        max_length = int(spec["max_length"])
        threshold = float(spec["threshold"])
        device = model_module._resolve_device()
        state_dict = ensembles.load_state_dict(checkpoint_path, device)
        head_weight = state_dict.get("binary_head.fc1.weight")
        if head_weight is None:
            raise KeyError(f"{checkpoint_path} is missing 'binary_head.fc1.weight'")
        model = model_module._binary_model_class()(
            runtime_hf_id,
            dropout=float(spec.get("dropout", 0.3)),
            head_dropout=float(spec.get("head_dropout", 0.1)),
            intermediate=int(head_weight.shape[0]),
        )

        def logits_for(ids: Any, mask: Any) -> Any:
            return model(input_ids=ids, attention_mask=mask)

    else:
        raise ValueError(f"Unknown header {header!r}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(runtime_hf_id)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    def predict(text: str) -> ModelOutput:
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            logits = logits_for(ids, mask)
            probability = float(torch.softmax(logits, dim=1)[0, 1].item())
        return ModelOutput(int(probability >= threshold), probability)

    return LoadedPredictor(predict, (model, tokenizer))


def _ensemble_predictor(header: str) -> LoadedPredictor:
    _ensure_project_imports()
    if header == "stress":
        # Exercise the exact predictor used by the React diary backend. Its
        # config reproduces the research BERT + DeBERTa-v3 probability mean.
        module = importlib.import_module(
            "backend.c3_text_stressor_distortion.app.stress_model"
        )
        predictor = module.StressPredictor()

        def predict(text: str) -> ModelOutput:
            result = predictor.predict(text)
            return ModelOutput(
                int(result.is_stressed),
                float(result.probabilities["stressed"]),
            )

        resources = (predictor,)

    elif header == "cbt":
        module = importlib.import_module(
            "backend.c3_text_stressor_distortion.app.cbt_model"
        )
        predictor = module.CBTPredictor()

        def predict(text: str) -> ModelOutput:
            result = predictor.predict(text)
            return ModelOutput(
                int(result.has_distortion),
                float(result.probabilities["distortion"]),
            )

        resources = (predictor,)

    else:
        raise ValueError(f"Unknown header {header!r}")
    return LoadedPredictor(predict, resources)


def _baseline_predictor(header: str, model_name: str) -> LoadedPredictor:
    """Load one saved TF-IDF baseline and preserve its trained threshold."""
    if model_name not in ("Logistic Regression", "SVM"):
        raise ValueError(f"Unknown baseline {model_name!r}")
    key = "LR" if model_name == "Logistic Regression" else "SVM"
    if header == "stress":
        artifact_path = (
            PROJECT_ROOT
            / "models/c3_text_stressor_distortion/Stress header"
            / f"baseline_{key}.pkl"
        )
        fallback_threshold = 0.5
    elif header == "cbt":
        artifact_path = (
            PROJECT_ROOT
            / "models/c3_text_stressor_distortion/CBT header"
            / f"binary_baseline_{key}.pkl"
        )
        fallback_threshold = 0.5 if key == "LR" else 0.62
    else:
        raise ValueError(f"Unknown header {header!r}")
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing baseline artifact: {artifact_path}")

    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, dict):
        raise TypeError(f"Unsupported baseline artifact: {artifact_path}")
    vectorizer = artifact["vectorizer"]
    model = artifact["model"]
    threshold = float(artifact.get("binary_threshold", fallback_threshold))
    classes = [int(value) for value in getattr(model, "classes_", [0, 1])]
    if 1 not in classes:
        raise ValueError(f"{artifact_path} does not contain positive class 1")
    positive_index = classes.index(1)

    def predict(text: str) -> ModelOutput:
        features = vectorizer.transform([text])
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0][positive_index])
        elif hasattr(model, "decision_function"):
            score = float(model.decision_function(features)[0])
            probability = 1.0 / (1.0 + math.exp(-score))
        else:
            raise TypeError(f"{model_name} does not expose a probability score")
        return ModelOutput(int(probability >= threshold), probability)

    return LoadedPredictor(predict, (vectorizer, model))


def load_predictor(header: str, model_name: str) -> LoadedPredictor:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")
    if model_name == "Ensemble":
        return _ensemble_predictor(header)
    if model_name in ("Logistic Regression", "SVM"):
        return _baseline_predictor(header, model_name)
    return _single_member_predictor(header, model_name)


def build_model_test_case(
    *, header: str, model_name: str, cases_path: Path
) -> type[unittest.TestCase]:
    """Create a unittest class with one independently reported test per row."""
    cases = load_cases(cases_path)

    class SemanticModelTests(unittest.TestCase):
        predictor: LoadedPredictor

        @classmethod
        def setUpClass(cls) -> None:
            if RUN_MODEL_TESTS:
                cls.predictor = load_predictor(header, model_name)

        @classmethod
        def tearDownClass(cls) -> None:
            predictor = getattr(cls, "predictor", None)
            if predictor is not None:
                predictor.close()

    SemanticModelTests.__name__ = f"{model_name.replace('-', '').replace(' ', '')}{header.title()}Tests"
    SemanticModelTests.__qualname__ = SemanticModelTests.__name__
    SemanticModelTests.__doc__ = (
        f"Fifty labelled semantic cases for the {header} {model_name} predictor."
    )

    def make_test(case: ModelCase) -> Callable[[unittest.TestCase], None]:
        @unittest.skipUnless(
            RUN_MODEL_TESTS,
            "set RUN_C3_MODEL_TESTS=1 (or use run_model_tests.py) for model inference",
        )
        def test_method(self: unittest.TestCase) -> None:
            output = self.predictor.predict(case.text)
            self.assertTrue(
                math.isfinite(output.positive_probability),
                f"{case.case_id} returned a non-finite probability",
            )
            self.assertGreaterEqual(output.positive_probability, 0.0)
            self.assertLessEqual(output.positive_probability, 1.0)
            self.assertEqual(
                output.predicted_label,
                case.expected_label,
                (
                    f"{case.case_id} [{case.category}] expected {case.expected_label}, "
                    f"got {output.predicted_label} "
                    f"(positive_probability={output.positive_probability:.6f}). "
                    f"Rationale: {case.rationale}"
                ),
            )

        test_method.__name__ = f"test_{case.case_id.lower()}"
        test_method.__doc__ = f"{case.category}: {case.rationale}"
        return test_method

    for case in cases:
        setattr(SemanticModelTests, f"test_{case.case_id.lower()}", make_test(case))
    return SemanticModelTests
