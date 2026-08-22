"""Fifty cases for the equal-weight BERT + DeBERTa-v3 Stress ensemble."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_case_runner import build_model_test_case  # noqa: E402


TestEnsemble = build_model_test_case(
    header="stress",
    model_name="Ensemble",
    cases_path=Path(__file__).with_name("cases.csv"),
)
