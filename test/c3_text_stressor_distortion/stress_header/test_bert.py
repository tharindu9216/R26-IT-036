"""Fifty semantic cases for the Stress BERT checkpoint."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_case_runner import build_model_test_case  # noqa: E402


TestBERT = build_model_test_case(
    header="stress",
    model_name="BERT",
    cases_path=Path(__file__).with_name("cases.csv"),
)
