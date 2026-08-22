"""Fifty semantic cases for the Stress TF-IDF calibrated SVM baseline."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_case_runner import build_model_test_case  # noqa: E402


TestSVM = build_model_test_case(
    header="stress",
    model_name="SVM",
    cases_path=Path(__file__).with_name("cases.csv"),
)
