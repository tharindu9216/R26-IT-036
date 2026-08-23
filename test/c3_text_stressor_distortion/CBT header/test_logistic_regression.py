"""Fifty semantic cases for the CBT TF-IDF Logistic Regression baseline."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_case_runner import build_model_test_case  # noqa: E402


TestLogisticRegression = build_model_test_case(
    header="cbt",
    model_name="Logistic Regression",
    cases_path=Path(__file__).with_name("cases.csv"),
)
