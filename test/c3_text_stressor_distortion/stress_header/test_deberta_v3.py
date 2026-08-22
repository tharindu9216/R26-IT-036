"""Fifty semantic cases for the Stress DeBERTa-v3 checkpoint."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_case_runner import build_model_test_case  # noqa: E402


TestDeBERTaV3 = build_model_test_case(
    header="stress",
    model_name="DeBERTa-v3",
    cases_path=Path(__file__).with_name("cases.csv"),
)
