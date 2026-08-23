"""Fifty inference-contract cases for the research CBT BERTopic model."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topic_case_runner import build_bertopic_test_case  # noqa: E402


TestCBTBERTopic = build_bertopic_test_case(
    header="cbt",
    cases_path=Path(__file__).with_name("bertopic_cases.csv"),
)
