from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import RAW_FEATURES


def validate_input(df: pd.DataFrame, dataset_name: str) -> None:
    required = ["file_name", "actual_class"] + RAW_FEATURES
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )

    if df[RAW_FEATURES].isna().any().any():
        raise ValueError(
            f"{dataset_name} contains missing A/D/V values."
        )


def ravdess_speaker(file_name: str) -> str:
    """
    Example:
        03-01-05-02-01-02-05.wav -> Actor_05
    """
    parts = Path(str(file_name)).stem.split("-")

    if len(parts) != 7:
        raise ValueError(
            f"Unexpected RAVDESS filename format: {file_name}"
        )

    return f"Actor_{parts[-1]}"


def savee_speaker(file_name: str) -> str:
    """
    Example:
        DC_a01.wav -> DC
        JK_h02.wav -> JK
    """
    stem = Path(str(file_name)).stem

    if "_" not in stem:
        raise ValueError(
            f"Unexpected SAVEE filename format: {file_name}"
        )

    return stem.split("_", 1)[0]


def load_datasets(
    ravdess_csv: str,
    savee_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ravdess = pd.read_csv(ravdess_csv)
    savee = pd.read_csv(savee_csv)

    validate_input(ravdess, "RAVDESS")
    validate_input(savee, "SAVEE")

    ravdess["corpus"] = "RAVDESS"
    savee["corpus"] = "SAVEE"

    ravdess["speaker_id"] = ravdess["file_name"].map(
        ravdess_speaker
    )
    savee["speaker_id"] = savee["file_name"].map(
        savee_speaker
    )

    return ravdess, savee


def speaker_disjoint_split(
    ravdess_df: pd.DataFrame,
    holdout_size: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=holdout_size,
        random_state=seed,
    )

    train_idx, test_idx = next(
        splitter.split(
            ravdess_df,
            groups=ravdess_df["speaker_id"],
        )
    )

    reference = ravdess_df.iloc[train_idx].copy()
    heldout = ravdess_df.iloc[test_idx].copy()

    train_speakers = set(reference["speaker_id"])
    test_speakers = set(heldout["speaker_id"])

    assert train_speakers.isdisjoint(test_speakers)

    reference["split"] = "reference_train"
    heldout["split"] = "heldout"

    return reference, heldout
