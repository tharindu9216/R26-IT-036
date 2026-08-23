from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPONENT_BACKEND_ROOT = Path(__file__).resolve().parents[1]

DEVICE = os.getenv("STRESS_DEVICE", "auto")

# Stress header — research probability ensemble selected by the training
# pipeline: BERT + DeBERTa-v3 with equal weights.
DATA_DIR = Path(os.getenv("STRESS_DATA_DIR", PROJECT_ROOT / "data/Stress header/processed"))
MODEL_DIR = Path(
    os.getenv(
        "STRESS_MODEL_DIR",
        PROJECT_ROOT / "models/c3_text_stressor_distortion/Stress header",
    )
)
METADATA_PATH = Path(os.getenv("STRESS_METADATA_PATH", DATA_DIR / "metadata.json"))

DEFAULT_HF_ID = "microsoft/deberta-v3-base"
DEFAULT_MAX_LEN = 192
DEFAULT_NUM_SUBREDDITS = 10

# Per-member checkpoints for the two transformers selected by the saved
# Stress training ensemble.
STRESS_ENSEMBLE_MEMBERS = {
    "BERT": {
        "checkpoint": Path(os.getenv("STRESS_BERT_CHECKPOINT", MODEL_DIR / "BERT_best.pt")),
        "hf_id": "bert-base-uncased",
        "runtime_hf_id": "bert-base-uncased",
        "max_len": 192,
        "weight": 0.5,
    },
    "DeBERTa-v3": {
        "checkpoint": Path(
            os.getenv("DEBERTA_CHECKPOINT", MODEL_DIR / "DeBERTa-v3_best.pt")
        ),
        "hf_id": "microsoft/deberta-v3-base",
        "runtime_hf_id": "microsoft/deberta-v3-base",
        "max_len": 192,
        "weight": 0.5,
    },
}
# This exactly reproduces STEP 4 in Stress header/train/train.py and the saved
# Ensemble_BERT_DeBERTa-v3 entry in all_results.json.
STRESS_DECISION_THRESHOLD = 0.5

# CBT header — 3-transformer ensemble (BERT + MentalBERT + DeBERTa-v3),
# weights + threshold come from the existing OOF-selected deployment config.
CBT_MODEL_DIR = Path(
    os.getenv(
        "CBT_MODEL_DIR",
        PROJECT_ROOT / "models/c3_text_stressor_distortion/CBT header",
    )
)
CBT_BINARY_CONFIG_PATH = Path(
    os.getenv("CBT_BINARY_CONFIG_PATH", CBT_MODEL_DIR / "binary_config.json")
)
CBT_TRAIN_MODEL_PATH = Path(
    os.getenv(
        "CBT_TRAIN_MODEL_PATH",
        PROJECT_ROOT
        / "ai_components/c3_text_stressor_distortion/CBT header/train/model.py",
    )
)
# MentalBERT is gated on HF — reuse the public BERT-base scaffold, same as
# the existing CBT XAI loader (ai_components/.../CBT header/xai/model_loader.py).
CBT_RUNTIME_HF_ID_OVERRIDES = {"MentalBERT": "bert-base-uncased"}

# Stress header — BERTopic life-theme tagging (work, relationships, health, ...).
# Live inference reuses the saved topic centroids rather than reloading the
# full BERTopic/UMAP/HDBSCAN pipeline — same approach already proven in
# streamlit_app/.../stressor head/app.py (predict_stress_topic).
STRESS_BERTOPIC_MODEL_DIR = Path(
    os.getenv("STRESS_BERTOPIC_MODEL_DIR", MODEL_DIR / "BERTopic")
)
STRESS_BERTOPIC_REPORT_DIR = Path(
    os.getenv(
        "STRESS_BERTOPIC_REPORT_DIR",
        PROJECT_ROOT / "reports/c3_text_stressor_distortion/Stress header/BERTopic",
    )
)
BERTOPIC_FRONTEND_PATH = Path(
    os.getenv(
        "BERTOPIC_FRONTEND_PATH",
        PROJECT_ROOT
        / "streamlit_app/c3_text_stressor_distortion/stressor head/bertopic_frontend.py",
    )
)
DEFAULT_BERTOPIC_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Diary persistence (plain sqlite3, single table).
DIARY_DB_PATH = Path(
    os.getenv("DIARY_DB_PATH", COMPONENT_BACKEND_ROOT / "diary.db")
)
