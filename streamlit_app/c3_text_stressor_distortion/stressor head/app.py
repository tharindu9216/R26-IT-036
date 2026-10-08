from __future__ import annotations

import html
import importlib
import importlib.util
import json
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

# Multiple cached transformer families share a small consumer GPU in this app.
# Expandable segments reduce allocator fragmentation during gradient-based XAI.
# This must be configured before the first CUDA context is initialized.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

# Streamlit's test runner does not always add the script directory to sys.path,
# especially when the directory name contains a space.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from bertopic_frontend import (
    assess_topic_match,
    largest_topics,
    load_cbt_topic_catalog,
    load_topic_catalog,
    rank_topic_scores,
    resolve_embedding_topic_ids,
)


PROJECT_ROOT = APP_DIR.parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "c3_text_stressor_distortion" / "Stress header"
BERTOPIC_MODEL_DIR = MODEL_DIR / "BERTopic"
BERTOPIC_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "c3_text_stressor_distortion"
    / "Stress header"
    / "BERTopic"
)
ENV_PATH = PROJECT_ROOT / ".env"
XAI_DIR = PROJECT_ROOT / "ai_components" / "c3_text_stressor_distortion" / "Stress header" / "xai"
XAI_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "c3_text_stressor_distortion"
    / "Stress header"
    / "evaluation"
    / "xai_outputs"
    / "streamlit"
)
CBT_MODEL_DIR = (
    PROJECT_ROOT / "models" / "c3_text_stressor_distortion" / "CBT header"
)
CBT_CONFIG_PATH = CBT_MODEL_DIR / "binary_config.json"
CBT_BERTOPIC_MODEL_DIR = CBT_MODEL_DIR / "BERTopic"
CBT_BERTOPIC_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "c3_text_stressor_distortion"
    / "CBT header"
    / "BERTopic"
)
CBT_XAI_DIR = (
    PROJECT_ROOT
    / "ai_components"
    / "c3_text_stressor_distortion"
    / "CBT header"
    / "xai"
)
CBT_XAI_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "c3_text_stressor_distortion"
    / "CBT header"
    / "evaluation"
    / "xai_outputs"
    / "streamlit"
)

LABELS = {
    0: "Not Stressed",
    1: "Stressed",
}

SUBREDDIT_LABELS = {
    0: "almosthomeless",
    1: "anxiety",
    2: "assistance",
    3: "domesticviolence",
    4: "food_pantry",
    5: "homeless",
    6: "ptsd",
    7: "relationships",
    8: "stress",
    9: "survivorsofabuse",
}

TRANSFORMER_MODELS = {
    "BERT": {
        "hf_id": "bert-base-uncased",
        "checkpoint": "BERT_final.pt",
        "max_len": 192,
    },
    "MentalBERT": {
        "hf_id": "mental/mental-bert-base-uncased",
        "runtime_hf_id": "bert-base-uncased",
        "checkpoint": "MentalBERT_final.pt",
        "max_len": 192,
    },
    "DeBERTa-v3": {
        "hf_id": "microsoft/deberta-v3-base",
        "checkpoint": "DeBERTa-v3_final.pt",
        "max_len": 192,
    },
}

BASELINE_MODELS = {
    "TF-IDF + Logistic Regression": "baseline_LR.pkl",
    "TF-IDF + SVM": "baseline_SVM.pkl",
}


def load_env_value(key: str) -> str | None:
    if os.getenv(key):
        return os.getenv(key)
    if not ENV_PATH.exists():
        return None

    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'") or None
    return None


def hf_auth_kwargs() -> dict[str, str]:
    token = load_env_value("HF_TOKEN")
    return {"token": token} if token else {}


@dataclass
class PredictionResult:
    model: str
    model_type: str
    prediction: str
    stress_probability: float | None
    confidence: float | None
    subreddit_category: str
    status: str


@dataclass
class CBTPredictionResult:
    model: str
    model_type: str
    prediction: str
    distortion_probability: float | None
    confidence: float | None
    threshold: float | None
    status: str


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"/?r/[A-Za-z0-9_]+|/?u/[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"&#x200B;|\u200b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def expand_common_contractions(text: str) -> str:
    replacements = {
        "can't": "can not",
        "cannot": "can not",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'t": " not",
        "'ve": " have",
        "'m": " am",
    }
    lowered = text.lower()
    for src, dst in replacements.items():
        lowered = lowered.replace(src, dst)
    return lowered


def preprocess_for_transformers(text: str) -> str:
    return expand_common_contractions(clean_text(text))


def unpack_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_local_module(module_name: str, path: Path):
    """Import a local module by path without relying on space-containing packages."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import local module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@st.cache_resource(show_spinner=False)
def load_transformer_model(model_name: str):
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    class ClassificationHead(nn.Module):
        def __init__(self, hidden_size: int, intermediate: int, num_classes: int):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, intermediate)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(0.1)
            self.fc2 = nn.Linear(intermediate, num_classes)

        def forward(self, x):
            return self.fc2(self.dropout(self.act(self.fc1(x))))

    class DualHeadStressModel(nn.Module):
        def __init__(self, hf_id: str, auth_kwargs: dict[str, str]):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(hf_id, **auth_kwargs)
            hidden = self.encoder.config.hidden_size
            self.layer_norm = nn.LayerNorm(hidden)
            self.dropout = nn.Dropout(0.3)
            self.head_1a = ClassificationHead(hidden, 256, 2)
            self.head_1b = ClassificationHead(hidden, 256, 10)

        def forward(self, input_ids, attention_mask):
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = output.last_hidden_state[:, 0, :]
            cls = self.dropout(self.layer_norm(cls))
            return self.head_1a(cls), self.head_1b(cls)

    spec = TRANSFORMER_MODELS[model_name]
    runtime_hf_id = spec.get("runtime_hf_id", spec["hf_id"])
    checkpoint_path = MODEL_DIR / spec["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    # Cache checkpoint weights on CPU. Callers move one model to CUDA only for
    # the duration of their prediction/XAI action and offload it afterward.
    device = torch.device("cpu")
    auth_kwargs = hf_auth_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(runtime_hf_id, **auth_kwargs)
    model = DualHeadStressModel(runtime_hf_id, auth_kwargs).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = strip_module_prefix(unpack_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return tokenizer, model, device, spec["max_len"]


@st.cache_resource(show_spinner=False)
def load_baseline_model(file_name: str):
    path = MODEL_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline model: {path}")
    with path.open("rb") as file:
        data = pickle.load(file)
    return data["vectorizer"], data["model"]


@st.cache_data(show_spinner=False)
def load_cbt_config() -> dict[str, Any]:
    if not CBT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing CBT binary config: {CBT_CONFIG_PATH}")
    return json.loads(CBT_CONFIG_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_cbt_transformer_resources(model_name: str):
    loader = load_local_module(
        "streamlit_cbt_model_loader",
        CBT_XAI_DIR / "model_loader.py",
    )
    # Persist cached CBT weights on CPU; CUDA is borrowed only per action.
    return loader.load_cbt_xai_resources(model_name, device="cpu")


@st.cache_resource(show_spinner=False)
def load_cbt_baseline_model(file_name: str):
    path = CBT_MODEL_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing CBT baseline model: {path}")
    with path.open("rb") as file:
        artifact = pickle.load(file)
    return artifact


def preprocess_for_cbt(text: str, model_name: str) -> str:
    preprocessing = load_local_module(
        "streamlit_cbt_preprocessing",
        CBT_XAI_DIR / "preprocessing.py",
    )
    return preprocessing.preprocess_for_model(text, model_name)


@st.cache_resource(show_spinner=False)
def load_topic_embedding_model(model_id: str):
    from sentence_transformers import SentenceTransformer

    # BERTopic inference encodes one short text at a time. Keeping this encoder
    # on CPU reserves limited CUDA memory for transformer predictions and XAI.
    return SentenceTransformer(model_id, device="cpu")


@st.cache_data(show_spinner=False)
def load_stress_topic_catalog() -> dict[str, Any]:
    return load_topic_catalog(BERTOPIC_REPORT_DIR)


@st.cache_resource(show_spinner=False)
def load_stress_topic_resources():
    """Load the configured encoder and saved Stress topic embeddings."""
    import json

    import numpy as np
    from safetensors.numpy import load_file

    required = (
        BERTOPIC_MODEL_DIR / "config.json",
        BERTOPIC_MODEL_DIR / "topic_embeddings.safetensors",
        BERTOPIC_REPORT_DIR / "topic_info.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Stress BERTopic artifacts: " + ", ".join(missing))

    config = json.loads(
        (BERTOPIC_MODEL_DIR / "config.json").read_text())
    embedding_model_id = config.get(
        "embedding_model",
        "sentence-transformers/all-mpnet-base-v2",
    )
    embedding_model = load_topic_embedding_model(embedding_model_id)

    tensors = load_file(
        BERTOPIC_MODEL_DIR / "topic_embeddings.safetensors")
    topic_embeddings = np.asarray(
        tensors["topic_embeddings"], dtype=np.float32)
    norms = np.linalg.norm(topic_embeddings, axis=1, keepdims=True)
    topic_embeddings = topic_embeddings / np.clip(norms, 1e-12, None)

    catalog = load_stress_topic_catalog()
    topic_ids = resolve_embedding_topic_ids(
        catalog["topics"], len(topic_embeddings))
    return embedding_model, topic_embeddings, topic_ids


def predict_stress_topic(text: str) -> dict[str, Any]:
    """Assign text to the nearest saved non-outlier topic embedding."""
    import numpy as np

    embedding_model, topic_embeddings, topic_ids = (
        load_stress_topic_resources())
    document_embedding = embedding_model.encode(
        [clean_text(text)],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    similarities = np.matmul(topic_embeddings, document_embedding)
    ranked = rank_topic_scores(
        topic_ids,
        similarities.tolist(),
        top_n=5,
    )
    if not ranked:
        raise RuntimeError("No non-outlier topics are available")

    catalog = load_stress_topic_catalog()
    topic = catalog["topics"][ranked[0]["topic_id"]]
    alternatives = []
    for rank, candidate in enumerate(ranked[1:], start=2):
        candidate_topic = catalog["topics"][candidate["topic_id"]]
        alternatives.append({
            "Rank": rank,
            "Topic": candidate["topic_id"],
            "Theme": candidate_topic["display_name"],
            "Cosine similarity": candidate["similarity"],
        })

    subreddit_total = sum(topic["subreddit_counts"].values())
    top_subreddits = sorted(
        topic["subreddit_counts"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    return {
        "topic_id": topic["topic_id"],
        "theme": topic["display_name"],
        "similarity": ranked[0]["similarity"],
        "match": assess_topic_match(ranked),
        "training_documents": topic["training_documents"],
        "assigned_documents": topic["assigned_documents"],
        "terms": topic["terms"],
        "stress_distribution": topic["stress_distribution"],
        "top_subreddits": [
            {
                "Subreddit": subreddit,
                "Documents": count,
                "Share within topic": (
                    count / subreddit_total if subreddit_total else 0.0),
            }
            for subreddit, count in top_subreddits
        ],
        "alternatives": alternatives,
    }


@st.cache_data(show_spinner=False)
def load_cbt_topic_catalog_cached() -> dict[str, Any]:
    return load_cbt_topic_catalog(CBT_BERTOPIC_REPORT_DIR)


@st.cache_resource(show_spinner=False)
def load_cbt_topic_resources():
    """Load the configured encoder and saved CBT topic embeddings."""
    import numpy as np
    from safetensors.numpy import load_file

    required = (
        CBT_BERTOPIC_MODEL_DIR / "config.json",
        CBT_BERTOPIC_MODEL_DIR / "topic_embeddings.safetensors",
        CBT_BERTOPIC_REPORT_DIR / "topic_info.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing CBT BERTopic artifacts: " + ", ".join(missing))

    config = json.loads(
        (CBT_BERTOPIC_MODEL_DIR / "config.json").read_text())
    embedding_model_id = config.get(
        "embedding_model",
        "sentence-transformers/all-mpnet-base-v2",
    )
    embedding_model = load_topic_embedding_model(embedding_model_id)
    tensors = load_file(
        CBT_BERTOPIC_MODEL_DIR / "topic_embeddings.safetensors")
    topic_embeddings = np.asarray(
        tensors["topic_embeddings"], dtype=np.float32)
    norms = np.linalg.norm(topic_embeddings, axis=1, keepdims=True)
    topic_embeddings = topic_embeddings / np.clip(norms, 1e-12, None)

    catalog = load_cbt_topic_catalog_cached()
    topic_ids = resolve_embedding_topic_ids(
        catalog["topics"], len(topic_embeddings))
    return embedding_model, topic_embeddings, topic_ids


def release_cached_gpu_resources() -> None:
    """Drop cached neural models so the next GPU task starts cleanly."""
    import gc

    # Clear every Streamlit resource cache, including entries created by older
    # versions of this script that a per-function clear can no longer reach.
    clear_all = getattr(st.cache_resource, "clear", None)
    if clear_all is not None:
        clear_all()

    # Clear outer BERTopic resources before the shared embedding encoder that
    # they reference. Classifier checkpoints are reloaded lazily on demand.
    cached_loaders = (
        load_stress_topic_resources,
        load_cbt_topic_resources,
        load_topic_embedding_model,
        load_transformer_model,
        load_cbt_transformer_resources,
    )
    for loader in cached_loaders:
        clear = getattr(loader, "clear", None)
        if clear is not None:
            clear()
    gc.collect()

    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Release CUDA IPC allocations when supported. This is best-effort;
        # some local CUDA builds do not expose an active IPC pool.
        try:
            torch.cuda.ipc_collect()
        except (RuntimeError, AttributeError):
            pass


def preferred_torch_device():
    """Return CUDA when available, otherwise CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def offload_model_from_gpu(model: Any | None) -> None:
    """Move a completed action's model to CPU and release CUDA allocations."""
    if model is None:
        return
    import gc
    import torch

    try:
        model.to("cpu")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def predict_cbt_topic(text: str) -> dict[str, Any]:
    """Assign text to its closest saved non-outlier CBT topic."""
    import numpy as np

    embedding_model, topic_embeddings, topic_ids = load_cbt_topic_resources()
    document_embedding = embedding_model.encode(
        [clean_text(text)],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    similarities = np.matmul(topic_embeddings, document_embedding)
    ranked = rank_topic_scores(
        topic_ids,
        similarities.tolist(),
        top_n=5,
    )
    if not ranked:
        raise RuntimeError("No non-outlier CBT topics are available")

    catalog = load_cbt_topic_catalog_cached()
    topic = catalog["topics"][ranked[0]["topic_id"]]
    alternatives = []
    for rank, candidate in enumerate(ranked[1:], start=2):
        candidate_topic = catalog["topics"][candidate["topic_id"]]
        alternatives.append({
            "Rank": rank,
            "Topic": candidate["topic_id"],
            "Theme": candidate_topic["display_name"],
            "Cosine similarity": candidate["similarity"],
        })

    type_rows = [
        {
            "Distortion type": label,
            "Documents": values["documents"],
            "Share within topic": values["percentage"],
        }
        for label, values in topic[
            "distortion_type_distribution"
        ].items()
    ]
    type_rows.sort(key=lambda row: row["Documents"], reverse=True)
    return {
        "topic_id": topic["topic_id"],
        "theme": topic["display_name"],
        "similarity": ranked[0]["similarity"],
        "match": assess_topic_match(ranked),
        "training_documents": topic["training_documents"],
        "assigned_documents": topic["assigned_documents"],
        "terms": topic["terms"],
        "distortion_status_distribution": topic[
            "distortion_status_distribution"
        ],
        "distortion_type_distribution": type_rows,
        "alternatives": alternatives,
    }


def load_xai_classes():
    if str(XAI_DIR) not in sys.path:
        sys.path.insert(0, str(XAI_DIR))
    rationale_module = importlib.import_module("rationale_extractor")
    importlib.reload(rationale_module)
    ig_module = importlib.import_module("integrated_gradients")
    shap_module = importlib.import_module("shap_explainer")
    lime_module = importlib.import_module("lime_explainer")
    combined_module = importlib.import_module("combined_explainer")
    counterfactual_module = importlib.import_module("counterfactual_explainer")
    importlib.reload(ig_module)
    importlib.reload(shap_module)
    importlib.reload(lime_module)
    importlib.reload(combined_module)
    importlib.reload(counterfactual_module)

    return (
        ig_module.StressIG,
        shap_module.StressSHAP,
        lime_module.StressLIME,
        combined_module.StressCombinedXAI,
        counterfactual_module.StressCounterfactual,
    )


def load_cbt_xai_classes():
    module = load_local_module(
        "streamlit_cbt_shared_explainers",
        CBT_XAI_DIR / "shared_explainers.py",
    )
    return (
        module.CBTIG,
        module.CBTSHAP,
        module.CBTLIME,
        module.CBTCombinedXAI,
        module.CBTCounterfactual,
    )


def predict_transformer(model_name: str, text: str) -> PredictionResult:
    model = None
    try:
        import torch

        tokenizer, model, _, max_len = load_transformer_model(model_name)
        device = preferred_torch_device()
        model.to(device).eval()
        prepared_text = preprocess_for_transformers(text)
        encoded = tokenizer(
            prepared_text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            stress_logits, subreddit_logits = model(
                encoded["input_ids"],
                encoded["attention_mask"],
            )
            stress_probs = torch.softmax(stress_logits, dim=1)[0].detach().cpu()
            subreddit_probs = torch.softmax(subreddit_logits, dim=1)[0].detach().cpu()

        label_id = int(torch.argmax(stress_probs).item())
        subreddit_id = int(torch.argmax(subreddit_probs).item())
        stress_probability = float(stress_probs[1].item())
        confidence = float(stress_probs[label_id].item())
        subreddit_category = (
            SUBREDDIT_LABELS.get(subreddit_id, "Unknown")
            if label_id == 1
            else "N/A"
        )

        return PredictionResult(
            model=model_name,
            model_type="Transformer",
            prediction=LABELS[label_id],
            stress_probability=stress_probability,
            confidence=confidence,
            subreddit_category=subreddit_category,
            status="OK",
        )
    except Exception as exc:
        return PredictionResult(
            model=model_name,
            model_type="Transformer",
            prediction="Unavailable",
            stress_probability=None,
            confidence=None,
            subreddit_category="-",
            status=f"{type(exc).__name__}: {exc}",
        )
    finally:
        offload_model_from_gpu(model)


def predict_baseline(model_name: str, file_name: str, text: str) -> PredictionResult:
    try:
        vectorizer, model = load_baseline_model(file_name)
        prepared_text = clean_text(text)
        features = vectorizer.transform([prepared_text])
        label_id = int(model.predict(features)[0])

        stress_probability = None
        if hasattr(model, "predict_proba"):
            stress_probability = float(model.predict_proba(features)[0][1])
        elif hasattr(model, "decision_function"):
            score = float(model.decision_function(features)[0])
            stress_probability = 1.0 / (1.0 + pow(2.718281828, -score))

        confidence = None
        if stress_probability is not None:
            confidence = max(stress_probability, 1.0 - stress_probability)

        return PredictionResult(
            model=model_name,
            model_type="Baseline",
            prediction=LABELS.get(label_id, str(label_id)),
            stress_probability=stress_probability,
            confidence=confidence,
            subreddit_category="N/A",
            status="OK",
        )
    except Exception as exc:
        return PredictionResult(
            model=model_name,
            model_type="Baseline",
            prediction="Unavailable",
            stress_probability=None,
            confidence=None,
            subreddit_category="-",
            status=f"{type(exc).__name__}: {exc}",
        )


def run_all_models(text: str) -> list[PredictionResult]:
    release_cached_gpu_resources()
    results = []
    for model_name in TRANSFORMER_MODELS:
        results.append(predict_transformer(model_name, text))
    for model_name, file_name in BASELINE_MODELS.items():
        results.append(predict_baseline(model_name, file_name, text))
    return results


def predict_cbt_transformer(
    model_name: str,
    text: str,
) -> CBTPredictionResult:
    resources = None
    try:
        import torch

        resources = load_cbt_transformer_resources(model_name)
        device = preferred_torch_device()
        resources.model.to(device).eval()
        prepared_text = preprocess_for_cbt(text, model_name)
        encoded = resources.tokenizer(
            prepared_text,
            max_length=resources.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            logits, _ = resources.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            probabilities = torch.softmax(
                logits, dim=1)[0].detach().cpu()

        distortion_probability = float(probabilities[1])
        label_id = int(
            distortion_probability >= resources.decision_threshold)
        return CBTPredictionResult(
            model=model_name,
            model_type="Transformer",
            prediction=(
                "Distortion" if label_id == 1 else "No Distortion"),
            distortion_probability=distortion_probability,
            confidence=float(probabilities[label_id]),
            threshold=resources.decision_threshold,
            status="OK",
        )
    except Exception as exc:
        return CBTPredictionResult(
            model=model_name,
            model_type="Transformer",
            prediction="Unavailable",
            distortion_probability=None,
            confidence=None,
            threshold=None,
            status=f"{type(exc).__name__}: {exc}",
        )
    finally:
        offload_model_from_gpu(
            resources.model if resources is not None else None)


def predict_cbt_baseline(
    model_key: str,
    model_name: str,
    file_name: str,
    text: str,
) -> CBTPredictionResult:
    try:
        artifact = load_cbt_baseline_model(file_name)
        vectorizer = artifact["vectorizer"]
        model = artifact["model"]
        features = vectorizer.transform([text])
        threshold = float(
            artifact.get(
                "binary_threshold",
                load_cbt_config()["research_baselines"][model_key][
                    "threshold"
                ],
            )
        )
        if hasattr(model, "predict_proba"):
            class_order = list(getattr(model, "classes_", [0, 1]))
            positive_index = class_order.index(1)
            distortion_probability = float(
                model.predict_proba(features)[0][positive_index])
        elif hasattr(model, "decision_function"):
            score = float(model.decision_function(features)[0])
            distortion_probability = 1.0 / (1.0 + pow(2.718281828, -score))
        else:
            raise TypeError("CBT baseline does not expose probability scores")

        label_id = int(distortion_probability >= threshold)
        confidence = (
            distortion_probability
            if label_id == 1
            else 1.0 - distortion_probability
        )
        return CBTPredictionResult(
            model=model_name,
            model_type="Baseline",
            prediction=(
                "Distortion" if label_id == 1 else "No Distortion"),
            distortion_probability=distortion_probability,
            confidence=confidence,
            threshold=threshold,
            status="OK",
        )
    except Exception as exc:
        return CBTPredictionResult(
            model=model_name,
            model_type="Baseline",
            prediction="Unavailable",
            distortion_probability=None,
            confidence=None,
            threshold=None,
            status=f"{type(exc).__name__}: {exc}",
        )


def run_cbt_models(text: str) -> list[CBTPredictionResult]:
    release_cached_gpu_resources()
    config = load_cbt_config()
    results = [
        predict_cbt_transformer(model_name, text)
        for model_name in config["models"]
    ]
    for model_key, spec in config["research_baselines"].items():
        results.append(
            predict_cbt_baseline(
                model_key,
                spec["name"],
                spec["checkpoint_file"],
                text,
            )
        )
    return results


def cbt_result_rows(
    results: list[CBTPredictionResult],
) -> list[dict[str, Any]]:
    return [
        {
            "Model": result.model,
            "Type": result.model_type,
            "Prediction": result.prediction,
            "Distortion Probability": result.distortion_probability,
            "Decision Threshold": result.threshold,
            "Confidence": result.confidence,
            "Status": result.status,
        }
        for result in results
    ]


def cbt_comparison_stats(
    results: list[CBTPredictionResult],
) -> dict[str, Any]:
    usable = [
        result
        for result in results
        if result.status == "OK" and result.confidence is not None
    ]
    if not usable:
        return {
            "best": None,
            "majority": "No result",
            "distortion_votes": 0,
            "no_distortion_votes": 0,
            "completed": 0,
        }
    distortion_votes = sum(
        result.prediction == "Distortion" for result in usable)
    no_distortion_votes = sum(
        result.prediction == "No Distortion" for result in usable)
    majority = (
        "Tie"
        if distortion_votes == no_distortion_votes
        else (
            "Distortion"
            if distortion_votes > no_distortion_votes
            else "No Distortion"
        )
    )
    return {
        "best": max(usable, key=lambda item: item.confidence or 0.0),
        "majority": majority,
        "distortion_votes": distortion_votes,
        "no_distortion_votes": no_distortion_votes,
        "completed": len(usable),
    }


def result_rows(results: list[PredictionResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        rows.append(
            {
                "Model": result.model,
                "Type": result.model_type,
                "Prediction": result.prediction,
                "Stress Probability": result.stress_probability,
                "Confidence": result.confidence,
                "Stress Category": result.subreddit_category,
                "Status": result.status,
            }
        )
    return rows


def summarize_results(results: list[PredictionResult]) -> tuple[str, str]:
    usable = [item for item in results if item.status == "OK" and item.confidence is not None]
    if not usable:
        return "No model completed successfully.", "Install dependencies and confirm model artifacts exist."

    best = max(usable, key=lambda item: item.confidence or 0.0)
    stressed_votes = sum(item.prediction == "Stressed" for item in usable)
    not_stressed_votes = sum(item.prediction == "Not Stressed" for item in usable)
    majority = "Stressed" if stressed_votes > not_stressed_votes else "Not Stressed"
    if stressed_votes == not_stressed_votes:
        majority = "Tie"

    summary = f"Highest-confidence model: {best.model} ({best.prediction}, confidence {best.confidence:.3f})."
    vote_text = f"Model vote: {stressed_votes} stressed, {not_stressed_votes} not stressed. Majority: {majority}."
    return summary, vote_text


def comparison_stats(results: list[PredictionResult]) -> dict[str, Any]:
    usable = [item for item in results if item.status == "OK" and item.confidence is not None]
    if not usable:
        return {
            "best": None,
            "majority": "No result",
            "stressed_votes": 0,
            "not_stressed_votes": 0,
            "completed": 0,
        }

    best = max(usable, key=lambda item: item.confidence or 0.0)
    stressed_votes = sum(item.prediction == "Stressed" for item in usable)
    not_stressed_votes = sum(item.prediction == "Not Stressed" for item in usable)
    if stressed_votes == not_stressed_votes:
        majority = "Tie"
    else:
        majority = "Stressed" if stressed_votes > not_stressed_votes else "Not Stressed"

    return {
        "best": best,
        "majority": majority,
        "stressed_votes": stressed_votes,
        "not_stressed_votes": not_stressed_votes,
        "completed": len(usable),
    }


def percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def css_class(value: str) -> str:
    normalized = value.lower().replace(" ", "-")
    return re.sub(r"[^a-z0-9-]", "", normalized)


def render_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }
        h1 {
            letter-spacing: 0 !important;
        }
        .app-kicker {
            color: #8b95a7;
            font-size: 0.92rem;
            margin-bottom: 1.8rem;
        }
        .section-label {
            color: #aab3c2;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin: 1.25rem 0 0.6rem;
            text-transform: uppercase;
        }
        .result-card {
            border: 1px solid #293241;
            background: #121722;
            border-radius: 8px;
            padding: 1rem;
            min-height: 188px;
            margin-bottom: 1rem;
        }
        .result-head {
            align-items: center;
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.8rem;
        }
        .model-name {
            color: #f4f7fb;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.25;
        }
        .model-type {
            color: #8b95a7;
            font-size: 0.78rem;
            margin-top: 0.1rem;
        }
        .pill {
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 800;
            padding: 0.28rem 0.55rem;
            white-space: nowrap;
        }
        .pill.stressed {
            background: #3a171d;
            color: #ff7f8a;
        }
        .pill.not-stressed {
            background: #102b22;
            color: #65d69b;
        }
        .pill.distortion {
            background: #3a171d;
            color: #ff7f8a;
        }
        .pill.no-distortion {
            background: #102b22;
            color: #65d69b;
        }
        .pill.unavailable {
            background: #2b2434;
            color: #c7a5ff;
        }
        .metric-grid {
            display: grid;
            gap: 0.65rem;
            grid-template-columns: 1fr 1fr;
            margin-top: 0.6rem;
        }
        .mini-metric {
            background: #0d1119;
            border: 1px solid #242b38;
            border-radius: 7px;
            padding: 0.65rem;
        }
        .metric-label {
            color: #8b95a7;
            font-size: 0.72rem;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            color: #f4f7fb;
            font-size: 1.05rem;
            font-weight: 800;
        }
        .category-line {
            color: #aab3c2;
            font-size: 0.82rem;
            margin-top: 0.7rem;
        }
        .status-ok {
            color: #65d69b;
        }
        .status-failed {
            color: #ff7f8a;
        }
        .topic-theme {
            color: #f4f7fb;
            font-size: 1.25rem;
            font-weight: 800;
            margin: 0.25rem 0 0.9rem;
        }
        .topic-note {
            border-left: 3px solid #4e81ee;
            color: #aab3c2;
            font-size: 0.84rem;
            margin: 0.7rem 0 1rem;
            padding: 0.2rem 0 0.2rem 0.8rem;
        }
        div[data-testid="stMetric"] {
            background: #121722;
            border: 1px solid #293241;
            border-radius: 8px;
            padding: 0.95rem 1rem;
        }
        div[data-testid="stMetricLabel"] p {
            color: #8b95a7;
            font-size: 0.78rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.45rem;
        }
        .stTextArea textarea {
            border-radius: 7px;
            min-height: 142px;
        }
        .stButton > button {
            border-radius: 7px;
            font-weight: 800;
            height: 2.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_result_cards(results: list[PredictionResult]) -> None:
    columns = st.columns(3)
    for index, result in enumerate(results):
        prediction_class = css_class(result.prediction)
        status_class = "status-ok" if result.status == "OK" else "status-failed"
        status_text = "Ready" if result.status == "OK" else "Needs attention"
        card = f"""
        <div class="result-card">
            <div class="result-head">
                <div>
                    <div class="model-name">{html.escape(result.model)}</div>
                    <div class="model-type">{html.escape(result.model_type)}</div>
                </div>
                <div class="pill {prediction_class}">{html.escape(result.prediction)}</div>
            </div>
            <div class="metric-grid">
                <div class="mini-metric">
                    <div class="metric-label">Stress probability</div>
                    <div class="metric-value">{percent(result.stress_probability)}</div>
                </div>
                <div class="mini-metric">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{percent(result.confidence)}</div>
                </div>
            </div>
            <div class="category-line">Category: {html.escape(result.subreddit_category)}</div>
            <div class="category-line {status_class}">Status: {html.escape(status_text)}</div>
        </div>
        """
        columns[index % 3].markdown(card, unsafe_allow_html=True)


def render_artifacts() -> None:
    with st.expander("Model artifacts and environment", expanded=False):
        st.write(f"Model folder: `{MODEL_DIR}`")
        st.write(f"HF token loaded: `{'Yes' if load_env_value('HF_TOKEN') else 'No'}`")
        artifact_rows = []
        for model_name, spec in TRANSFORMER_MODELS.items():
            artifact_rows.append(
                {
                    "Model": model_name,
                    "File": spec["checkpoint"],
                    "Exists": (MODEL_DIR / spec["checkpoint"]).exists(),
                }
            )
        for model_name, file_name in BASELINE_MODELS.items():
            artifact_rows.append(
                {
                    "Model": model_name,
                    "File": file_name,
                    "Exists": (MODEL_DIR / file_name).exists(),
                }
            )
        artifact_rows.extend([
            {
                "Model": "Stress BERTopic",
                "File": "BERTopic/topic_embeddings.safetensors",
                "Exists": (
                    BERTOPIC_MODEL_DIR
                    / "topic_embeddings.safetensors"
                ).exists(),
            },
            {
                "Model": "Stress BERTopic reports",
                "File": "reports/.../Stress header/BERTopic/topic_info.csv",
                "Exists": (
                    BERTOPIC_REPORT_DIR / "topic_info.csv").exists(),
            },
        ])
        st.dataframe(artifact_rows, use_container_width=True, hide_index=True)


def render_cbt_result_cards(results: list[CBTPredictionResult]) -> None:
    columns = st.columns(3)
    for index, result in enumerate(results):
        prediction_class = css_class(result.prediction)
        status_class = "status-ok" if result.status == "OK" else "status-failed"
        status_text = "Ready" if result.status == "OK" else "Needs attention"
        card = f"""
        <div class="result-card">
            <div class="result-head">
                <div>
                    <div class="model-name">{html.escape(result.model)}</div>
                    <div class="model-type">{html.escape(result.model_type)}</div>
                </div>
                <div class="pill {prediction_class}">{html.escape(result.prediction)}</div>
            </div>
            <div class="metric-grid">
                <div class="mini-metric">
                    <div class="metric-label">Distortion probability</div>
                    <div class="metric-value">{percent(result.distortion_probability)}</div>
                </div>
                <div class="mini-metric">
                    <div class="metric-label">Decision threshold</div>
                    <div class="metric-value">{percent(result.threshold)}</div>
                </div>
            </div>
            <div class="category-line">Selected-class probability: {percent(result.confidence)}</div>
            <div class="category-line {status_class}">Status: {html.escape(status_text)}</div>
        </div>
        """
        columns[index % 3].markdown(card, unsafe_allow_html=True)


def render_cbt_artifacts() -> None:
    with st.expander("CBT model artifacts and environment", expanded=False):
        st.write(f"Model folder: `{CBT_MODEL_DIR}`")
        st.write(
            f"HF token loaded: `{'Yes' if load_env_value('HF_TOKEN') else 'No'}`"
        )
        artifact_rows = []
        try:
            config = load_cbt_config()
            for model_name, spec in config["models"].items():
                artifact_rows.append({
                    "Model": model_name,
                    "File": spec["checkpoint_file"],
                    "Exists": (
                        CBT_MODEL_DIR / spec["checkpoint_file"]
                    ).exists(),
                })
            for spec in config["research_baselines"].values():
                artifact_rows.append({
                    "Model": spec["name"],
                    "File": spec["checkpoint_file"],
                    "Exists": (
                        CBT_MODEL_DIR / spec["checkpoint_file"]
                    ).exists(),
                })
        except Exception as exc:
            st.error(f"Could not read CBT model configuration: {exc}")
        artifact_rows.extend([
            {
                "Model": "CBT BERTopic",
                "File": "BERTopic/topic_embeddings.safetensors",
                "Exists": (
                    CBT_BERTOPIC_MODEL_DIR
                    / "topic_embeddings.safetensors"
                ).exists(),
            },
            {
                "Model": "CBT BERTopic reports",
                "File": "reports/.../CBT header/BERTopic/topic_info.csv",
                "Exists": (
                    CBT_BERTOPIC_REPORT_DIR / "topic_info.csv"
                ).exists(),
            },
        ])
        st.dataframe(artifact_rows, use_container_width=True, hide_index=True)


def render_topic_match_summary(
        result: dict[str, Any], weak_heading: str | None = None,
) -> dict[str, Any]:
    """Explain the single primary candidate and its ranking uncertainty."""
    match = result.get("match")
    if match is None:
        # Preserve a useful display across Streamlit hot reloads when session
        # state still contains a result produced by the previous UI version.
        ranking = [{
            "topic_id": result["topic_id"],
            "similarity": result["similarity"],
        }]
        seen_topic_ids = {int(result["topic_id"])}
        for candidate in result.get("alternatives", []):
            topic_id = int(candidate["Topic"])
            if topic_id in seen_topic_ids:
                continue
            seen_topic_ids.add(topic_id)
            ranking.append({
                "topic_id": topic_id,
                "similarity": candidate["Cosine similarity"],
            })
        match = assess_topic_match(ranking)
    hide_weak_candidate = match["is_weak"] and weak_heading is not None
    if hide_weak_candidate:
        st.markdown(
            f'<div class="topic-theme">{html.escape(weak_heading)}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f'Nearest candidate, not assigned: Topic {result["topic_id"]} — '
            f'{result["theme"]}'
        )
    else:
        heading = (
            "Primary semantic topic"
            if match["is_clear"]
            else "Closest topic candidate"
        )
        st.markdown(
            f'<div class="topic-theme">{heading} — Topic '
            f'{result["topic_id"]}: '
            f'{html.escape(result["theme"])}</div>',
            unsafe_allow_html=True,
        )

    if match["is_clear"]:
        st.success(match["explanation"])
    else:
        st.warning(match["explanation"])

    metric_columns = st.columns(4)
    metric_columns[0].metric("Match interpretation", match["status"])
    metric_columns[1].metric(
        "Semantic similarity", f'{result["similarity"]:.3f}')
    margin = match["top_two_margin"]
    metric_columns[2].metric(
        "Top-two score gap",
        f"{margin:.3f}" if margin is not None else "N/A",
    )
    metric_columns[3].metric("Candidate topic ID", result["topic_id"])

    st.markdown(
        '<div class="topic-note"><strong>How to read this:</strong> This '
        'interface selects one primary topic candidate. The topics listed '
        'below are ranked alternatives, not additional assignments. Semantic '
        'similarity is a ranking score—not a probability, classifier '
        'confidence, or diagnosis.</div>',
        unsafe_allow_html=True,
    )
    term_label = (
        "Nearest candidate terms" if hide_weak_candidate else "Top topic terms"
    )
    st.write(f"**{term_label}:** " + ", ".join(result["terms"][:10]))

    st.markdown("**Other close themes — not additional assignments**")
    alternative_rows = [
        candidate
        for candidate in result.get("alternatives", [])
        if int(candidate["Topic"]) != int(result["topic_id"])
    ]
    if alternative_rows:
        st.dataframe(
            alternative_rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cosine similarity": st.column_config.NumberColumn(
                    "Semantic similarity", format="%.3f"),
            },
        )
    else:
        st.caption("No alternative topics are available.")
    st.caption(
        "The 0.45 similarity and 0.05 top-two-gap rules are UI interpretation "
        "aids; they are not calibrated probabilities."
    )
    return match


def render_bertopic_assignment(result: dict[str, Any]) -> None:
    render_topic_match_summary(result)

    with st.expander(
        f'Historical dataset statistics for Topic {result["topic_id"]} '
        "— not predictions",
        expanded=False,
    ):
        st.caption(
            "These counts describe documents associated with this topic in "
            "the saved dataset. They do not predict whether the current text "
            "is Stressed or Not Stressed."
        )
        size_columns = st.columns(2)
        size_columns[0].metric(
            "Topic-forming training documents",
            result["training_documents"],
        )
        size_columns[1].metric(
            "Documents assigned across all splits",
            result["assigned_documents"],
        )

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Historical stress labels within this topic**")
            stress_rows = [
                {
                    "Label": label,
                    "Documents": values["documents"],
                    "Share within topic": values["percentage"],
                }
                for label, values in result["stress_distribution"].items()
            ]
            st.dataframe(
                stress_rows,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Share within topic": st.column_config.ProgressColumn(
                        "Share within topic",
                        min_value=0.0,
                        max_value=1.0,
                        format="percent",
                    ),
                },
            )
        with right:
            st.markdown("**Historical source communities**")
            st.dataframe(
                result["top_subreddits"],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Share within topic": st.column_config.ProgressColumn(
                        "Share within topic",
                        min_value=0.0,
                        max_value=1.0,
                        format="percent",
                    ),
                },
            )


def render_bertopic_panel(text: str) -> None:
    with st.expander("Stress BERTopic — semantic theme testing", expanded=False):
        st.caption(
            "Output: one primary semantic topic candidate plus four ranked "
            "alternatives. Weak or closely tied results are marked unclear. "
            "This is separate from Stress/Not-Stressed prediction."
        )
        test_tab, overview_tab = st.tabs(["Test a text", "Model overview"])

        with test_tab:
            artifacts_ready = all((
                (BERTOPIC_MODEL_DIR / "config.json").exists(),
                (
                    BERTOPIC_MODEL_DIR
                    / "topic_embeddings.safetensors"
                ).exists(),
                (BERTOPIC_REPORT_DIR / "topic_info.csv").exists(),
            ))
            if not artifacts_ready:
                st.error(
                    "Stress BERTopic artifacts are incomplete. Expected model "
                    f"files in `{BERTOPIC_MODEL_DIR}` and reports in "
                    f"`{BERTOPIC_REPORT_DIR}`."
                )
            run_topic = st.button(
                "Analyze Stress semantic theme",
                key="run_stress_bertopic",
                type="primary",
                use_container_width=True,
                disabled=not artifacts_ready,
            )
            if run_topic:
                if not text.strip():
                    st.warning("Enter text before running BERTopic.")
                else:
                    try:
                        with st.spinner(
                                "Encoding text and comparing learned topics..."):
                            result = predict_stress_topic(text)
                        st.session_state["last_bertopic_result"] = result
                        st.session_state["last_bertopic_text"] = text
                    except ModuleNotFoundError as exc:
                        st.error(
                            f"Missing BERTopic frontend dependency: {exc.name}. "
                            "Install the project's requirements.")
                    except Exception as exc:
                        st.error(
                            "BERTopic inference failed: "
                            f"{type(exc).__name__}: {exc}")

            previous_result = st.session_state.get("last_bertopic_result")
            previous_text = st.session_state.get("last_bertopic_text")
            if previous_result and previous_text == text:
                render_bertopic_assignment(previous_result)
            elif previous_result and previous_text != text:
                st.info(
                    "The text changed. Run BERTopic again to refresh the topic.")

        with overview_tab:
            try:
                catalog = load_stress_topic_catalog()
                metrics = catalog["metrics"]
                metric_columns = st.columns(4)
                metric_columns[0].metric(
                    "Learned topics",
                    metrics["topic_count_excluding_outliers"],
                )
                metric_columns[1].metric(
                    "Training documents", metrics["documents"])
                metric_columns[2].metric(
                    "Training outliers", f'{metrics["outlier_rate"]:.1%}')
                metric_columns[3].metric(
                    "Topic diversity",
                    f'{metrics["topic_diversity_top_10"]:.3f}',
                )
                st.markdown("**Largest learned topics**")
                st.dataframe(
                    largest_topics(catalog),
                    use_container_width=True,
                    hide_index=True,
                )
                plot_columns = st.columns(2)
                topic_sizes = BERTOPIC_REPORT_DIR / "plots/topic_sizes.png"
                stress_plot = (
                    BERTOPIC_REPORT_DIR
                    / "plots/topic_stress_distribution.png"
                )
                if topic_sizes.exists():
                    plot_columns[0].image(
                        str(topic_sizes),
                        caption="Largest training topics",
                        use_container_width=True,
                    )
                if stress_plot.exists():
                    plot_columns[1].image(
                        str(stress_plot),
                        caption="Stress-label distribution by topic",
                        use_container_width=True,
                    )
            except Exception as exc:
                st.error(
                    "Could not load the BERTopic overview: "
                    f"{type(exc).__name__}: {exc}")


def render_cbt_bertopic_assignment(result: dict[str, Any]) -> None:
    match = render_topic_match_summary(
        result,
        weak_heading="No reliable CBT topic found",
    )
    if match["is_weak"]:
        st.info(
            "Historical topic statistics are hidden because this text was "
            "not reliably assigned to a CBT semantic topic."
        )
        return

    with st.expander(
        f'Historical dataset statistics for Topic {result["topic_id"]} '
        "— not predictions",
        expanded=False,
    ):
        st.caption(
            "These counts describe CBT dataset documents associated with this "
            "topic. They do not predict Distortion/No Distortion or a "
            "distortion type for the current text."
        )
        size_columns = st.columns(2)
        size_columns[0].metric(
            "Topic-forming training documents",
            result["training_documents"],
        )
        size_columns[1].metric(
            "Documents assigned across all splits",
            result["assigned_documents"],
        )

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Historical distortion status within this topic**")
            status_rows = [
                {
                    "Status": label,
                    "Documents": values["documents"],
                    "Share within topic": values["percentage"],
                }
                for label, values in result[
                    "distortion_status_distribution"
                ].items()
            ]
            st.dataframe(
                status_rows,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Share within topic": st.column_config.ProgressColumn(
                        "Share within topic",
                        min_value=0.0,
                        max_value=1.0,
                        format="percent",
                    ),
                },
            )
        with right:
            st.markdown("**Historical distortion labels within this topic**")
            st.dataframe(
                result["distortion_type_distribution"][:8],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Share within topic": st.column_config.ProgressColumn(
                        "Share within topic",
                        min_value=0.0,
                        max_value=1.0,
                        format="percent",
                    ),
                },
            )


def render_cbt_bertopic_panel(text: str) -> None:
    with st.expander("CBT BERTopic — semantic theme testing", expanded=False):
        st.caption(
            "Topics were learned from short annotated distortion excerpts. "
            "The output is one primary semantic topic plus ranked alternatives. "
            "Weak results show no reliable CBT topic instead of an assignment. "
            "This is separate from Distortion/No Distortion prediction."
        )
        test_tab, overview_tab = st.tabs(["Test a text", "Model overview"])

        with test_tab:
            artifacts_ready = all((
                (CBT_BERTOPIC_MODEL_DIR / "config.json").exists(),
                (
                    CBT_BERTOPIC_MODEL_DIR
                    / "topic_embeddings.safetensors"
                ).exists(),
                (CBT_BERTOPIC_REPORT_DIR / "topic_info.csv").exists(),
            ))
            if not artifacts_ready:
                st.error(
                    "CBT BERTopic artifacts are incomplete. Expected model "
                    f"files in `{CBT_BERTOPIC_MODEL_DIR}` and reports in "
                    f"`{CBT_BERTOPIC_REPORT_DIR}`."
                )
            run_topic = st.button(
                "Analyze CBT semantic theme",
                key="run_cbt_bertopic",
                type="primary",
                use_container_width=True,
                disabled=not artifacts_ready,
            )
            if run_topic:
                if not text.strip():
                    st.warning("Enter text before running CBT BERTopic.")
                else:
                    try:
                        with st.spinner(
                                "Encoding text and comparing learned CBT topics..."):
                            result = predict_cbt_topic(text)
                        st.session_state["last_cbt_bertopic_result"] = result
                        st.session_state["last_cbt_bertopic_text"] = text
                    except ModuleNotFoundError as exc:
                        st.error(
                            f"Missing BERTopic frontend dependency: {exc.name}. "
                            "Install the project's requirements."
                        )
                    except Exception as exc:
                        st.error(
                            "CBT BERTopic inference failed: "
                            f"{type(exc).__name__}: {exc}"
                        )

            previous_result = st.session_state.get(
                "last_cbt_bertopic_result")
            previous_text = st.session_state.get("last_cbt_bertopic_text")
            if previous_result and previous_text == text:
                render_cbt_bertopic_assignment(previous_result)
            elif previous_result and previous_text != text:
                st.info(
                    "The CBT text changed. Run BERTopic again to refresh "
                    "the topic."
                )

        with overview_tab:
            try:
                catalog = load_cbt_topic_catalog_cached()
                metrics = catalog["metrics"]
                metric_columns = st.columns(4)
                metric_columns[0].metric(
                    "Learned topics",
                    metrics["topic_count_excluding_outliers"],
                )
                metric_columns[1].metric(
                    "Training documents", metrics["documents"])
                metric_columns[2].metric(
                    "Training outliers", f'{metrics["outlier_rate"]:.1%}')
                metric_columns[3].metric(
                    "Topic diversity",
                    f'{metrics["topic_diversity_top_10"]:.3f}',
                )
                st.markdown("**Largest learned topics**")
                st.dataframe(
                    largest_topics(catalog),
                    use_container_width=True,
                    hide_index=True,
                )
                plot_columns = st.columns(3)
                plots = (
                    (
                        "topic_sizes.png",
                        "Largest CBT training topics",
                    ),
                    (
                        "topic_distortion_status_distribution.png",
                        "Distortion status by topic",
                    ),
                    (
                        "topic_distortion_type_distribution.png",
                        "Distortion labels by topic",
                    ),
                )
                for column, (file_name, caption) in zip(plot_columns, plots):
                    path = CBT_BERTOPIC_REPORT_DIR / "plots" / file_name
                    if path.exists():
                        column.image(
                            str(path),
                            caption=caption,
                            use_container_width=True,
                        )
            except Exception as exc:
                st.error(
                    "Could not load the CBT BERTopic overview: "
                    f"{type(exc).__name__}: {exc}"
                )


def render_xai_panel(text: str) -> None:
    with st.expander("XAI explanations", expanded=False):
        st.caption("Generate local explanations for one transformer model.")
        col_a, col_b, col_c = st.columns([1.4, 1, 1])
        with col_a:
            model_name = st.selectbox(
                "XAI model",
                list(TRANSFORMER_MODELS.keys()),
                index=2,
                help="DeBERTa-v3 is the backend model used for app predictions.",
            )
        with col_b:
            target_label = st.selectbox(
                "Class to explain",
                ["Predicted class", "Not Stressed", "Stressed"],
                index=0,
            )
        with col_c:
            methods = st.multiselect(
                "Methods",
                [
                    "Integrated Gradients",
                    "SHAP",
                    "LIME",
                    "Combined: SHAP + LIME + Integrated Gradients",
                    "Counterfactual explanations",
                ],
                default=["Integrated Gradients"],
            )

        run_xai = st.button("Generate XAI", use_container_width=True)
        if not run_xai:
            return

        if not methods:
            st.warning("Select at least one XAI method.")
            return

        release_cached_gpu_resources()
        model = None
        try:
            (
                StressIG,
                StressSHAP,
                StressLIME,
                StressCombinedXAI,
                StressCounterfactual,
            ) = load_xai_classes()
            tokenizer, model, _, max_len = load_transformer_model(model_name)
            device = preferred_torch_device()
            model.to(device).eval()
            prepared_text = preprocess_for_transformers(text)
            labels = {str(key): value for key, value in LABELS.items()}
            XAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            stem = f"{model_name.lower().replace('-', '_')}_{timestamp}"
            target_class = None
            if target_label == "Not Stressed":
                target_class = 0
            elif target_label == "Stressed":
                target_class = 1

            if "Integrated Gradients" in methods:
                with st.spinner("Generating Integrated Gradients..."):
                    ig = StressIG(
                        model,
                        tokenizer,
                        labels,
                        device=str(device),
                        max_length=max_len,
                    )
                    ig_result = ig.explain(prepared_text, target_class=target_class)
                    ig_path = XAI_OUTPUT_DIR / f"{stem}_ig.png"
                    ig.save_plot(ig_result, ig_path)
                st.write(
                    f"Integrated Gradients: {ig_result['predicted_label']} "
                    f"({ig_result['confidence']:.1%})"
                )
                render_rationales(ig_result, "Integrated Gradients")
                st.image(str(ig_path), use_container_width=True)

            if "SHAP" in methods:
                with st.spinner("Generating SHAP explanation..."):
                    shap_xai = StressSHAP(
                        model,
                        tokenizer,
                        labels,
                        device=str(device),
                        max_length=max_len,
                    )
                    shap_result = shap_xai.explain(
                        prepared_text,
                        target_class=target_class,
                    )
                    shap_png = XAI_OUTPUT_DIR / f"{stem}_shap.png"
                    shap_html = XAI_OUTPUT_DIR / f"{stem}_shap.html"
                    shap_xai.save_bar_plot(shap_result, shap_png)
                    shap_xai.save_text_plot(shap_result, shap_html)
                st.write(
                    f"SHAP: predicted {shap_result['predicted_label']} "
                    f"({shap_result['confidence']:.1%}); explaining "
                    f"{shap_result['explained_label']} "
                    f"({shap_result['explained_probability']:.1%})"
                )
                render_rationales(shap_result, "SHAP")
                st.image(str(shap_png), use_container_width=True)
                st.markdown(f"SHAP HTML saved to `{shap_html}`")

            if "LIME" in methods:
                st.caption(
                    "LIME masks words and fits a sparse local surrogate around "
                    "this text. Positive weights support the explained class."
                )
                with st.spinner("Generating LIME explanation..."):
                    lime_xai = StressLIME(
                        model,
                        tokenizer,
                        labels,
                        device=str(device),
                        max_length=max_len,
                    )
                    lime_result = lime_xai.explain(
                        prepared_text,
                        target_class=target_class,
                    )
                    lime_png = XAI_OUTPUT_DIR / f"{stem}_lime.png"
                    lime_html = XAI_OUTPUT_DIR / f"{stem}_lime.html"
                    lime_xai.save_plot(lime_result, lime_png)
                    lime_xai.save_html(lime_result, lime_html)
                st.write(
                    f"LIME: predicted {lime_result['predicted_label']} "
                    f"({lime_result['confidence']:.1%}); explaining "
                    f"{lime_result['explained_label']} "
                    f"({lime_result['explained_probability']:.1%})"
                )
                st.caption(
                    f"Local surrogate fidelity (R²): {lime_result['local_r2']:.3f} "
                    f"using {lime_result['num_samples']:,} perturbations."
                )
                render_rationales(
                    lime_result,
                    f"LIME — {lime_result['explained_label']}",
                )
                st.image(str(lime_png), use_container_width=True)
                st.markdown(f"LIME HTML saved to `{lime_html}`")

            combined_method = "Combined: SHAP + LIME + Integrated Gradients"
            if combined_method in methods:
                st.caption(
                    "This ensemble normalizes and aligns word evidence from SHAP, "
                    "LIME, and Integrated Gradients, then averages their signed "
                    "contributions. It can take several minutes because all three "
                    "methods are computed."
                )
                with st.spinner("Generating the combined XAI explanation..."):
                    combined_xai = StressCombinedXAI(
                        model,
                        tokenizer,
                        labels,
                        device=str(device),
                        max_length=max_len,
                    )
                    combined_result = combined_xai.explain(
                        prepared_text,
                        target_class=target_class,
                    )
                    combined_path = XAI_OUTPUT_DIR / f"{stem}_combined_xai.png"
                    combined_xai.save_plot(combined_result, combined_path)
                st.write(
                    f"Combined XAI: predicted {combined_result['predicted_label']} "
                    f"({combined_result['confidence']:.1%}); explaining "
                    f"{combined_result['explained_label']} "
                    f"({combined_result['explained_probability']:.1%})"
                )
                lime_component = combined_result["lime_result"]
                st.caption(
                    f"LIME local R²: {lime_component['local_r2']:.3f} · "
                    "All contributions are normalized per method before averaging."
                )
                st.success(
                    "SHAP, LIME, and Integrated Gradients all contributed to "
                    "the consensus."
                )

                st.markdown("**Combined word evidence**")
                st.dataframe(
                    [
                        {
                            "Word": row["word"],
                            "LIME": round(float(row["lime"]), 4),
                            "SHAP": round(float(row["shap"]), 4),
                            "Integrated Gradients": round(
                                float(row["integrated_gradients"]), 4
                            ),
                            "Combined": round(float(row["consensus"]), 4),
                            "Agreement": f"{row['agreement']:.0%}",
                        }
                        for row in combined_result["ranked_features"][:15]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                render_rationales(
                    combined_result,
                    "Combined SHAP + LIME + Integrated Gradients",
                )
                st.image(str(combined_path), use_container_width=True)

            if "Counterfactual explanations" in methods:
                st.caption(
                    "Counterfactual search targets the class opposite to the "
                    "model's original prediction."
                )
                with st.spinner("Searching for minimal counterfactual edits..."):
                    counterfactual_xai = StressCounterfactual(
                        model,
                        tokenizer,
                        labels,
                        device=str(device),
                        max_length=max_len,
                    )
                    counterfactual_result = counterfactual_xai.explain(prepared_text)
                best = counterfactual_result.get("best_counterfactual")
                if best and counterfactual_result["counterfactual_found"]:
                    st.success(
                        "Counterfactual found: "
                        f"{counterfactual_result['predicted_label']} → "
                        f"{best['predicted_label']} in {best['num_edits']} edit(s)"
                    )
                elif best:
                    st.warning(
                        "No class-flipping counterfactual was found within the "
                        "edit limit. Showing the closest candidate."
                    )
                else:
                    st.warning("No valid counterfactual candidates were generated.")

                if best:
                    st.markdown("**Counterfactual text**")
                    st.code(best["text"], language=None)
                    st.dataframe(
                        [
                            {
                                "Operation": edit["operation"],
                                "Original": edit["original"],
                                "Replacement": edit["replacement"] or "∅",
                            }
                            for edit in best["edits"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        f"Target probability: {best['target_probability']:.1%} · "
                        f"Text similarity: {best['similarity']:.1%}"
                    )
                st.info(counterfactual_result["disclaimer"])

        except ModuleNotFoundError as exc:
            st.error(
                f"Missing XAI dependency: {exc.name}. "
                "Install the project's requirements."
            )
        except Exception as exc:
            st.error(f"XAI generation failed: {type(exc).__name__}: {exc}")
        finally:
            offload_model_from_gpu(model)


def render_cbt_xai_panel(text: str) -> None:
    with st.expander("CBT XAI explanations", expanded=False):
        st.caption(
            "Explain the binary Distortion/No Distortion decision from one "
            "trained CBT transformer."
        )
        config = load_cbt_config()
        col_a, col_b, col_c = st.columns([1.4, 1, 1])
        with col_a:
            model_name = st.selectbox(
                "CBT XAI model",
                list(config["models"]),
                index=2,
                key="cbt_xai_model",
                help="Each explanation uses that model's validated threshold.",
            )
        with col_b:
            target_label = st.selectbox(
                "CBT class to explain",
                ["Predicted class", "No Distortion", "Distortion"],
                index=0,
                key="cbt_xai_target",
            )
        with col_c:
            methods = st.multiselect(
                "CBT XAI methods",
                [
                    "Integrated Gradients",
                    "SHAP",
                    "LIME",
                    "Combined: SHAP + LIME + Integrated Gradients",
                    "Counterfactual explanations",
                ],
                default=["Integrated Gradients"],
                key="cbt_xai_methods",
            )

        run_xai = st.button(
            "Generate CBT XAI",
            key="run_cbt_xai",
            use_container_width=True,
        )
        if not run_xai:
            return
        if not methods:
            st.warning("Select at least one CBT XAI method.")
            return
        if not text.strip():
            st.warning("Enter text before generating CBT XAI.")
            return

        release_cached_gpu_resources()
        resources = None
        try:
            (
                CBTIG,
                CBTSHAP,
                CBTLIME,
                CBTCombinedXAI,
                CBTCounterfactual,
            ) = load_cbt_xai_classes()
            resources = load_cbt_transformer_resources(model_name)
            device = preferred_torch_device()
            resources.model.to(device).eval()
            prepared_text = preprocess_for_cbt(text, model_name)
            CBT_XAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            stem = f"{model_name.lower().replace('-', '_')}_{timestamp}"
            target_class = None
            if target_label == "No Distortion":
                target_class = 0
            elif target_label == "Distortion":
                target_class = 1
            common = {
                "label_names": resources.labels,
                "device": str(device),
                "max_length": resources.max_length,
                "decision_threshold": resources.decision_threshold,
            }

            if "Integrated Gradients" in methods:
                with st.spinner("Generating CBT Integrated Gradients..."):
                    ig = CBTIG(
                        resources.model,
                        resources.tokenizer,
                        **common,
                    )
                    ig_result = ig.explain(
                        prepared_text,
                        target_class=target_class,
                    )
                    ig_path = CBT_XAI_OUTPUT_DIR / f"{stem}_ig.png"
                    ig.save_plot(ig_result, ig_path)
                st.write(
                    f"Integrated Gradients: predicted "
                    f"{ig_result['predicted_label']} "
                    f"({ig_result['confidence']:.1%}); explaining "
                    f"{ig_result['explained_label']}"
                )
                render_rationales(ig_result, "CBT Integrated Gradients")
                st.image(str(ig_path), use_container_width=True)

            if "SHAP" in methods:
                with st.spinner("Generating CBT SHAP explanation..."):
                    shap_xai = CBTSHAP(
                        resources.model,
                        resources.tokenizer,
                        **common,
                    )
                    shap_result = shap_xai.explain(
                        prepared_text,
                        target_class=target_class,
                    )
                    shap_png = CBT_XAI_OUTPUT_DIR / f"{stem}_shap.png"
                    shap_html = CBT_XAI_OUTPUT_DIR / f"{stem}_shap.html"
                    shap_xai.save_bar_plot(shap_result, shap_png)
                    shap_xai.save_text_plot(shap_result, shap_html)
                st.write(
                    f"SHAP: predicted {shap_result['predicted_label']} "
                    f"({shap_result['confidence']:.1%}); explaining "
                    f"{shap_result['explained_label']} "
                    f"({shap_result['explained_probability']:.1%})"
                )
                render_rationales(shap_result, "CBT SHAP")
                st.image(str(shap_png), use_container_width=True)
                st.markdown(f"SHAP HTML saved to `{shap_html}`")

            if "LIME" in methods:
                st.caption(
                    "LIME masks words and fits a sparse local surrogate around "
                    "the CBT model's current decision."
                )
                with st.spinner("Generating CBT LIME explanation..."):
                    lime_xai = CBTLIME(
                        resources.model,
                        resources.tokenizer,
                        **common,
                    )
                    lime_result = lime_xai.explain(
                        prepared_text,
                        target_class=target_class,
                    )
                    lime_png = CBT_XAI_OUTPUT_DIR / f"{stem}_lime.png"
                    lime_html = CBT_XAI_OUTPUT_DIR / f"{stem}_lime.html"
                    lime_xai.save_plot(lime_result, lime_png)
                    lime_xai.save_html(lime_result, lime_html)
                st.write(
                    f"LIME: predicted {lime_result['predicted_label']} "
                    f"({lime_result['confidence']:.1%}); explaining "
                    f"{lime_result['explained_label']} "
                    f"({lime_result['explained_probability']:.1%})"
                )
                st.caption(
                    f"Local surrogate fidelity (R²): "
                    f"{lime_result['local_r2']:.3f} using "
                    f"{lime_result['num_samples']:,} perturbations."
                )
                render_rationales(lime_result, "CBT LIME")
                st.image(str(lime_png), use_container_width=True)
                st.markdown(f"LIME HTML saved to `{lime_html}`")

            combined_method = "Combined: SHAP + LIME + Integrated Gradients"
            if combined_method in methods:
                st.caption(
                    "The consensus aligns and normalizes SHAP, LIME, and "
                    "Integrated Gradients word evidence before averaging it."
                )
                with st.spinner("Generating combined CBT XAI..."):
                    combined_xai = CBTCombinedXAI(
                        resources.model,
                        resources.tokenizer,
                        **common,
                    )
                    combined_result = combined_xai.explain(
                        prepared_text,
                        target_class=target_class,
                    )
                    combined_path = (
                        CBT_XAI_OUTPUT_DIR / f"{stem}_combined_xai.png"
                    )
                    combined_xai.save_plot(combined_result, combined_path)
                st.write(
                    f"Combined XAI: predicted "
                    f"{combined_result['predicted_label']} "
                    f"({combined_result['confidence']:.1%}); explaining "
                    f"{combined_result['explained_label']} "
                    f"({combined_result['explained_probability']:.1%})"
                )
                st.markdown("**Combined CBT word evidence**")
                st.dataframe(
                    [
                        {
                            "Word": row["word"],
                            "LIME": round(float(row["lime"]), 4),
                            "SHAP": round(float(row["shap"]), 4),
                            "Integrated Gradients": round(
                                float(row["integrated_gradients"]), 4
                            ),
                            "Combined": round(float(row["consensus"]), 4),
                            "Agreement": f"{row['agreement']:.0%}",
                        }
                        for row in combined_result["ranked_features"][:15]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                render_rationales(
                    combined_result,
                    "Combined CBT SHAP + LIME + Integrated Gradients",
                )
                st.image(str(combined_path), use_container_width=True)

            if "Counterfactual explanations" in methods:
                with st.spinner("Searching for CBT counterfactual edits..."):
                    counterfactual_xai = CBTCounterfactual(
                        resources.model,
                        resources.tokenizer,
                        **common,
                    )
                    counterfactual_result = counterfactual_xai.explain(
                        prepared_text)
                best = counterfactual_result.get("best_counterfactual")
                if best and counterfactual_result["counterfactual_found"]:
                    st.success(
                        "Counterfactual found: "
                        f"{counterfactual_result['predicted_label']} → "
                        f"{best['predicted_label']} in "
                        f"{best['num_edits']} edit(s)"
                    )
                elif best:
                    st.warning(
                        "No class-flipping counterfactual was found within "
                        "the edit limit. Showing the closest candidate."
                    )
                else:
                    st.warning("No valid counterfactual candidates were generated.")
                if best:
                    st.markdown("**Counterfactual text**")
                    st.code(best["text"], language=None)
                    st.dataframe(
                        [
                            {
                                "Operation": edit["operation"],
                                "Original": edit["original"],
                                "Replacement": edit["replacement"] or "∅",
                            }
                            for edit in best["edits"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        f"Target probability: "
                        f"{best['target_probability']:.1%} · "
                        f"Text similarity: {best['similarity']:.1%}"
                    )
                st.info(counterfactual_result["disclaimer"])

        except ModuleNotFoundError as exc:
            st.error(
                f"Missing CBT XAI dependency: {exc.name}. "
                "Install the project's requirements."
            )
        except Exception as exc:
            st.error(
                f"CBT XAI generation failed: {type(exc).__name__}: {exc}"
            )
        finally:
            offload_model_from_gpu(
                resources.model if resources is not None else None)


def render_rationales(result: dict, method: str) -> None:
    result = refresh_rationales(result)
    rationales = result.get("rationales", [])
    st.markdown(f"**{method} rationale spans**")
    if not rationales:
        st.caption("No rationale span passed the attribution threshold.")
        return

    rows = [
        {
            "Span": rationale["text"],
            "Score": round(float(rationale["score"]), 4),
            "Token Start": rationale["token_start"],
            "Token End": rationale["token_end"],
        }
        for rationale in rationales
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown(
        f"""
        <div style="border:1px solid #293241;border-radius:7px;padding:0.8rem;
                    background:#0d1119;line-height:1.8;">
            {result.get("highlighted_text", html.escape(result.get("text", "")))}
        </div>
        """,
        unsafe_allow_html=True,
    )


def refresh_rationales(result: dict) -> dict:
    if str(XAI_DIR) not in sys.path:
        sys.path.insert(0, str(XAI_DIR))
    rationale_module = importlib.import_module("rationale_extractor")
    importlib.reload(rationale_module)

    tokens = result.get("tokens", [])
    scores = result.get("attributions", [])
    text_value = result.get("text", "")
    if not tokens or not scores:
        return result

    refreshed = dict(result)
    rationales = rationale_module.extract_rationale_spans(tokens, scores)
    refreshed["rationales"] = rationales
    refreshed["highlighted_text"] = rationale_module.highlight_rationales(
        text_value,
        rationales,
    )
    return refreshed


def render_stress_tab() -> None:
    st.subheader("Stress Header")
    st.markdown(
        '<div class="app-kicker">Run one text sample through three transformer models and two TF-IDF baselines, then compare agreement and confidence.</div>',
        unsafe_allow_html=True,
    )

    sample_text = (
        "I have been feeling overwhelmed with everything lately. "
        "I cannot sleep properly and I keep worrying that I will fail."
    )
    with st.container(border=True):
        left, right = st.columns([3, 1], gap="large")
        with left:
            text = st.text_area(
                "User input",
                value=sample_text,
                height=150,
                placeholder="Type or paste a Reddit-style post here...",
            )
        with right:
            expected = st.selectbox(
                "Actual label",
                ["Unknown", "Not Stressed", "Stressed"],
                index=0,
                help="Optional label for manually checking model agreement.",
            )
            st.write("")
            run_button = st.button("Run all models", type="primary", use_container_width=True)

    render_bertopic_panel(text)

    if run_button:
        if not text.strip():
            st.warning("Please enter text before running the models.")
            return

        with st.spinner("Running all models..."):
            results = run_all_models(text)
        st.session_state["last_results"] = results
        st.session_state["last_text"] = text
        st.session_state["last_expected"] = expected
    elif "last_results" in st.session_state:
        results = st.session_state["last_results"]
        text = st.session_state.get("last_text", text)
        expected = st.session_state.get("last_expected", expected)
    else:
        st.markdown('<div class="section-label">Ready</div>', unsafe_allow_html=True)
        st.info("Enter text and run all models to compare predictions.")
        render_artifacts()
        return

    summary, vote_text = summarize_results(results)
    stats = comparison_stats(results)
    best = stats["best"]

    st.markdown('<div class="section-label">Comparison Summary</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Majority", stats["majority"])
    metric_cols[1].metric("Completed", f"{stats['completed']} / {len(results)}")
    metric_cols[2].metric("Best model", best.model if best else "-")
    metric_cols[3].metric("Best confidence", percent(best.confidence if best else None))

    st.write(summary)
    st.write(vote_text)

    st.markdown('<div class="section-label">Model Cards</div>', unsafe_allow_html=True)
    render_result_cards(results)

    rows = result_rows(results)
    if expected != "Unknown":
        for row in rows:
            row["Matches Actual"] = row["Prediction"] == expected

    st.markdown('<div class="section-label">Detailed Outputs</div>', unsafe_allow_html=True)
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Stress Probability": st.column_config.ProgressColumn(
                "Stress Probability",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
        },
    )

    render_xai_panel(text)

    failed = [result for result in results if result.status != "OK"]
    if failed:
        with st.expander("Model loading or prediction issues", expanded=True):
            for result in failed:
                st.error(f"{result.model}: {result.status}")

    render_artifacts()


def render_cbt_tab() -> None:
    st.subheader("CBT Cognitive Distortion Header")
    st.markdown(
        '<div class="app-kicker">Compare three CBT transformer models and '
        'two TF-IDF baselines for Distortion vs No Distortion, inspect '
        'semantic topics, and generate local explanations.</div>',
        unsafe_allow_html=True,
    )

    sample_text = (
        "If I make one mistake, everyone will think I am a complete failure. "
        "Nothing I do is ever good enough."
    )
    with st.container(border=True):
        left, right = st.columns([3, 1], gap="large")
        with left:
            text = st.text_area(
                "CBT user input",
                value=sample_text,
                height=150,
                placeholder="Type or paste a thought or patient-style question...",
                key="cbt_user_input",
            )
        with right:
            expected = st.selectbox(
                "Actual CBT label",
                ["Unknown", "No Distortion", "Distortion"],
                index=0,
                key="cbt_expected_label",
                help="Optional label for manually checking model agreement.",
            )
            st.write("")
            run_button = st.button(
                "Run CBT models",
                key="run_cbt_models",
                type="primary",
                use_container_width=True,
            )

    render_cbt_bertopic_panel(text)

    if run_button:
        if not text.strip():
            st.warning("Please enter text before running the CBT models.")
            return
        with st.spinner("Running all CBT models..."):
            results = run_cbt_models(text)
        st.session_state["last_cbt_results"] = results
        st.session_state["last_cbt_text"] = text
        st.session_state["last_cbt_expected"] = expected
    elif "last_cbt_results" in st.session_state:
        results = st.session_state["last_cbt_results"]
        text = st.session_state.get("last_cbt_text", text)
        expected = st.session_state.get("last_cbt_expected", expected)
    else:
        st.markdown(
            '<div class="section-label">Ready</div>',
            unsafe_allow_html=True,
        )
        st.info(
            "Enter text and run the CBT models, or test CBT BERTopic "
            "independently above."
        )
        render_cbt_artifacts()
        return

    stats = cbt_comparison_stats(results)
    best = stats["best"]
    st.markdown(
        '<div class="section-label">CBT Comparison Summary</div>',
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Majority", stats["majority"])
    metric_cols[1].metric(
        "Completed", f"{stats['completed']} / {len(results)}")
    metric_cols[2].metric("Best model", best.model if best else "-")
    metric_cols[3].metric(
        "Best selected-class probability",
        percent(best.confidence if best else None),
    )
    if best:
        st.write(
            f"Highest selected-class probability: {best.model} "
            f"({best.prediction}, {best.confidence:.3f})."
        )
    st.write(
        f"Model vote: {stats['distortion_votes']} distortion, "
        f"{stats['no_distortion_votes']} no distortion. "
        f"Majority: {stats['majority']}."
    )

    st.markdown(
        '<div class="section-label">CBT Model Cards</div>',
        unsafe_allow_html=True,
    )
    render_cbt_result_cards(results)

    rows = cbt_result_rows(results)
    if expected != "Unknown":
        for row in rows:
            row["Matches Actual"] = row["Prediction"] == expected
    st.markdown(
        '<div class="section-label">Detailed CBT Outputs</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Distortion Probability": st.column_config.ProgressColumn(
                "Distortion Probability",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
            "Decision Threshold": st.column_config.NumberColumn(
                "Decision Threshold", format="%.2f"),
            "Confidence": st.column_config.ProgressColumn(
                "Selected-class probability",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
        },
    )

    render_cbt_xai_panel(text)

    failed = [result for result in results if result.status != "OK"]
    if failed:
        with st.expander("CBT model loading or prediction issues", expanded=True):
            for result in failed:
                st.error(f"{result.model}: {result.status}")
    render_cbt_artifacts()


def render_app() -> None:
    st.set_page_config(
        page_title="C3 Text Stressor and Cognitive Distortion",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_styles()
    st.title("C3 Text Stressor and Cognitive Distortion Detection")
    st.markdown(
        '<div class="app-kicker">Use the two research headers independently: '
        'Stress detection and CBT cognitive-distortion detection.</div>',
        unsafe_allow_html=True,
    )
    stress_tab, cbt_tab = st.tabs(["Stress Header", "CBT Header"])
    with stress_tab:
        render_stress_tab()
    with cbt_tab:
        render_cbt_tab()


if __name__ == "__main__":
    render_app()
