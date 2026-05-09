from __future__ import annotations

import html
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "c3_text_stressor_distortion" / "Stress header"
ENV_PATH = PROJECT_ROOT / ".env"

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
        "checkpoint": "BERT_best.pt",
        "max_len": 192,
    },
    "MentalBERT": {
        "hf_id": "mental/mental-bert-base-uncased",
        "checkpoint": "MentalBERT_best.pt",
        "max_len": 192,
    },
    "DeBERTa-v3": {
        "hf_id": "microsoft/deberta-v3-base",
        "checkpoint": "DeBERTa-v3_best.pt",
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
    checkpoint_path = MODEL_DIR / spec["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auth_kwargs = hf_auth_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"], **auth_kwargs)
    model = DualHeadStressModel(spec["hf_id"], auth_kwargs).to(device)

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


def predict_transformer(model_name: str, text: str) -> PredictionResult:
    try:
        import torch

        tokenizer, model, device, max_len = load_transformer_model(model_name)
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
    results = []
    for model_name in TRANSFORMER_MODELS:
        results.append(predict_transformer(model_name, text))
    for model_name, file_name in BASELINE_MODELS.items():
        results.append(predict_baseline(model_name, file_name, text))
    return results


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
        st.dataframe(artifact_rows, use_container_width=True, hide_index=True)


def render_app() -> None:
    st.set_page_config(
        page_title="C3 Text Stressor Distortion",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_styles()

    st.title("C3 Text Stressor and Stress Detection")
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

    if not run_button:
        st.markdown('<div class="section-label">Ready</div>', unsafe_allow_html=True)
        st.info("Enter text and run all models to compare predictions.")
        render_artifacts()
        return

    if not text.strip():
        st.warning("Please enter text before running the models.")
        return

    with st.spinner("Running all models..."):
        results = run_all_models(text)

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

    failed = [result for result in results if result.status != "OK"]
    if failed:
        with st.expander("Model loading or prediction issues", expanded=True):
            for result in failed:
                st.error(f"{result.model}: {result.status}")


    render_artifacts()


if __name__ == "__main__":
    render_app()
