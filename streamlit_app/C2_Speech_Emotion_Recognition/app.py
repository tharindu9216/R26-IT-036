"""Streamlit interface for speech emotion and appraisal analysis."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
APPRAISAL_ARTIFACT = (
    PROJECT_ROOT / "model" / "C2_Speech_Emotion_Recognition" / "audeering_pretrained"
    / "artifacts" / "appraisal" / "advanced_appraisal_engine.joblib"
)
AUDEERING_MODEL_DIR = (
    PROJECT_ROOT / "model" / "C2_Speech_Emotion_Recognition" / "audeering_pretrained"
    / "pretrained" / "audeering_emotion_model"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from appraisal_stress.engine import AdvancedAppraisalEngine


st.set_page_config(
    page_title="Voice Appraisal Lab",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
      :root {
        --ink: #172033;
        --muted: #607086;
        --brand: #246bfe;
        --brand-soft: #eaf1ff;
        --surface: rgba(255, 255, 255, 0.78);
        --line: #dce5f1;
      }

      .stApp {
        background:
          radial-gradient(circle at 8% 0%, rgba(36,107,254,.10), transparent 28rem),
          radial-gradient(circle at 95% 10%, rgba(35,191,166,.09), transparent 25rem),
          #f7f9fc;
        color: var(--ink);
      }

      [data-testid="stSidebar"] {
        background: rgba(248, 250, 253, .94);
        border-right: 1px solid var(--line);
      }

      .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
      }

      .hero {
        padding: 1.8rem 2rem;
        border: 1px solid rgba(36,107,254,.16);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(255,255,255,.96), rgba(238,244,255,.88));
        box-shadow: 0 18px 55px rgba(37, 57, 88, .08);
        margin-bottom: 1.3rem;
      }

      .hero-kicker {
        color: var(--brand);
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: .45rem;
      }

      .hero h1 {
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.05;
        letter-spacing: -.045em;
        margin: 0 0 .65rem;
      }

      .hero p {
        color: var(--muted);
        max-width: 760px;
        font-size: 1.02rem;
        line-height: 1.65;
        margin: 0;
      }

      .result-banner {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.15rem 1.3rem;
        background: var(--surface);
        margin: .4rem 0 1rem;
      }

      .result-banner .eyebrow {
        color: var(--muted);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
      }

      .result-banner .value {
        color: var(--ink);
        font-size: 1.55rem;
        font-weight: 760;
        margin-top: .2rem;
      }

      .pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: .26rem .66rem;
        font-size: .78rem;
        font-weight: 700;
        margin-left: .45rem;
      }

      .pill-low { color: #086c55; background: #dff8ef; }
      .pill-moderate { color: #855700; background: #fff0bf; }
      .pill-high { color: #9d2b31; background: #ffe3e5; }
      .pill-uncertain { color: #6941a5; background: #efe5ff; }

      [data-testid="stMetric"] {
        background: rgba(255,255,255,.80);
        border: 1px solid var(--line);
        padding: 1rem 1.05rem;
        border-radius: 16px;
      }

      [data-testid="stMetricLabel"] { color: var(--muted); }
      [data-testid="stMetricValue"] { color: var(--ink); }

      div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,.72);
        border-radius: 16px;
      }

      .method-note {
        color: var(--muted);
        font-size: .88rem;
        line-height: 1.6;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_appraisal_engine():
    if not APPRAISAL_ARTIFACT.is_file():
        raise FileNotFoundError(
            f"Appraisal artifact not found: {APPRAISAL_ARTIFACT}"
        )
    return AdvancedAppraisalEngine.load(APPRAISAL_ARTIFACT)


@st.cache_resource(show_spinner=False)
def load_audeering_predictor():
    if not AUDEERING_MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"AudEERING model not found: {AUDEERING_MODEL_DIR}"
        )

    # Importing this module loads the local Wav2Vec2 model once. Streamlit's
    # resource cache keeps it available for subsequent UI reruns.
    from inference.audeering_inference import predict_emotion_dimensions

    return predict_emotion_dimensions


def as_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [as_builtin(item) for item in value]
    if pd.isna(value):
        return None
    return value


def run_appraisal(arousal, dominance, valence, file_name="manual_input"):
    engine = load_appraisal_engine()
    sample = pd.DataFrame(
        [
            {
                "file_name": file_name,
                "actual_class": "unknown",
                "arousal": float(arousal),
                "dominance": float(dominance),
                "valence": float(valence),
            }
        ]
    )
    row = engine.transform(sample).iloc[0]
    return {key: as_builtin(value) for key, value in row.to_dict().items()}


def analyze_audio(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower() or ".wav"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(uploaded_file.getvalue())
            temporary_path = Path(temporary.name)

        predictor = load_audeering_predictor()
        dimensions = predictor(temporary_path)
        return run_appraisal(
            dimensions["arousal"],
            dimensions["dominance"],
            dimensions["valence"],
            file_name=uploaded_file.name,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def uncertainty_pill():
    return '<span class="pill pill-uncertain">Uncertain</span>'


def render_result(result):
    """
    Render the appraisal result safely.

    Important UI rule:
    If the engine marks the sample as uncertain, "Uncertain" becomes the
    final interpretation and the dominant pattern is shown as tentative.
    """

    ood_flag = bool(result.get("ood_flag", False))
    low_confidence = bool(
        result.get("low_interpretation_confidence", False)
    )

    uncertain = (
        bool(result.get("interpretation_uncertain", False))
        or ood_flag
        or low_confidence
    )

    dominant_state = (
        str(result.get("dominant_appraisal_state", "unknown"))
        .replace("_", " ")
        .title()
    )
    dominant_weight = float(
        result.get("dominant_appraisal_weight", 0.0)
    )

    # ========================================================
    # Final interpretation banner
    # ========================================================

    if uncertain:
        st.warning(
            "⚠️ Final interpretation: UNCERTAIN. "
            "The dominant appraisal pattern below is tentative."
        )

        st.markdown(
            f"""
            <div class="result-banner">
              <div class="eyebrow">Final interpretation</div>
              <div class="value">
                Uncertain
                {uncertainty_pill()}
              </div>

              <div style="margin-top:.8rem; color:var(--muted);">
                <strong>Tentative appraisal pattern:</strong>
                {dominant_state}
                <br>
                <strong>Tentative pattern weight:</strong>
                {dominant_weight:.3f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            f"""
            <div class="result-banner">
              <div class="eyebrow">Dominant appraisal pattern</div>
              <div class="value">
                {dominant_state}
              </div>

              <div style="margin-top:.8rem; color:var(--muted);">
                <strong>Pattern weight:</strong>
                {dominant_weight:.3f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # Arousal / Dominance / Valence
    # ========================================================

    dimension_columns = st.columns(3)

    for column, label, key in zip(
        dimension_columns,
        ("Arousal", "Dominance", "Valence"),
        ("arousal", "dominance", "valence"),
    ):
        column.metric(
            label,
            f"{float(result[key]):.4f}",
        )

    # ========================================================
    # Appraisal interpretation metrics
    # ========================================================

    st.markdown("#### Appraisal interpretation")

    outcome_columns = st.columns(4)

    outcome_columns[0].metric(
        "Dominant pattern weight",
        f"{dominant_weight:.3f}",
    )

    outcome_columns[1].metric(
        "Interpretation confidence",
        f"{float(result['interpretation_confidence']):.3f}",
    )

    outcome_columns[2].metric(
        "Reference similarity",
        f"{float(result['reference_similarity']):.3f}",
    )

    outcome_columns[3].metric(
        "Appraisal ambiguity",
        f"{float(result.get('appraisal_ambiguity', 0.0)):.3f}",
    )

    # ========================================================
    # Appraisal-state strengths
    # ========================================================

    state_labels = {
        "state_threat": "Threat",
        "state_challenge": "Challenge",
        "state_withdrawal_distress": "Withdrawal distress",
        "state_negative_activation": "Negative activation",
        "state_positive_activation": "Positive activation",
        "state_calm_recovery": "Calm / recovery",
    }

    available_states = {
        key: label
        for key, label in state_labels.items()
        if key in result
    }

    state_data = pd.DataFrame(
        {
            "Appraisal state": list(available_states.values()),
            "Strength": [
                float(result[key])
                for key in available_states
            ],
        }
    ).set_index("Appraisal state")

    chart_column, explanation_column = st.columns([1.2, 1])

    with chart_column:
        st.markdown("#### Appraisal-state strengths")

        if not state_data.empty:
            st.bar_chart(
                state_data,
                horizontal=True,
                color="#246bfe",
            )
        else:
            st.info(
                "No appraisal-state strengths were returned."
            )

    # ========================================================
    # Reliability and explanation
    # ========================================================

    with explanation_column:
        st.markdown("#### Reliability and explanation")

        confidence = float(
            result.get(
                "interpretation_confidence",
                0.0,
            )
        )

        ambiguity = float(
            result.get(
                "appraisal_ambiguity",
                0.0,
            )
        )

        reference_similarity = float(
            result.get(
                "reference_similarity",
                0.0,
            )
        )

        engine = load_appraisal_engine()

        confidence_threshold = getattr(
            engine,
            "confidence_threshold",
            None,
        )

        if ood_flag:
            st.error(
                "Final interpretation is uncertain because this sample "
                "was flagged as outside the fitted RAVDESS reference "
                "distribution."
            )

        elif low_confidence:
            st.warning(
                "Final interpretation is uncertain because the appraisal "
                "evidence has low interpretation confidence."
            )

        else:
            st.success(
                "No uncertainty flag was triggered for this sample."
            )

        st.markdown(
            f"""
            **Interpretation confidence:** `{confidence:.3f}`  
            **Appraisal ambiguity:** `{ambiguity:.3f}`  
            **Reference similarity:** `{reference_similarity:.3f}`
            """
        )

        if confidence_threshold is not None:
            st.markdown(
                f"**Reference-derived confidence threshold:** "
                f"`{float(confidence_threshold):.3f}`"
            )

        # Do not convert reference similarity into ad-hoc qualitative
        # categories. The engine's OOD flag is the actual decision rule.
        if not ood_flag:
            st.caption(
                "The sample was not flagged as out-of-distribution. "
                "Reference similarity is shown as a diagnostic value, "
                "not as a probability."
            )

        xai_text = result.get(
            "xai_top_appraisal_contributions",
            "No explanation returned.",
        )

        st.markdown("**Top appraisal contributions**")
        st.code(
            str(xai_text),
            language=None,
        )

    # ========================================================
    # Technical details
    # ========================================================

    with st.expander("Technical details"):

        detail_columns = st.columns(3)

        detail_columns[0].metric(
            "Arousal z-score",
            f"{float(result['arousal_z']):+.3f}",
        )

        detail_columns[1].metric(
            "Dominance z-score",
            f"{float(result['dominance_z']):+.3f}",
        )

        detail_columns[2].metric(
            "Valence z-score",
            f"{float(result['valence_z']):+.3f}",
        )

        technical_details = pd.DataFrame(
            {
                "Field": list(result),
                "Value": [
                    str(result[key])
                    for key in result
                ],
            }
        )

        st.dataframe(
            technical_details,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # Export
    # ========================================================

    export_record = {
        key: as_builtin(value)
        for key, value in result.items()
    }

    # Add explicit presentation-level fields to exported results.
    export_record["interpretation_status"] = (
        "uncertain" if uncertain else "interpretable"
    )
    export_record["dominant_appraisal_pattern"] = dominant_state
    export_record["dominant_appraisal_weight"] = dominant_weight

    download_columns = st.columns(2)

    download_columns[0].download_button(
        "Download JSON",
        data=json.dumps(
            export_record,
            indent=2,
        ),
        file_name="voice_appraisal_result.json",
        mime="application/json",
        use_container_width=True,
    )

    download_columns[1].download_button(
        "Download CSV",
        data=pd.DataFrame(
            [export_record]
        ).to_csv(index=False),
        file_name="voice_appraisal_result.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.info(
        "Research-use notice: appraisal patterns are exploratory, corpus-relative "
        "descriptors, not clinical or psychological assessments. When the "
        "interpretation is uncertain, the displayed dominant pattern is tentative."
    )


with st.sidebar:
    st.markdown("### Voice Appraisal Lab")
    st.caption("Speech emotion dimensions → appraisal interpretation")
    st.divider()

    model_ready = AUDEERING_MODEL_DIR.is_dir()
    artifact_ready = APPRAISAL_ARTIFACT.is_file()
    st.markdown(f"{'✅' if model_ready else '❌'} AudEERING model")
    st.markdown(f"{'✅' if artifact_ready else '❌'} Appraisal artifact")
    st.markdown(f"**Compute:** `{'CUDA' if torch.cuda.is_available() else 'CPU'}`")

    st.divider()
    st.markdown("**Pipeline**")
    st.caption("1. Decode and resample audio to 16 kHz")
    st.caption("2. Predict Arousal, Dominance, Valence")
    st.caption("3. Apply the fitted fuzzy appraisal engine")
    st.caption("4. Report uncertainty and XAI contributions")

    st.divider()
    st.caption("RAVDESS is the appraisal reference corpus; SAVEE is external validation.")


st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Speech emotion research interface</div>
      <h1>Listen beyond the words.</h1>
      <p>
        Upload a voice recording to estimate its arousal, dominance, and
        valence, then inspect an appraisal-theory-guided interpretation with
        uncertainty and explainability signals.
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)


audio_tab, manual_tab, method_tab = st.tabs(
    ["🎙️ Audio analysis", "🎛️ Manual ADV", "📘 Method"]
)


with audio_tab:
    st.markdown("### Analyze a recording")
    st.caption("Use a clear speech-only WAV, FLAC, or OGG recording.")
    uploaded_audio = st.file_uploader(
        "Drop an audio file here",
        type=["wav", "flac", "ogg"],
        accept_multiple_files=False,
        key="audio_upload",
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio.getvalue())

    analyze_clicked = st.button(
        "Analyze voice",
        type="primary",
        disabled=uploaded_audio is None,
        use_container_width=True,
    )

    if analyze_clicked and uploaded_audio is not None:
        try:
            with st.spinner("Loading the model and analyzing the recording…"):
                st.session_state["audio_result"] = analyze_audio(uploaded_audio)
                st.session_state["audio_result_name"] = uploaded_audio.name
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            with st.expander("Error details"):
                st.exception(exc)

    if "audio_result" in st.session_state:
        result_name = st.session_state.get("audio_result_name", "uploaded audio")
        st.caption(f"Latest analysis: `{result_name}`")
        render_result(st.session_state["audio_result"])


with manual_tab:
    st.markdown("### Explore the appraisal engine")
    st.caption(
        "Enter Arousal, Dominance, and Valence directly without running audio inference."
    )

    with st.form("manual_adv_form"):
        slider_columns = st.columns(3)
        manual_arousal = slider_columns[0].slider(
            "Arousal", 0.0, 1.0, 0.50, 0.01
        )
        manual_dominance = slider_columns[1].slider(
            "Dominance", 0.0, 1.0, 0.50, 0.01
        )
        manual_valence = slider_columns[2].slider(
            "Valence", 0.0, 1.0, 0.50, 0.01
        )
        manual_submit = st.form_submit_button(
            "Interpret dimensions",
            type="primary",
            use_container_width=True,
        )

    if manual_submit:
        try:
            st.session_state["manual_result"] = run_appraisal(
                manual_arousal,
                manual_dominance,
                manual_valence,
            )
        except Exception as exc:
            st.error(f"Appraisal failed: {exc}")

    if "manual_result" in st.session_state:
        render_result(st.session_state["manual_result"])


with method_tab:
    st.markdown("### How the analysis works")
    st.markdown(
        """
        1. **AudEERING Wav2Vec2** estimates continuous Arousal, Dominance, and
           Valence from the uploaded waveform.
        2. The **advanced appraisal engine** normalizes those dimensions using
           its fitted RAVDESS reference speakers.
        3. Fuzzy appraisal states—including threat, challenge, withdrawal
           distress, negative activation, positive activation, and calm
           recovery—are the primary predictions. The strongest state is reported
           as the dominant appraisal pattern with its pattern weight.
        4. Reference similarity, out-of-distribution detection, appraisal
           ambiguity, and interpretation confidence communicate uncertainty.
        """
    )
    st.markdown(
        """
        <p class="method-note">
          RAVDESS and SAVEE contain acted emotion labels, not independently
          measured psychological appraisal labels. These outputs are exploratory
          corpus-relative descriptors, not clinical or psychological assessments.
        </p>
        """,
        unsafe_allow_html=True,
    )