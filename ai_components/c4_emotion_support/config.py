"""Configuration for the C4 demo.

Two separately trained models are wired into this demo:

* the current-emotion classifier (RoBERTa, DailyDialog-derived) predicts 5 labels
* the next-emotion-STATE forecaster (TextCNN / BiLSTM / BiGRU / CNN-BiLSTM,
  trained by ../../../emotion_forecasting_pipeline) predicts 13 labels

The forecaster used to predict a bare next *emotion* in its own 8-label
vocabulary, which meant two things had to be papered over: it answered a
different question from the one the product asks, and its labels had to be
projected onto the classifier's through a lossy 8 -> 5 mapping.

The retrained model answers the product's actual question -- what state does
this emotion move to next -- over 13 `next_emotion_state` classes:

    neutral
    joy / sadness / anger / fear          (onset, only reachable from neutral)
    low_X / high_X for each of the four   (the emotion persists, intensity moves)

and it takes the current emotion in the classifier's own five labels, so the
input-side projection is now the identity. `c4_pipeline/label_mapping.py`
decomposes a state into (base emotion, intensity, trajectory); downstream
stages keep working in the 5-label CLASSIFIER space via the base emotion, and
the trajectory is the new signal the strategy rules use.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
CURRENT_EMOTION_MODEL_PATH = MODELS_DIR / "current_emotion_classifier"
NEXT_EMOTION_MODEL_PATH = MODELS_DIR / "next_emotion_forecaster"

# ----------------------------------------------------------------- label spaces
# Current emotion classifier (id order must match the model's config.json).
EMOTION_LABELS = [
    "neutral",   # 0
    "anger",     # 1
    "fear",      # 2
    "joy",       # 3
    "sadness",   # 4
]

# Next-emotion-STATE forecaster. The id order must match
# `label_classes.json` / `meta_forecast.json`, which preprocess.py writes as
# `sorted(unique(next_emotion_state))` -- so it is alphabetical, not grouped.
# Do not "tidy" this into a nicer order without retraining.
FORECAST_LABELS = [
    "anger",         # 0   onset from neutral
    "fear",          # 1   onset from neutral
    "high_anger",    # 2
    "high_fear",     # 3
    "high_joy",      # 4
    "high_sadness",  # 5
    "joy",           # 6   onset from neutral
    "low_anger",     # 7
    "low_fear",      # 8
    "low_joy",       # 9
    "low_sadness",   # 10
    "neutral",       # 11
    "sadness",       # 12  onset from neutral
]

# The current emotion the forecaster conditions on. Identical to EMOTION_LABELS
# by construction -- that is the point of the retrain, and label_mapping.py
# asserts it rather than trusting it.
FORECAST_CURRENT_EMOTIONS = ["neutral", "anger", "fear", "joy", "sadness"]

# Transition rules of the training corpus (preprocess.validate_transition).
# Deterministic, so the forecaster masks everything outside the reachable set
# instead of hoping the model learned it.
FORECAST_TRANSITIONS = {
    "neutral": ["neutral", "joy", "sadness", "anger", "fear"],
    "anger":   ["neutral", "low_anger", "high_anger"],
    "fear":    ["neutral", "low_fear", "high_fear"],
    "joy":     ["neutral", "low_joy", "high_joy"],
    "sadness": ["neutral", "low_sadness", "high_sadness"],
}

# ----------------------------------------------------------------- forecaster
# Which checkpoint the forecaster loads by default. All four are small enough to
# keep in the repository, so the choice is about accuracy, not size: bigru wins
# the neural leaderboard on test macro-F1 (0.3295) at the smallest footprint
# (1.5 MB). See FORECASTER_METRICS at the bottom of this file, and read the
# note there before quoting any of these numbers as a result.
DEFAULT_FORECASTER = "bigru"
AVAILABLE_FORECASTERS = ["textcnn", "bilstm", "bigru", "cnn_bilstm"]

# The DistilBERT forecaster is gone. It existed to squeeze signal out of the old
# free-text dialogue context; the retrained model reads one utterance plus the
# current emotion, where a 265 MB subword encoder bought nothing measurable.
# `distilbert-base-uncased` is therefore no longer a dependency of this demo.

# Apply the corpus transition rules as a hard mask over the forecast
# distribution. The rules are deterministic, so a state outside the reachable
# set is not an unlikely prediction, it is an invalid one -- masking guarantees
# the pipeline can never be handed "high_joy" for a user who is currently sad.
# Set False to see the unmasked head, which is what the reported test metrics
# measure.
FORECAST_CONSTRAIN_TRANSITIONS = True


def _vendored_or_hub(folder: str, repo_id: str) -> str:
    """Prefer a local copy in models/, fall back to the Hugging Face hub id.

    `vendor_models.py` writes these folders so a second machine can run with no
    network and no Hugging Face cache. A hub id is returned unchanged when no
    local copy exists, so nothing breaks on a machine that never vendored.
    """
    local = MODELS_DIR / folder
    if (local / "config.json").exists():
        return str(local)
    return repo_id


# ----------------------------------------------------------------- deviation
# Valence groups spanning the classifier labels AND the forecaster's next-state
# labels, so deviation tracking and the reply router work on either vocabulary.
POSITIVE = {"joy", "low_joy", "high_joy"}
NEGATIVE = {
    "sadness", "anger", "fear", "disgust",
    "low_sadness", "high_sadness",
    "low_anger", "high_anger",
    "low_fear", "high_fear",
}
NEUTRAL = {"neutral"}
OTHER = {"surprise"}

SAME_EMOTION = 0.0
LOW_DEVIATION = 0.25
MODERATE_DEVIATION = 0.60
HIGH_DEVIATION = 0.90

# ----------------------------------------------------------------- low-VRAM mode
# Set C4_LOW_VRAM=1 for cards around 4 GB (e.g. a laptop RTX 3050).
#
# Measured footprint of the full stack on a 12 GB card: RoBERTa 0.47 GB, TextCNN
# negligible, Qwen3-4B in 4-bit NF4 2.50 GB, peak 3.05 GB allocated / 3.12 GB
# reserved during generation. PyTorch's numbers exclude the CUDA context, which
# costs a further ~0.3-0.6 GB, so the true requirement is ~3.5-3.7 GB. That does
# not reliably fit 4 GB.
#
# Low-VRAM mode moves the classifier and forecaster to the CPU. They are small
# and their latency there is a few tens of milliseconds, and it hands the whole
# ~0.5 GB back to the language model -- which is the component that cannot be
# shrunk without changing what the demo demonstrates.
LOW_VRAM = os.environ.get("C4_LOW_VRAM", "").strip().lower() in ("1", "true", "yes", "on")

# Device for the classifier and forecaster. None lets each pick CUDA if present.
SMALL_MODEL_DEVICE = os.environ.get("C4_SMALL_MODEL_DEVICE") or ("cpu" if LOW_VRAM else None)

# ----------------------------------------------------------------- reply generation
# Stage 6 of the pipeline (see c4_pipeline/reply_graph.py). Qwen3-4B answers,
# with an ESConv QLoRA adapter enabled for distress turns and disabled for
# positive/neutral ones.
#
# -Instruct-2507 is the default rather than the original hybrid `Qwen/Qwen3-4B`
# because it has no thinking mode to suppress: the hybrid checkpoint spends
# tokens on a <think> block before a two-sentence reply, which is pure latency
# in a chat UI. Both are supported; the generator disables thinking for the
# hybrid automatically.
# Resolution order: explicit env var, a vendored copy in models/, then the hub.
# The vendored copy is what makes an offline second machine possible -- see
# vendor_models.py.
LLM_BASE_MODEL = os.environ.get("C4_BASE_MODEL") or _vendored_or_hub(
    "qwen3-4b-instruct", "Qwen/Qwen3-4B-Instruct-2507"
)

# Where the trained LoRA adapter lives. Resolution order: explicit env var, a
# local copy inside this component, then the sibling training project. The demo
# runs without any of them -- base model only, and the sidebar says so.
_ADAPTER_CANDIDATES = [
    Path(os.environ["C4_ADAPTER_PATH"]) if os.environ.get("C4_ADAPTER_PATH") else None,
    MODELS_DIR / "esconv_reply_adapter",
    BASE_DIR.parent.parent.parent
    / "qwen3_esconv_finetune"
    / "outputs"
    / "qwen3-4b-esconv-qlora"
    / "final_adapter",
]
LLM_ADAPTER_PATH = next(
    (path for path in _ADAPTER_CANDIDATES
     if path is not None and (path / "adapter_config.json").exists()),
    _ADAPTER_CANDIDATES[-1],
)

# 4-bit NF4. Qwen3-4B in bf16 needs ~8 GB before activations, which does not sit
# comfortably next to RoBERTa and the forecaster on a 12 GB card; NF4 measures
# 2.50 GB for the base weights.
LLM_LOAD_IN_4BIT = True

# Allow running the 4B model on the CPU when no GPU is present. Off by default
# because it needs ~9 GB of RAM and takes tens of seconds per reply -- for a
# machine without a usable GPU, the template path is usually the better demo.
LLM_ALLOW_CPU = os.environ.get("C4_ALLOW_CPU", "").strip().lower() in ("1", "true", "yes", "on")

# 160 gives a three-sentence reply room to finish. Low-VRAM mode trims it: the
# KV cache grows with generated length, and on a 4 GB card that tail matters.
LLM_MAX_NEW_TOKENS = 96 if LOW_VRAM else 160
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_REPETITION_PENALTY = 1.05
# Retries resample cooler: the first attempt failed the guard, and creativity is
# rarely what was missing.
LLM_RETRY_TEMPERATURE = 0.4
LLM_MAX_RETRIES = 1

# Dialogue turns fed to the model. preprocess.py trained on up to 12, but the
# demo's turns are longer than ESConv's, so 8 keeps the prompt near the length
# the adapter saw. Low-VRAM mode halves it -- a shorter prompt is a smaller KV
# cache, and the adapter still sees a real conversation at 4 turns.
LLM_MAX_CONTEXT_TURNS = 4 if LOW_VRAM else 8

# Output guard. ESConv replies are short -- the corpus median is roughly 20
# words -- so a reply running past four sentences has usually drifted into
# advice-giving or started narrating the prompt.
LLM_MIN_REPLY_WORDS = 3
LLM_MAX_REPLY_SENTENCES = 4

# Whether to append the current/next-emotion signals to the prompt. This block
# is the one part the adapter never saw during fine-tuning; set False to run the
# exact training distribution and A/B the difference.
LLM_INCLUDE_SIGNALS = True

# ----------------------------------------------------------------- XAI
# Token attribution settings. Integrated Gradients needs one forward+backward
# per step. 64 was chosen by measurement, not habit: at 32 steps the relative
# completeness gap on RoBERTa reached 0.30 on some turns (the Riemann sum had
# not converged and the attributions were not trustworthy); 64 brings it under
# 0.01-0.07 across the test conversation.
IG_STEPS = 64
# If a run still misses this, the explainer retries once at double the steps.
IG_MAX_RELATIVE_GAP = 0.15
XAI_TOP_K = 10

# ----------------------------------------------------------------- voice
# Speech in and out, both pinned to the CPU.
#
# The VRAM table above is measured to ~0.1 GB and low-VRAM mode leaves roughly
# 0.8 GB of margin on a 4 GB card. Spending any of that on speech would make the
# 4B reply model -- the component the demo exists to show -- the thing that stops
# fitting. Whisper and Kokoro are small enough that their CPU latency (well under
# a second each) is a fraction of one Qwen turn, so the GPU never has to pay.
VOICE_ENABLED = os.environ.get("C4_VOICE", "").strip().lower() in ("1", "true", "yes", "on")

# ---- speech to text (faster-whisper) ----
# "base.en" over the smaller checkpoints for a reason that is specific to this
# pipeline rather than general: Whisper emits casing and punctuation, which keeps
# transcripts inside the distribution the RoBERTa classifier and the TextCNN
# forecaster were trained on. Lowercase, unpunctuated ASR output (Vosk and
# friends) is a measurable shift in the input those two models see. Drop to
# "tiny.en" (~75 MB) only if transcription latency becomes the bottleneck.
STT_MODEL = os.environ.get("C4_STT_MODEL", "base.en")
STT_DEVICE = "cpu"
STT_COMPUTE_TYPE = "int8"
# Greedy rather than faster-whisper's default beam of 5. Turns here are one or
# two sentences of ordinary speech, where the beam rarely changes the transcript,
# and greedy roughly halves the decode.
STT_BEAM_SIZE = 1
STT_LANGUAGE = "en"

# Whisper reports a mean token log-probability per segment. Clean speech sits
# around -0.5 or better; this threshold is where transcripts start being wrong
# often enough to be worth flagging in the UI. It never blocks a turn -- see
# `VOICE_AUTO_SUBMIT` -- it only annotates one.
STT_LOW_CONFIDENCE_LOGPROB = -0.8

# ---- text to speech (Kokoro-82M ONNX, with a Windows SAPI fallback) ----
# Kokoro is ~6x slower than Piper per second of audio, which does not matter
# here: LLM_MAX_REPLY_SENTENCES caps a reply at four sentences and the ESConv
# median is ~20 words, so a turn is a couple of seconds of synthesis against a
# reply that took Qwen far longer to write. What it buys is a voice that sounds
# supportive rather than synthetic, which is not merely cosmetic for this system.
KOKORO_DIR = MODELS_DIR / "kokoro"
# Release assets ship fp32, fp16 and int8 builds under different names, so this
# is a preference order rather than a requirement -- the loader takes whichever
# build is on disk.
#
# Measured on the target laptop CPU (4 cores / 8 threads), best of three runs,
# synthesising the same 6.8 s reply:
#
#     build   threads   time     RTF
#     int8    default   9.89 s   1.45
#     fp32    4         2.47 s   0.36
#     fp16    4         2.34 s   0.34
#
# int8 is the trap here. It is the smallest download and the obvious default,
# and it is roughly 4x SLOWER than fp16: ONNX int8 on this CPU spends more in
# quantise/dequantise than the narrower weights save. fp16 leads on both speed
# and quality at ~160 MB, so it is the default and int8 is ranked last.
KOKORO_MODEL_PREFERENCE = ["fp16", "", "int8"]
KOKORO_VOICES_FILE = "voices-v1.0.bin"

# onnxruntime defaults to one thread per logical core. On a hyperthreaded 4-core
# laptop that is a measurable loss -- the same reply took 4.29 s at 8 threads
# against 2.47 s at 4 -- because the HT siblings contend rather than help.
# Physical cores, near enough, and capped so a big desktop does not oversubscribe.
KOKORO_INTRA_OP_THREADS = max(1, min(6, (os.cpu_count() or 4) // 2))

# af_heart is Kokoro's highest-rated voice. 0.95 speaks a little under natural
# pace, which reads as unhurried rather than slow and suits the register the
# ESConv adapter was trained to write in.
TTS_VOICE = os.environ.get("C4_TTS_VOICE", "af_heart")
TTS_SPEED = 0.95
TTS_LANGUAGE = "en-us"
# Kokoro degrades on very long inputs, so the synthesiser splits past this and
# concatenates. Four ESConv-length sentences sit well under it; the fixed crisis
# text is the turn that can approach it.
TTS_MAX_CHARS_PER_CHUNK = 400
TTS_SAMPLE_WIDTH = 2  # 16-bit PCM

# ---- interaction ----
# Push-to-talk with no confirmation step: the recording is transcribed and sent
# in one gesture. The cost is that an ASR error reaches the pipeline unreviewed,
# including safety.py's crisis regex, where a mis-transcription is a missed
# detection. That is mitigated rather than prevented -- the transcript is shown
# in the user's own bubble, low-confidence turns are captioned, and the last turn
# can be undone -- because a confirm-and-send step removes the point of talking.
VOICE_AUTO_SUBMIT = True
VOICE_AUTOPLAY = True

# Reply budget while voice mode is on. Generation is 60-80% of a spoken turn's
# latency on a 4 GB laptop card, so this is the largest responsiveness lever the
# demo has -- and it cuts in the right direction anyway, since a reply that is
# listened to wants to be shorter than one that is read.
LLM_VOICE_MAX_NEW_TOKENS = 64

# ----------------------------------------------------------------- reported metrics
# Test-set numbers from the training runs, surfaced in the UI so the demo cannot
# silently drift from what was actually measured.
CLASSIFIER_METRICS = {
    "model": "roberta-base",
    "test_accuracy": 0.8214,
    "test_macro_f1": 0.8162,
    "dataset": "DailyDialog-derived sentiment (5 classes)",
}

# Next-emotion-STATE forecaster, 13 classes, 1501 held-out rows. Every model in
# the pipeline is listed, including the two the demo does not serve, because the
# comparison is the finding.
#
# READ THIS BEFORE QUOTING THE NUMBERS. Accuracy clusters at 0.34-0.36 for all
# six trained models -- and `baseline_prior_by_current_emotion`, which reads no
# text whatsoever and just applies the transition rules plus the training prior,
# scores 0.3438. The models are not beating it on accuracy. They beat it on
# macro-F1 (0.30-0.34 against 0.18) only because balanced class weights spread
# predictions across the reachable states instead of collapsing onto the
# majority one.
#
# That is a property of the corpus, not of the training. The `next_emotion_state`
# labels are synthetic and were not conditioned on the utterance: within every
# current-emotion group a TF-IDF model on the text alone scores at or below the
# majority baseline (measured: 0.335 against 0.372 for intensity). The one
# column that does predict the target is `context_text`, at 1.0000 test
# accuracy -- it was generated from the target and the pipeline excludes it as
# leakage.
#
# So: the forecaster reliably encodes WHICH next states are possible and their
# base rates. It does not, on this data, read a sentence and tell you whether
# distress is about to escalate. Replacing the labels with observed sequential
# annotations is what would change that, not a bigger model.
#
# ROC-AUC ~0.90 alongside ~0.35 accuracy is the signature of exactly this: the
# ranking correctly separates the 3 reachable states from the 10 unreachable
# ones, and is near chance within the reachable set.
FORECASTER_METRICS = {
    # --- served by this demo (PyTorch) ---
    "bigru": {"test_accuracy": 0.3538, "test_macro_f1": 0.3295,
              "test_roc_auc": 0.8989, "constrained_accuracy": 0.3538, "size_mb": 1.51},
    "bilstm": {"test_accuracy": 0.3411, "test_macro_f1": 0.3137,
               "test_roc_auc": 0.8996, "constrained_accuracy": 0.3411, "size_mb": 1.58},
    "cnn_bilstm": {"test_accuracy": 0.3418, "test_macro_f1": 0.3131,
                   "test_roc_auc": 0.8983, "constrained_accuracy": 0.3431, "size_mb": 4.15},
    "textcnn": {"test_accuracy": 0.3444, "test_macro_f1": 0.3013,
                "test_roc_auc": 0.9009, "constrained_accuracy": 0.3458, "size_mb": 2.99},
    # --- trained by the same pipeline, not served (scikit-learn) ---
    "linear_svm": {"test_accuracy": 0.3578, "test_macro_f1": 0.3381},
    "logistic_regression": {"test_accuracy": 0.3538, "test_macro_f1": 0.3281},
    # --- references ---
    "baseline_prior_by_current_emotion": {"test_accuracy": 0.3438, "test_macro_f1": 0.1757},
    "baseline_majority": {"test_accuracy": 0.2432, "test_macro_f1": 0.0301},
}

# Shown verbatim in the sidebar. The demo should not be able to drift from what
# was actually measured, and it should not be able to overstate it either.
FORECASTER_CAVEAT = (
    "Trained on synthetic, rule-constrained labels. The model learns which "
    "next states are reachable and their base rates; on this corpus the "
    "utterance text does not separate them (no trained model beats a "
    "text-free prior on accuracy). Treat a forecast as structure plus base "
    "rate, not as evidence about this sentence."
)
