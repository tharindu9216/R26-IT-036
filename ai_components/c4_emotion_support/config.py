"""Configuration for the C4 demo.

Two separately trained models are wired into this demo and they do NOT share a
label space:

* the current-emotion classifier (RoBERTa, DailyDialog-derived) predicts 5 labels
* the next-emotion forecaster (TextCNN / BiLSTM / DistilBERT, conversational
  corpus) predicts 8 labels

`c4_pipeline/label_mapping.py` bridges the two. Everything downstream of the
classifier (deviation tracking, strategy selection) works in the 5-label
CLASSIFIER space; the forecaster's native 8 labels are shown as-is for detail
and projected down when a strategy has to be picked.
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

# Next-emotion forecaster (id order must match artifacts/meta_forecast.json).
FORECAST_LABELS = [
    "angry",     # 0
    "anxious",   # 1
    "calm",      # 2
    "excited",   # 3
    "happy",     # 4
    "neutral",   # 5
    "sad",       # 6
    "stressed",  # 7
]

# ----------------------------------------------------------------- forecaster
# Which checkpoint the forecaster loads by default. "textcnn" is the demo
# default: 0.6 MB, no external base model to download, and its test macro-F1
# (0.7003) is within noise of DistilBERT's (0.7025) at ~450x the file size.
DEFAULT_FORECASTER = "textcnn"
AVAILABLE_FORECASTERS = ["textcnn", "bilstm", "distilbert"]


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


DISTILBERT_BASE = _vendored_or_hub("distilbert-base-uncased", "distilbert-base-uncased")

# Number of previous dialogue turns concatenated into the forecaster's input.
# Must match `--context` used in preprocess.py (meta_forecast.json records it).
FORECAST_CONTEXT_TURNS = 3

# ----------------------------------------------------------------- deviation
# Valence groups spanning BOTH label spaces so deviation tracking works on
# either vocabulary.
POSITIVE = {"joy", "happy", "excited"}
NEGATIVE = {"sadness", "anger", "fear", "disgust", "sad", "angry", "anxious", "stressed"}
NEUTRAL = {"neutral", "calm"}
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

FORECASTER_METRICS = {
    "textcnn": {"test_accuracy": 0.7217, "test_macro_f1": 0.7003},
    "bilstm": {"test_accuracy": 0.7164, "test_macro_f1": 0.6903},
    "distilbert": {"test_accuracy": 0.7177, "test_macro_f1": 0.7025},
    "baseline_persistence": {"test_accuracy": 0.7270, "test_macro_f1": 0.7098},
    "baseline_majority": {"test_accuracy": 0.2437, "test_macro_f1": 0.0490},
}
