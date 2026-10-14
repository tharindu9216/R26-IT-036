# C4 — Emotion Forecasting and Supportive Dialogue

Runnable Streamlit prototype for Component C4, wired to the two models actually
trained for this component and instrumented with Explainable AI at every stage.

## Status

| Stage | Status | Detail |
|---|---|---|
| Current emotion classifier — train & evaluate | **Done** | `../../../classification/`, 4 models + 2 baselines compared under Optuna |
| Next-emotion-**state** forecaster — train & evaluate | **Done** | `../../../emotion_forecasting_pipeline/`, 6 models + 2 baselines |
| Classifier integrated into the demo | **Done** | RoBERTa checkpoint loads and runs on GPU |
| Forecaster integrated into the demo | **Done** | TextCNN / BiLSTM / BiGRU / CNN-BiLSTM selectable in the sidebar |
| Forecast drives the strategy rules | **Done** | rules read the trajectory (escalating vs easing), not just the emotion |
| Deviation tracking, strategy selection, safety, templates | **Done** | rule-based, fully inspectable |
| Explainable AI | **Done** | Integrated Gradients, occlusion, counterfactual probing, rule traces |
| Reply generation — LangGraph orchestration | **Done** | `c4_pipeline/reply_graph.py`, routing + guard + retry |
| Reply generation — Qwen3-4B wiring | **Done** | 4-bit NF4 on the 3080, 1.4–4.3 s per reply, verified live |
| ESConv LoRA adapter trained | **Done** | eval loss 2.091, perplexity 8.094, 126 MB, peak 5.46 GB |
| End-to-end verification | **Done** | `smoke_test.py --all` and `--llm` — all checks pass |
| Forecaster beating a text-free prior on accuracy | **Open** | no model does; see the measurement below — it is a label problem, not a model problem |
| Real-corpus forecast labels (MELD / EmpatheticDialogues) | **Open** | the blocker: `next_emotion_state` is synthetic and not conditioned on the utterance |
| Adapter reply length calibration | **Open** | adapter averages 12.0 words vs ESConv's 18.9 — see below |

## The two models

### Current emotion classifier — RoBERTa

Trained in `../../../classification/training.ipynb` on a DailyDialog-derived
sentiment set. `roberta-base` won the comparison:

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| **roberta-base** | **0.8214** | **0.8162** |
| bert-base-uncased | 0.8087 | 0.8034 |
| tfidf + LinearSVC | 0.6864 | 0.6859 |
| tfidf + LogReg | 0.6786 | 0.6767 |
| microsoft/deberta-v3-base | 0.1417 | 0.0497 |

The DeBERTa run failed — 0.14 accuracy on 5 classes is at chance, so that trial
did not train, most likely a tokenizer/`sentencepiece` problem rather than a
modelling result. It is reported as measured rather than dropped.

**Labels (5):** `neutral, anger, fear, joy, sadness`

### Next emotion **state** forecaster — TextCNN / BiLSTM / BiGRU / CNN-BiLSTM

Retrained in `../../../emotion_forecasting_pipeline/`. The previous forecaster
predicted which *emotion* came next; this one predicts the next emotional
**state** — the emotion together with where its intensity is going.

```
neutral
joy | sadness | anger | fear        onset   (only reachable from neutral)
low_X | high_X                      the emotion persists, intensity moves
```

That distinction is the point. `low_sadness` and `high_sadness` are the same
emotion and opposite situations: one is distress that is lifting, the other is
distress that is deepening. A bare emotion label cannot tell a support system
which of those it is looking at, so `Comfort` and `Encourage` used to fire on
essentially the same evidence.

**Model input:** the current utterance + the current emotion. Nothing else.

**Labels (13):** `neutral`, `joy`/`sadness`/`anger`/`fear`,
`low_*`/`high_*` for each of those four.

#### Results — read the caveat before quoting these

1501 held-out rows, 13 classes.

| Model | Test accuracy | Test macro-F1 | ROC-AUC (OvR) |
|---|---|---|---|
| linear_svm (TF-IDF) | **0.3578** | **0.3381** | 0.9006 |
| **bigru** (demo default) | 0.3538 | 0.3295 | 0.8989 |
| logistic_regression (TF-IDF) | 0.3538 | 0.3281 | 0.9011 |
| bilstm | 0.3411 | 0.3137 | 0.8996 |
| cnn_bilstm | 0.3418 | 0.3131 | 0.8983 |
| textcnn | 0.3444 | 0.3013 | 0.9009 |
| baseline_prior_by_current_emotion | 0.3438 | 0.1757 | — |
| baseline_majority | 0.2432 | 0.0301 | — |

`baseline_prior_by_current_emotion` reads **no text at all** — it applies the
corpus transition rules plus the training base rates. It scores 0.3438, which
is inside a percentage point of every trained model. The models beat it on
macro-F1 (0.30–0.34 vs 0.18) only because balanced class weights spread
predictions across the reachable states instead of collapsing onto the majority
one.

**Why:** the `next_emotion_state` labels are synthetic and were not conditioned
on the utterance. Measured directly — within every current-emotion group a
TF-IDF model on the text alone scores at or below the majority baseline (0.335
vs 0.372 for intensity). The one column that *does* predict the target is
`context_text`, at **1.0000** test accuracy; it was generated from the target
and the pipeline excludes it as leakage.

ROC-AUC ≈ 0.90 next to accuracy ≈ 0.35 is the signature of exactly this: the
ranking cleanly separates the 3 reachable states from the 10 unreachable ones,
and is near chance within the reachable set.

**So what the forecaster is:** a reliable model of which next states are
possible and how often each occurs. It is not, on this data, a model that reads
a sentence and tells you distress is about to escalate. Replacing the labels
with observed sequential annotations is what changes that — not a bigger model.
The sidebar says all of this in the app.

#### The transition constraint

The corpus transition rules are deterministic, so the demo masks unreachable
states out of the distribution rather than hoping the model learned them:

```
neutral -> neutral | joy | sadness | anger | fear
sadness -> neutral | low_sadness | high_sadness
anger   -> neutral | low_anger   | high_anger
fear    -> neutral | low_fear    | high_fear
joy     -> neutral | low_joy     | high_joy
```

The pipeline can therefore never be handed `high_joy` for a user who is
currently sad. `smoke_test.py` asserts it. Toggle with
`FORECAST_CONSTRAIN_TRANSITIONS` in `config.py`; the reported metrics above are
unmasked.

### The label spaces now line up

The retrain removed the old 5 ↔ 8 projection. The forecaster conditions on the
current emotion in the **classifier's own five labels**, so the input side is
the identity, and `label_mapping.assert_label_spaces_aligned()` fails loudly at
import if that ever stops being true.

The output side decomposes exactly:

```
next state       base emotion   intensity   trajectory (with current emotion)
high_sadness  -> sadness        high        escalating
low_sadness   -> sadness        low         easing
neutral       -> neutral        —           resolving / steady
fear          -> fear           —           onset
```

Nothing is merged, so nothing is lost: the projection to 5 labels drops the
intensity, and the intensity is shown alongside rather than discarded. The
trajectory is what the strategy rules read — see `c4_pipeline/strategy_selector.py`.

### Reply generation — Qwen3-4B + ESConv LoRA

The third model. `Qwen/Qwen3-4B-Instruct-2507` in 4-bit NF4, with a QLoRA
adapter fine-tuned on ESConv (`../../../qwen3_esconv_finetune/`), orchestrated by
LangGraph in `c4_pipeline/reply_graph.py`.

**Why a graph and not a call.** Three decisions have to happen around the
generation, and two of them are conditional:

```
START -> route_request --crisis--> crisis_reply ------------------> END
             |
             | adapter / base
             v
         plan_response -> build_prompt -> generate_reply -> guard_reply
                              ^                                 |
                              |                                 |--pass--> finalize -> END
                              +-------------retry---------------|
                                                                |
                                                                +--fail--> template_fallback -> END
```

**1. Routing.** The ESConv EDA found no positive emotion label anywhere in the
corpus — all 11 are distress states (anxiety, depression, sadness, anger, fear,
shame, disgust, nervousness, guilt, jealousy, pain). An adapter trained only on
distress answers a piece of good news with consolation, so positive and neutral
turns run through the *base* model with the adapter switched off. A turn that
reads neutral but is *forecast* to deteriorate still takes the adapter route —
this is where the C4 forecaster earns its place in the pipeline.

**2. Crisis turns never reach the model.** A fluent, plausible, slightly-wrong
sentence is worse than a rigid correct one here, so the crisis branch returns
fixed text and exits after two nodes. The path is visible in the UI.

**3. The output is guarded, and failure loops back.** Generated replies are
checked for leaked internal vocabulary (strategy names, "the classifier
predicted", "your current emotion is"), diagnosis language, and length. A
failure re-enters `build_prompt` with the specific complaint and resamples at a
lower temperature; a second failure falls back to the strategy template. Bare
emotion words are deliberately *not* flagged — "I hear how sad you feel" is
ordinary supportive speech.

**Strategy vocabularies are translated, not passed through.** The adapter was
fine-tuned with ESConv's own 8-strategy annotation in the `Response strategy:`
slot, so C4's names are mapped in `c4_pipeline/strategy_mapping.py` before
prompting:

| C4 strategy | ESConv strategy | Why |
|---|---|---|
| Listen | Question | listening is enacted by asking, not asserting |
| Comfort | Reflection of feelings | comfort is acknowledgement — mirroring the emotion |
| Reassure | Affirmation and Reassurance | direct match |
| Encourage | Providing Suggestions | fires when the forecast improves, so the move is forward-looking |
| Maintain Tone | Restatement or Paraphrasing | adds no new emotional colour |
| Safe Fallback | *(never generated)* | crisis branch |

Passing "Maintain Tone" straight into that slot would put the model on a token
it never saw in that position, and the conditioning degrades to noise.

**One model, both routes.** The base weights load once and PEFT's
`disable_adapter()` switches the LoRA off for the base route. Two separate 4-bit
copies would cost ~6 GB before the classifier and forecaster are counted; one
copy plus an adapter costs ~3.2 GB, which is what makes this fit alongside
everything else on a 12 GB card.

**Without a trained adapter** the graph runs base-model-only, and without a GPU
it runs templates. Both states are reported in the sidebar rather than hidden.

Measured on the RTX 3080 (4-bit): 1.3–3.7 s per reply at 11–25 generated tokens,
~3.2 GB VRAM resident alongside RoBERTa and the BiGRU forecaster.

### What the adapter actually changed

Same prompts, same loaded weights, greedy decoding, `disable_adapter()` the only
difference:

| Prompt | Base | Adapter |
|---|---|---|
| "I feel hopeless about my exams…" | 62 words: *"It sounds like you're carrying a lot of weight right now… Would you like to talk more about what's specifically making it hard for you?"* | 6 words: *"I can understand how you feel."* |
| "My manager keeps criticising me…" | 92 words, ending *"…I'd be happy to help you think through how to handle this. You've got this."* | 14 words: *"I am sorry to hear that. It sounds like you are being treated unfairly."* |
| "I have not spoken to my best friend…" | 22 words: *"Have you thought about reaching out to her lately?"* | 8 words: *"Why haven't you spoken to your best friend?"* |

**Mean reply length: base 56.6 words → adapter 12.0 words.** The fine-tuning
took hold and in the intended direction: ESConv supporters do not deliver
92-word coaching paragraphs, they reflect briefly and hand the turn back.

**But it overshot.** The ESConv training split averages 18.9 words; the adapter
produces 12.0. Some replies are too terse to be useful — *"I can understand how
you feel"* is thinner than the corpus it was trained on would justify, and one
Reassure case produced the off-target *"I hope you are enjoying this time of
year."* Two contributing factors: that comparison used greedy decoding, which
biases short and generic (the demo runs at temperature 0.7 and is more varied),
and roughly 18% of ESConv supporter turns are annotated `Others`, many of them
short acknowledgements.

Worth trying if you revisit this: filter very short training replies with
`--min-reply-words 5` in `preprocess.py`, or drop the `Others` strategy from the
prompt conditioning. Neither is done yet — the current adapter is reported as
measured.

## Explainable AI

Implemented directly on PyTorch in `c4_pipeline/xai.py`; no captum, shap or lime
dependency.

| Method | Applies to | What it answers |
|---|---|---|
| **Integrated Gradients** | classifier + forecaster | which tokens pushed the prediction, and how hard |
| **Occlusion** (leave-one-out) | classifier | the same question without trusting gradients — a cross-check |
| **Aux-feature ablation** | forecaster | how much of the forecast is the text vs the classifier's output |
| **Counterfactual sweep** | forecaster | what the forecast would be under each of the 5 possible current emotions |
| **Vocabulary coverage** | forecaster | which words fell outside the 4,718-token training vocabulary |
| **Rule trace** | strategy selector | every rule, whether it fired, and which ones were reached |

Two details worth knowing when reading the output:

- **IG completeness is checked, not assumed.** The Riemann sum is validated
  against `sum(attributions) == F(x) - F(baseline)`; if the relative gap exceeds
  15% the explainer automatically retries at double the step count. At the
  original 32 steps some turns reached a 30% gap — attributions that look
  plausible and are wrong.
- **The forecaster's vocabulary is 4,718 tokens**, built from the training
  split of a ~10k-row corpus of real DailyDialog-style sentences. Free-text
  input still hits `<unk>` sometimes, and an attribution on an unknown token
  describes the unknown-token embedding, not the word. The UI warns with the
  exact OOV list. (The previous forecaster had 528 tokens and hit `<unk>`
  constantly, so this is much improved rather than solved.)
- **Read the forecaster's token attributions with the label caveat in mind.**
  On this corpus the utterance does not separate the reachable next states, so
  a large attribution marks a token the model *reacts to*, not evidence that
  the token predicts escalation. The aux-feature ablation is the honest
  comparison: removing the current emotion typically moves the forecast far
  more than any word does.

## Structure

```text
c4_emotion_support/
├── streamlit_app.py             demo UI
├── smoke_test.py                headless end-to-end check
├── config.py                    label spaces, paths, XAI settings, reported metrics
├── c4_pipeline/
│   ├── pipeline.py              the full turn
│   ├── emotion_classifier.py    RoBERTa wrapper + keyword fallback
│   ├── emotion_forecaster.py    .pt checkpoint wrapper + persistence fallback
│   ├── forecast_models.py       GENERATED by emotion_forecasting_pipeline/export_to_c4.py
│   ├── label_mapping.py         state -> (emotion, intensity, trajectory) decomposition
│   ├── xai.py                   IG, occlusion, counterfactuals, vocab coverage
│   ├── deviation_tracker.py     turn-to-turn emotion deviation
│   ├── strategy_selector.py     rules + evaluation trace
│   ├── reply_graph.py           LangGraph reply generation: route, prompt, guard, retry
│   ├── qwen_generator.py        Qwen3-4B loader, 4-bit, adapter on/off per turn
│   ├── strategy_mapping.py      C4 <-> ESConv strategy bridge + route selection
│   ├── response_generator.py    strategy templates (fallback)
│   ├── voice.py                 speech in (Whisper) and out (Kokoro / SAPI)
│   └── safety.py                regex crisis screen
├── models/                      generated, gitignored — see models/README.md
└── sample_outputs/
    └── sample_conversation_trace.json   regenerated from a real run
```

## Setup and run

```bash
pip install -r requirements.txt
python smoke_test.py          # verify the models load and the pipeline works
python smoke_test.py --llm    # also load Qwen3-4B and generate for real
streamlit run streamlit_app.py
```

`--llm` is the one that proves the reply stage end to end: it loads the model,
runs all five routing cases through the graph, prints each generated reply, and
asserts the adapter was enabled on exactly the distress routes.

Model files go in `models/` — see `models/README.md`. With them missing the app
still runs on a keyword classifier and a persistence heuristic, both flagged in
the sidebar.

### Reply generation

Qwen3-4B downloads from Hugging Face on first use (~8 GB) and needs a CUDA GPU.
Turn it off with the **Generate replies with Qwen3-4B** sidebar toggle to run on
templates instead — the graph handles the absence, no code change needed.

The LoRA adapter is found automatically at the first of these that exists:

1. `$env:C4_ADAPTER_PATH`
2. `models/esconv_reply_adapter/`
3. `../../../qwen3_esconv_finetune/outputs/qwen3-4b-esconv-qlora/final_adapter/`

To train it (RTX 3080 12 GB profile, ~2–3 hours):

```powershell
cd ..\..\..\qwen3_esconv_finetune
.\run_smoke_test.ps1        # 5 steps, catches OOM and setup errors first
.\run_train_rtx3080.ps1
```

Until then the base model answers both routes, and the sidebar says so.

### Running on a small GPU (4 GB, e.g. a laptop RTX 3050)

Measured VRAM for the full stack, on the demo's default settings:

| Stage | VRAM |
|---|---|
| RoBERTa classifier | 0.47 GB |
| BiGRU state forecaster | negligible (1.5 MB) |
| Qwen3-4B, 4-bit NF4 | 2.50 GB |
| **Peak during generation** | **3.05 GB allocated / 3.12 GB reserved** |

PyTorch's figures exclude the CUDA context, which costs a further ~0.3–0.6 GB,
so the real requirement is ~3.5–3.7 GB. **That does not reliably fit a 4 GB
card.** Set low-VRAM mode:

```powershell
$env:C4_LOW_VRAM = "1"
streamlit run streamlit_app.py
```

```bash
C4_LOW_VRAM=1 streamlit run streamlit_app.py
```

It moves the classifier and forecaster to the CPU and trims the reply budget
(160 → 96 new tokens) and context (8 → 4 turns), which brings the measured peak
to **2.53 GB allocated / 2.56 GB reserved** — roughly 3.0–3.2 GB including the
CUDA context, leaving ~0.8 GB of margin on a 4 GB card.

The cost of moving those two models off the GPU is small, because they are
small:

| | on GPU | on CPU |
|---|---|---|
| Classify a turn | 7.6 ms | 21.5 ms |
| Forecast next emotion | 1.3 ms | 0.5 ms |
| Classifier XAI (IG 64 steps + occlusion) | 1.10 s | 2.94 s |

Only the XAI path is noticeably slower, and it already has a sidebar toggle.
Predictions are identical — the classifier returns `sadness (0.96)` on either
device.

Verify on the target machine before demoing:

```bash
C4_LOW_VRAM=1 python smoke_test.py --llm
```

The smoke test honours the same flag, so it rehearses the real configuration.

**If it still runs out of memory**, in order:

1. Close anything else on the GPU. On a laptop, check whether the display is
   driven by the dGPU or the Intel iGPU — Optimus laptops leave nearly the full
   4 GB free, dGPU-driven displays do not.
2. Turn off **Compute explanations (XAI)** in the sidebar.
3. Turn off **Generate replies with Qwen3-4B**. The demo still runs the
   classifier, forecaster, deviation tracking, strategy rules, safety screen and
   all the XAI — only the replies fall back to templates.
4. Use a smaller base model: `$env:C4_BASE_MODEL = "Qwen/Qwen3-1.7B"`. Note that
   **the ESConv adapter will not load against a different base** — LoRA weights
   are shaped to the model they were trained on, so you would need to retrain
   with `--model-id Qwen/Qwen3-1.7B`.
5. As a last resort, `$env:C4_ALLOW_CPU = "1"` runs the 4B model on the CPU. It
   needs ~9 GB of RAM and takes tens of seconds per reply; the sidebar warns
   when this is active.

Anything you set through these variables is reported in the sidebar, so a demo
never silently misrepresents which configuration produced its output.

For a full walkthrough of moving the demo to another machine — which model files
to copy, installing a CUDA PyTorch build, and verifying before you present — see
[SETUP_SECOND_MACHINE.md](SETUP_SECOND_MACHINE.md).

## Voice mode

Push-to-talk speech in and spoken replies out, off by default:

```powershell
python vendor_models.py --voice      # ~310 MB, one time
$env:C4_VOICE = "1"
streamlit run streamlit_app.py
```

Or leave the environment variable alone and flip **Voice mode (push to talk)**
in the sidebar. Record, stop, and the turn is transcribed and sent in one
gesture — there is no confirm-and-send step, because a confirmation removes the
point of talking.

### What it costs

Speech does not touch the GPU. That is the whole design constraint: the VRAM
table above leaves ~0.8 GB of margin in low-VRAM mode, and spending it here
would take it from the 4B reply model, which is the one component that cannot be
made smaller without changing what the demo demonstrates.

| Stage | Model | Where | Measured |
|---|---|---|---|
| Speech in | faster-whisper `base.en`, int8 | CPU | ~0.8 s for 4 s of audio |
| Speech out | Kokoro-82M, fp16 ONNX | CPU | RTF 0.35 (a 6.9 s reply in 2.4 s) |
| Reply | Qwen3-4B NF4 | GPU | unchanged |

A spoken turn end to end measured **~4 s** on the reference laptop (i5-11th gen,
RTX 2050 4 GB) — most of it generation, not speech.

Two settings exist to keep it that way, and both are visible in the sidebar:

* **Explanations are paused while voice mode is on.** `IG_STEPS` is 64
  forward+backward passes across two models. That is affordable when you are
  reading a trace and fatal when you are waiting to be answered. Switching voice
  mode off restores whatever the XAI toggle was set to.
* **The reply budget drops to `LLM_VOICE_MAX_NEW_TOKENS` (64).** Generation is
  most of a spoken turn's latency, so this is the largest lever available — and
  it cuts the right way anyway, since a reply that is listened to wants to be
  shorter than one that is read.

### Why these two models

**faster-whisper over whisper.cpp** — the same speed on this class of CPU, but
it decodes from an in-memory buffer instead of a temp WAV. The recording is
someone describing their distress; whisper.cpp's Windows path is a subprocess
over a file, which gives that up for about 0.2 s.

**Whisper over Vosk** — Whisper emits casing and punctuation, which keeps
transcripts inside the distribution the RoBERTa classifier and the TextCNN
forecaster were trained on. Lowercase, unpunctuated ASR output is a real shift
in what those two models see, for 100 MB saved.

**Kokoro fp16, not int8** — int8 is the smallest download and the obvious
default, and it measured **~4x slower**: RTF 1.45 against 0.34, because ONNX
int8 on this CPU spends more in quantise/dequantise than the narrower weights
save. onnxruntime's thread default costs another 40% on a hyperthreaded laptop
(8 threads ran the same reply in 4.29 s against 2.47 s at 4), so the session is
built at one thread per physical core. Both numbers are in `config.py`.

**A Windows SAPI fallback under Kokoro** — the realistic failure mode for neural
TTS on Windows is not quality or speed, it is the phonemiser refusing to load on
a machine that was never set up for it. If Kokoro does not load, `voice.py`
falls back to .NET's `System.Speech` through a fresh PowerShell process, which
needs no package at all. It sounds worse and it always works; the sidebar says
which one is live.

### The tradeoff this makes

Auto-submit means a transcription error reaches the pipeline unreviewed —
including `safety.py`'s crisis regex, where a mis-hearing is a missed detection,
and a miss is the expensive failure. That is mitigated rather than prevented:

* the transcript is shown in the user's own bubble, so an error is visible;
* Whisper's mean token log-probability is recorded on every voice turn and a
  doubtful one is captioned in the chat;
* **Undo turn** removes the last exchange before it can bias the forecaster's
  context or the emotion chart.

Every trace records `input_modality` and, for spoken turns, a `voice_input`
block with the backend, the model, the clip duration and the confidence — so
voice and typed turns can be reported separately rather than assumed equivalent.

Prosody is discarded. Emotion is still classified from text alone, which means
the richest emotional signal in a spoken turn is thrown away before Stage 1.
That is a deliberate scope limit, not an oversight — an audio-emotion branch is
a different piece of work — and it is the first thing to say about these results.

Verify the whole path end to end, which is the check worth running on a second
machine:

```powershell
python smoke_test.py --voice
```

It synthesises a line, transcribes it back, and asserts the round trip recovers
the sentence. Both backends can fail independently and neither failure is loud.

## What the demo shows

Send something like:

```text
I feel sad and stressed about my exams.
I am getting scared that I will fail.
I feel a little better after talking.
```

and each turn reports the current emotion with its full distribution, the
deviation from the previous turn, the forecast next **state** with its
trajectory (escalating / easing / resolving / onset / steady) and its
projection onto the 5 classifier labels, the selected strategy with the rule
trace behind it, the supportive response, per-token explanations for both
models, and the full JSON trace.

## Research disclaimer

Academic prototype. Not a clinical or therapy system, and the safety screen is a
keyword guard rail, not a risk assessment. For serious mental health concerns,
seek support from qualified professionals.
