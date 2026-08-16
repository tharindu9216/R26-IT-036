# C4 — Emotion Forecasting and Supportive Dialogue

Runnable Streamlit prototype for Component C4, wired to the two models actually
trained for this component and instrumented with Explainable AI at every stage.

## Status

| Stage | Status | Detail |
|---|---|---|
| Current emotion classifier — train & evaluate | **Done** | `../../../classification/`, 4 models + 2 baselines compared under Optuna |
| Next-turn emotion forecaster — train & evaluate | **Done** | `../../../forcasting/`, 5 models + 2 baselines, 30 Optuna trials each |
| Classifier integrated into the demo | **Done** | RoBERTa checkpoint loads and runs on GPU |
| Forecaster integrated into the demo | **Done** | TextCNN / BiLSTM / DistilBERT selectable in the sidebar |
| Deviation tracking, strategy selection, safety, templates | **Done** | rule-based, fully inspectable |
| Explainable AI | **Done** | Integrated Gradients, occlusion, counterfactual probing, rule traces |
| Reply generation — LangGraph orchestration | **Done** | `c4_pipeline/reply_graph.py`, routing + guard + retry |
| Reply generation — Qwen3-4B wiring | **Done** | 4-bit NF4 on the 3080, 1.4–4.3 s per reply, verified live |
| ESConv LoRA adapter trained | **Done** | eval loss 2.091, perplexity 8.094, 126 MB, peak 5.46 GB |
| End-to-end verification | **Done** | `smoke_test.py --all` and `--llm` — all checks pass |
| Forecaster beating the persistence baseline | **Open** | best trained macro-F1 0.7025 vs baseline 0.7098 |
| Real-corpus forecaster (MELD / EmpatheticDialogues) | **Open** | current forecaster is trained on a synthetic corpus |
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

### Next emotion forecaster — TextCNN / BiLSTM / DistilBERT

Trained in `../../../forcasting/` on a synthetic conversational corpus with
conversation-level splits. Each model gets the previous 3 dialogue turns plus
the speaker's current emotion as an auxiliary feature.

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| baseline_persistence | 0.7270 | **0.7098** |
| distilbert | 0.7177 | 0.7025 |
| **textcnn** (demo default) | 0.7217 | 0.7003 |
| bilstm | 0.7164 | 0.6903 |
| tfidf + LightGBM | 0.7190 | 0.6791 |
| tfidf + LogReg | 0.6911 | 0.6570 |
| baseline_majority | 0.2437 | 0.0490 |

**No trained model beats persistence yet**, and the demo says so in the sidebar.
Emotion has strong inertia, so "they will feel the same next turn" is a hard
bar. TextCNN is the default because it is within 0.002 macro-F1 of DistilBERT at
1/450th the file size and needs no base model download.

**Labels (8):** `angry, anxious, calm, excited, happy, neutral, sad, stressed`

### The label spaces do not match

The two models were trained on different corpora with different label sets, so
`c4_pipeline/label_mapping.py` bridges them:

```
classifier -> forecaster (aux feature)   forecaster -> classifier (strategy rules)
  neutral -> neutral                       angry, -> anger
  anger   -> angry                         anxious, stressed -> fear
  fear    -> anxious                       calm, neutral -> neutral
  joy     -> happy                         excited, happy -> joy
  sadness -> sad                           sad -> sadness
```

Both directions are lossy — `fear` covers both `anxious` and `stressed` — and
the UI flags a projection when it merges states the 5-label space cannot
represent. It is not hidden.

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
~3.2 GB VRAM resident alongside RoBERTa and the TextCNN forecaster.

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
| **Counterfactual sweep** | forecaster | what the forecast would be under each of the 8 possible current emotions |
| **Vocabulary coverage** | forecaster | which words fell outside the 528-token training vocabulary |
| **Rule trace** | strategy selector | every rule, whether it fired, and which ones were reached |

Two details worth knowing when reading the output:

- **IG completeness is checked, not assumed.** The Riemann sum is validated
  against `sum(attributions) == F(x) - F(baseline)`; if the relative gap exceeds
  15% the explainer automatically retries at double the step count. At the
  original 32 steps some turns reached a 30% gap — attributions that look
  plausible and are wrong.
- **The forecaster's vocabulary is only 528 tokens**, because its training
  corpus is template-generated. Free-text input hits `<unk>` often, and an
  attribution on an unknown token describes the unknown-token embedding, not the
  word. The UI warns with the exact OOV list.

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
│   ├── forecast_models.py       architectures mirrored from forcasting/train_deep.py
│   ├── label_mapping.py         5 <-> 8 label bridge
│   ├── xai.py                   IG, occlusion, counterfactuals, vocab coverage
│   ├── deviation_tracker.py     turn-to-turn emotion deviation
│   ├── strategy_selector.py     rules + evaluation trace
│   ├── reply_graph.py           LangGraph reply generation: route, prompt, guard, retry
│   ├── qwen_generator.py        Qwen3-4B loader, 4-bit, adapter on/off per turn
│   ├── strategy_mapping.py      C4 <-> ESConv strategy bridge + route selection
│   ├── response_generator.py    strategy templates (fallback)
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
| TextCNN forecaster | negligible |
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

## What the demo shows

Send something like:

```text
I feel sad and stressed about my exams.
I am getting scared that I will fail.
I feel a little better after talking.
```

and each turn reports the current emotion with its full distribution, the
deviation from the previous turn, the forecast next emotion in both label
spaces, the selected strategy with the rule trace behind it, the supportive
response, per-token explanations for both models, and the full JSON trace.

## Research disclaimer

Academic prototype. Not a clinical or therapy system, and the safety screen is a
keyword guard rail, not a risk assessment. For serious mental health concerns,
seek support from qualified professionals.
