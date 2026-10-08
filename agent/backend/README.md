## Patient and doctor web portals

The React app now has separate `/patient` and `/doctor` pages, each requiring
login. See [frontend setup and account provisioning](../frontend/README.md).
Public registration creates patient accounts only. Provision a trusted doctor
with `python manage_accounts.py doctor --email doctor@example.com --name "Dr. Perera"`.
Patients explicitly link a doctor by email before that doctor can view activity.

The HTTP API now requires its login cookie and, for writes, the
`X-SentiVera-Client: web` header. Doctor patient-specific reads and sensor actions
also require `X-Patient-ID` identifying a linked patient. Client `session_id`
values are accepted for compatibility but never determine ownership. Patient
chat responses contain conversational content; traces and XAI are exposed only
to authorized doctors. Existing anonymous histories stay unassigned.

Accounts live in `accounts.sqlite3` (override with `AGENT_ACCOUNTS_DB`). For HTTPS,
set `AGENT_COOKIE_SECURE=1`. The current in-memory sensor/XAI runtime requires a
single API worker and supports one wearable assigned to one patient at a time.
The Streamlit and CLI demos remain separate local research tools; these web
login rules apply to the FastAPI application.

---

# Emotional support inference pipeline

This project joins the local emotional-support artifacts in two modes:

```text
user text
  -> RoBERTa current emotion (5 labels)
  -> next-intensity forecaster (low/high, sparse + frozen-RoBERTa hybrid)
  -> named as a state (family from the current emotion)
  -> deviation from the previous turn
  -> Qwen base or Qwen + ESConv LoRA
  -> reply (+ support contacts on an extremely negative turn)
```

```text
voice recording
  -> C2 AudEERING ADV + fuzzy appraisal/stress evidence
  -> Faster-Whisper transcript
  -> current emotion + next intensity (C3 is not called)
  -> C1/C2/emotion-conditioned Qwen base or ESConv adapter
  -> Kokoro WAV reply
```

No Hugging Face download is required. All paths resolve to `../model`.

This folder also holds the C1 sensor service (`c1/`), C2 voice runtime (`c2/`), and the C3 stress/CBT
ensembles (`c3/`) — everything that reads or writes to those model artifacts
lives here, so `../model` stays weights-and-configs only. `supportive_app.py`
wires C1 + C3 + this pipeline + LangGraph into the one end-to-end text-mode
demo; `app.py`/`cli.py` are the lighter emotion-chain-only demos.
`api.py` is the same wiring exposed as a FastAPI HTTP API for the `../frontend`
React app (`streamlit`/`supportive_app.py` and `api.py` are alternative
front doors to the same pipeline — run whichever one you need).

## Running it somewhere other than a 12 GB machine

With `profile: null`, `config.yaml` describes the single-machine deployment:
every model in this process, Qwen on the GPU in bf16 (~9.5 GB of VRAM -- it is
the only model on the GPU; everything else is already CPU). To run on a 4 GB
laptop, or to split the app across two machines on the LAN, name one of
[`profiles/`](profiles/) in that key and start the [`../node`](../node) server
on the second machine:

```yaml
profile: split-main     # or single-4gb / single-highend / null
```

A profile overrides only the deployment keys -- devices, quantization, node
URLs -- so the tuned values stay in `config.yaml` and are shared by all of
them. `AGENT_PROFILE` overrides the key for one run. See
[`../DEPLOYMENT.md`](../DEPLOYMENT.md) for the memory analysis, the measured
4-bit figures and the step-by-step setup.

`GET /api/deployment` reports where each model is running and whether the
support node is answering.

## Configuration

[`config.yaml`](config.yaml) is the one file to edit to point any supportive
module at a different model artifact or generation setting — no code changes
needed. It covers: the C1 sensor's artifact directory and default port, the
current-emotion and next-intensity artifact directories, and the Qwen
base/adapter directories plus
generation params (`max_new_tokens`/`temperature`/`top_p`), and the human
support contacts offered on a crisis or extremely negative turn. All four entry
points (`app.py`, `cli.py`, `supportive_app.py`, `api.py`) and the C1/C3
modules read it via [`settings.py`](settings.py), so a change there takes
effect everywhere without touching any of them.

The C3 stress/CBT ensembles' own per-member checkpoints/weights/thresholds
stay in their existing validated configs (`c3/stress/config.py`,
`c3/CBT/binary_config.json`) — `config.yaml` only centralizes their artifact
*directory*, not the ensemble tuning inside it.

[`c3/stress/bertopic/`](c3/stress/bertopic/) is a separate, offline topic-
discovery pipeline for the Stress header (BERTopic) — not part of the live
app or the classifier; see its own README for details and its unfinished
state (no dataset vendored here yet).

## C3 Stress + CBT fusion

The user-selected routing policy feeds the C3 Stress and CBT probabilities
through [`emotion_chain/fusion.py`](emotion_chain/fusion.py). It computes a
weighted decision-level (late) fusion signal, and `text_signal_flagged` is what
reply routing and the Qwen prompt see. C1's sensor stress is a different
modality (physiological, not text) and stays independent. The fusion weights
and threshold are configured in `config.yaml`.

C3 fusion is text-mode only. In Voice mode the transcript goes directly to
the safety and emotion nodes. C2 supplies the separate
voice-appraisal/stress signal used for routing and reply context.

## Emotion states and the intensity forecast

The current classifier predicts `neutral`, `anger`, `fear`, `joy`, or
`sadness`. The forecaster answers a different question from the one the
retired 13-state model answered: **will the next turn be more or less intense**
(`low`/`high`), given this turn's text plus the current-emotion model's label,
confidence and probability vector.

It is the locked hybrid selected in `../model/forcast`: a TF-IDF +
emotion logistic model and a three-seed MLP ensemble over frozen mean-pooled
embeddings from the *same* current-emotion RoBERTa, blended 50/50 and cut at a
validation-tuned threshold of `0.595`. Only one RoBERTa is loaded — the
forecaster embeds with the classifier's encoder, which is the encoder its
features were fitted on. Blend weights and threshold live in the artifact's own
`metadata.json` and are never retuned at load time.

The low/high call is written back into the same 13-label state vocabulary the
rest of the app reads: the **family carries over from the current emotion**, so
`sadness` + `high` names `high_sadness`, and the result is always one of the
states `STATE_TRANSITIONS` already allowed. Two consequences are worth stating
plainly:

- **`neutral` turns keep `neutral`.** There is no `low_neutral`/`high_neutral`
  in the label space, and an intensity alone cannot say which family a neutral
  turn moves into. The intensity is still reported in its own field
  (`next_intensity`); only the state name collapses.
- **The forecast no longer changes a route.** It cannot cross emotion
  families, so it only ever names a negative state when the current emotion is
  already negative — which routes to the supportive strategy on its own. What
  the forecast still decides is *escalation* (below), where a `high` call is
  one of the agreeing signals. Previously a neutral turn whose forecast crossed
  into a negative family could reach the supportive route on the forecast
  alone; it no longer can.

The deployment manifest is `../model/forcast/metadata.json`; it points to the
baseline, scaler cache, and fusion checkpoints under `model_train/artifacts/`.

## Emotional deviation

`emotion_chain/deviation_tracker.py` scores how far this turn's emotion moved
from the previous one, using the same levels and scores as C4's tracker in
`R26-IT-036/ai_components/c4_emotion_support/c4_pipeline/deviation_tracker.py`
so the two components can never disagree about the same pair of turns:

```text
no previous turn, or the same emotion  -> None      0.00
a move into or out of neutral          -> Low       0.25
negative to a different negative       -> Moderate  0.60
positive <-> negative reversal         -> High      0.90
```

The graph is stateless between invocations, so the caller carries the previous
turn's emotion: `api.py` keeps it per process, the Streamlit apps keep it in
`st.session_state`, and `cli.py` takes `--previous-emotion`. A crisis turn
short-circuits before the classifier runs and therefore leaves the baseline
untouched rather than clearing it.

The level reaches the Qwen prompt as context and feeds the escalation rule
below. It does not change the route on its own: every deviation that would
switch the route is already a negative current emotion, which routes to the
adapter regardless.

## Escalation contacts

A turn read as extremely negative gets a human contact appended to the reply,
listed in `support_contacts` in `config.yaml` (default: සුමිතුරෝ (Sumithuro),
`077766889`). `emotion_chain/support_contacts.py` holds the rule: a crisis
safety status always escalates, and otherwise the user must already be in a
negative emotion **and at least two** of these must agree:

- the forecast is that emotion's high-intensity state (`high_sadness`,
  `high_anger`, `high_fear`),
- the deviation from the previous turn is `High` (a positive-to-negative
  reversal in one turn),
- C1 reports sensor stress,
- the fused C3 Stress+CBT text signal is flagged,
- C2 reports voice stress.

Two, not one, and the forecast clause is why. The forecaster does genuinely
predict intensity rather than restating a prior — that is the point of
replacing the 13-state model, whose own metadata recorded that text did not
separate its reachable next states — but it predicts it *weakly*: ROC-AUC
`0.58` on the held-out split, recalling 9% of high-intensity turns, against AI
pseudo-labels rather than human gold ones. That earns one agreeing signal and
no more. *"I feel overwhelmed today"* classifies as sadness and can forecast
`high_sadness`; on its own that would hand an ordinary hard day a phone number,
which is both wrong and the fastest way to train someone to ignore the one that
matters. A C3 stress flag alone is just as common in ordinary venting.
`escalation_signals()` returns the reasons by name so a decision can be
explained rather than guessed at.

In voice mode the reply is synthesised through the contact's `spoken_name`
(`Sumithuro`), because Kokoro is an English voice and drops Sinhala script —
the on-screen text keeps the original name, only the audio is transliterated.

The number is data, never generation: both system prompts forbid Qwen from
writing a phone number or naming a helpline, and the sentence is appended to
the finished reply. The API returns the contacts as `support_contacts`, and
both chat UIs repeat them as a panel under the bubble.

This is a routing aid built from classifier output, not a validated risk
assessment.

## Reply routing

```text
neutral/joy with no stress or CBT signal -> base Qwen
anger/fear/sadness family               -> ESConv adapter
C1 stress, C3 text stress, or CBT flag  -> ESConv adapter
crisis safety status                    -> fixed safety response + contacts
extremely negative turn (see above)     -> routed reply + contacts appended
```

The Qwen base is loaded only once. The ESConv adapter is attached once and is
enabled/disabled for each request; a second base-model copy is not loaded.

### Routing stability and ablation

- C1 uses a configurable 2-of-3 majority vote over its latest predictions;
  the displayed probability is the mean across the same windows.
- A negative forecast is counted as a routing *reason* only when its
  confidence reaches `routing.next_negative_min_confidence` in `config.yaml`.
  The floor still filters — the decision threshold is `0.595`, so a `low` call
  made just under it reports a confidence below `0.5` and is dropped — but it
  can no longer change a route, for the reason given under *Emotion states*.
- Stress/CBT fusion supports `weighted_average`, `or`, `stress_only`, and
  `cbt_only`. The selected live policy is `decision_fusion.method`.

To compare fusion policies on a labelled validation split (never the held-out
test set), prepare `stress_probability,cbt_probability,label` columns and run:

```bash
python evaluate_fusion.py validation_router_labels.csv
```

The report includes all ablations and searches weighted-fusion weights and
thresholds for the best validation macro-F1.

## On-demand XAI

The doctor portal exposes an **Explain predictions (XAI)** action for retained
patient turns whose explanation context is still available. It calls `POST /api/xai` only when requested and displays Integrated
Gradients for the latest C1 Feature-MLP prediction, the DeBERTa-v3 members of
the C3 Stress and CBT headers, the current-emotion model and the
next-intensity forecaster. Positive scores support the displayed target and
negative scores oppose it. The forecaster panel attributes its frozen-RoBERTa
branch only — the sparse half of the blend is linear in its own n-gram
features and is not differentiable in the tokens, so the panel says which half
it is showing. Qwen is not included because a free-form generated reply has no
single classifier target.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run supportive_app.py   # full flow: mode select, C1 sensor gate, C3 heads, reply
streamlit run app.py              # lighter demo: current/next emotion + reply only
python cli.py "I feel overwhelmed today"
python cli.py --reply "I feel overwhelmed today"

# FastAPI backend for ../frontend
uvicorn api:app --reload --port 8005 --reload-exclude ".venv/*"
```

Voice endpoints are `POST /api/voice/chat` (multipart field `audio`) and
`GET /api/voice/audio/{turn_id}`. Browser recording supports formats decoded
by PyAV/Faster-Whisper, including WebM/Opus. Kokoro English phonemization works
best with `espeak-ng` installed on the host.

The `--reply` path loads the 4B Qwen model and needs substantial RAM/VRAM.

`--reload-exclude ".venv/*"` matters: uvicorn's `--reload` file-watcher
otherwise recurses into `.venv` too, which is slow with a fully installed
environment and can make the watcher restart the worker mid-request (seen as
`ECONNRESET` from the Vite proxy on a slower endpoint like
`/api/sensor/upload`). Drop `--reload` entirely if it still misbehaves.

## Tests

```bash
python -m unittest discover -s tests -v
```

`tests/test_intensity_forecaster.py` loads the real exported hybrid and
replays the validation split through the live inference path, because the
backend recomputes the emotion probabilities and the frozen embeddings at
request time rather than reading the training cache — a wrong feature order or
a skipped standardization step would otherwise still produce plausible
probabilities. `tests/test_models.py` still covers the retired 13-state
checkpoints, which remain on disk. The full Qwen weights are intentionally not
loaded by unit tests.

## Research limitation

Every next-intensity target — train, validation and test — is an AI
pseudo-label, not an adjudicated human one, so the reported numbers measure
agreement with that labelling process. On the untouched test split the locked
hybrid scores macro-F1 `0.50`, balanced accuracy `0.51`, ROC-AUC `0.58` and
recalls `9%` of the high-intensity turns, against a `0.46` macro-F1 majority
baseline. The pilot has 2,000 training rows and 32 high examples in test, so
the class-specific intervals are wide.

Treat the forecast as a reply-conditioning signal and as one weak agreeing
vote in the escalation rule — never as a clinical assessment or a measured
intensity. The full report is in `../model/forcast/Evaluation/results/`.
