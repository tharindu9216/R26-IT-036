# Deployment guide

How to run this project on one high-end machine, on one 4 GB laptop, or split
across two laptops on the same network.

The application code is the same in all three cases. What changes is one line
naming a profile in [`backend/profiles/`](backend/profiles/), which overrides
only the deployment-specific keys of `backend/config.yaml`. Fusion weights,
thresholds, routing confidence and the support contacts live in `config.yaml`
alone and are inherited by every profile, so they cannot drift between machines.

## Selecting a profile

Set it once in `backend/config.yaml`, near the top:

```yaml
profile: split-main     # or single-4gb / single-highend / null
```

That is the normal way — it belongs to the machine, so it survives reboots and
new terminals, and copying the folder to another laptop means editing one line.

For a one-off run, `AGENT_PROFILE` overrides that line without editing anything:

```powershell
$env:AGENT_PROFILE = "single-4gb"; uvicorn api:app --port 8005
```

Either place accepts `none` to force plain `config.yaml`, which is useful for
checking whether a problem is caused by the profile. A profile file cannot
itself set `profile:` — one hop only, so the active deployment is always
readable from those two places. `GET /api/deployment` reports which one won:

```json
{"services": {"profile": "split-main (from config.yaml)", ...}}
```

---

## 1. Where the memory actually goes

Measured on the development machine (RTX 3080 12 GB, 64 GB RAM) with the
default `config.yaml`:

| Component | Weights on disk | Runs on | Notes |
| --- | --- | --- | --- |
| **Qwen3-4B-Instruct + ESConv LoRA** | 7.69 GB + 137 MB | **GPU** | bf16. **This is the entire 9.5 GB of VRAM in use** |
| C3 CBT ensemble (BERT + MentalBERT + DeBERTa-v3) | 1.54 GB | CPU | 3 encoders, fp32 |
| C3 Stress ensemble (BERT + DeBERTa-v3) | 1.12 GB | CPU | 2 encoders, fp32 |
| C2 AudEERING ADV (wav2vec2-large) | 631 MB | CPU | voice mode only |
| Current-emotion RoBERTa | 479 MB | CPU | |
| Faster-Whisper (CTranslate2) | 464 MB | CPU | int8 |
| Kokoro TTS | 347 MB | CPU | voice mode only |
| Next-emotion forecaster | 269 MB | CPU | the deployed BiGRU is only 1.4 MB of that; the rest is the three unused variants and the archive |
| C1 sensor MLP + calibrators | 45 KB | CPU | |
| | **12.4 GB total** | | |

**The finding that drives everything below:** `config.yaml` already places
every model except Qwen on the CPU. The 9.5 GB of VRAM is one model. Nothing
else on the GPU needs to be moved, shrunk, or offloaded — and moving the small
CPU models to a second machine would save no VRAM at all.

So there is exactly one problem to solve: **Qwen3-4B in bf16 needs ~8 GB of
weights and neither the RTX 3050 nor the RTX 2050 has 4 GB to spare for it.**

### Two ways to solve it, both supported

| | 4-bit (NF4) on one machine | Offload to a support node |
| --- | --- | --- |
| VRAM for the reply model | ~2.9 GB idle, 3.4–3.8 GB peak | 0 on the main machine |
| Second machine needed | no | yes |
| Reply quality | slightly below bf16 | unchanged if the node runs bf16; same as 4-bit if it doesn't |
| Main machine's GPU free for C3 + emotion | no | yes (~2.8 GB of encoders move to CUDA) |
| Profile | `single-4gb` | `split-main` + `node/` |

Measured on the 3080 with `quantization: nf4`, reading PyTorch's own allocator
on the node (`GET /health` reports it, and `POST /v1/admin/reset-peak` clears
the high-water mark between measurements). Peak is reserved memory during
generation with long turns:

| Prompt | Peak reserved | + CUDA context (~0.3 GB) |
| --- | --- | --- |
| weights only, idle | 2.85 GB | ~3.1 GB |
| first turn, no history | 2.91 GB | ~3.2 GB |
| 2 replayed messages (`history_turns: 1`) | 3.08 GB | ~3.4 GB |
| **4 replayed messages (`history_turns: 2`)** | **3.40 GB** | **~3.7 GB** |
| 6 replayed messages (`history_turns: 3`) | 3.77 GB | ~4.1 GB |

`history_turns` counts *exchanges* — a user message plus its reply, so two
replayed messages each. (It was applied as a raw message count until the
per-session history store landed; a value copied from a config older than that
now replays twice as much, so halve it.)

The last row does not fit a 4 GB card, which is why the 4 GB profiles set
`reply_generation.history_turns: 2`. Every replayed message is also KV cache,
and six long ones cost ~0.9 GB on top of the weights. Four leaves roughly 300 MB
of headroom on a 4 GB GPU — enough if that GPU is not also driving a display,
which on a laptop with switchable graphics it usually is not. **If you still hit
`CUDA out of memory`, drop to 1**; the cost is that the model sees less of the
conversation, not that anything breaks.

These are worst-case numbers: each replayed message in the test was ~110 words.
Ordinary chat turns are far shorter and cost proportionally less.

---

## 2. Setup A — one high-end machine (unchanged)

Nothing to do. This is the default and still behaves exactly as before.

```powershell
cd backend
uvicorn api:app --port 8005
# separate terminal
cd frontend
npm run dev
```

Optionally set `profile: single-highend` in `config.yaml` to additionally move
the C3 and emotion models onto the GPU, which a 12 GB card has room for
alongside bf16 Qwen.

---

## 3. Setup B — one 4 GB laptop, no second machine

The fallback when the support laptop is not around. Everything runs on the one
machine; Qwen loads 4-bit and takes the GPU, everything else stays on CPU.

In `backend/config.yaml`:

```yaml
profile: single-4gb
```

```powershell
cd backend
uvicorn api:app --port 8005
```

Requires `bitsandbytes` (already in `backend/requirements.txt`). Replies are
slower than the split setup and slightly weaker than bf16; routing, safety and
escalation are identical.

---

## 4. Setup C — the two-laptop split

```text
RTX 3050 laptop  (main)                RTX 2050 laptop  (support node)
├── frontend/       React UI           └── node/     Qwen  (4-bit, ~3 GB VRAM)
├── backend/        api.py                           Faster-Whisper (CPU)
│   ├── C1 sensor   (serial port)                    Kokoro         (CPU)
│   ├── C3 Stress + CBT  (CUDA)            model/    only the folders it needs
│   └── current/next emotion (CUDA)
└── model/                                    ^
        │                                     │
        └──────── HTTP over the LAN ──────────┘
                  port 8010
```

### What moves and what cannot

**Moves:** the reply model, Faster-Whisper, Kokoro. All three are ordinary
library models — the node needs no project code for them.

**Cannot move:** C1 (reads the serial port the wearable is plugged into), C3
Stress/CBT and the current/next-emotion chain (built on this repository's own
model classes, so serving them would mean maintaining two copies of that
code). None of them are on the GPU today, so this costs nothing.

**C2 is in between.** Its fuzzy appraisal engine is a joblib artifact pickled
against `backend/c2/appraisal_stress`, so the node can only host it if that
package is there too. Copy `backend/` to the node as well (about 1 MB of `.py`
— all weights are in `model/`) and set `services.c2: true`, or leave C2 on the
main machine, where it is a CPU model anyway.

### Step 1 — the support node (RTX 2050 laptop)

Copy `node/` and `model/` side by side. From `model/` you only need:

| For | Folders | Size |
| --- | --- | --- |
| `reply` | `qwen3-4b-instruct/`, `esconv_reply_adapter/` | 7.8 GB |
| `stt` | `fasterwhisper/` | 464 MB |
| `tts` | `kokoro/` | 347 MB |
| `c2` (optional) | `c2/` + a copy of `backend/` | 631 MB |

```powershell
cd node
.\run.ps1
```

It creates a venv, installs requirements, and prints the LAN addresses to use.
See [`node/README.md`](node/README.md) for details.

### Step 2 — the main machine (RTX 3050 laptop)

Copy the whole `agent/` folder as you planned. **Delete `backend/.venv`** —
it is a stale Linux virtualenv checked in from another machine and is not
usable on Windows.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Edit [`backend/profiles/split-main.yaml`](backend/profiles/split-main.yaml) and
replace `192.168.1.42` with the node's address. Then set the profile in
`backend/config.yaml`:

```yaml
profile: split-main
```

```powershell
cd backend
uvicorn api:app --port 8005
```

```powershell
cd frontend
npm install
npm run dev
```

### Step 3 — confirm the wiring

```powershell
curl http://localhost:8005/api/deployment
```

```json
{"services": {"profile": "split-main (from config.yaml)",
              "reply": "http://192.168.1.42:8010",
              "stt": "http://192.168.1.42:8010",
              "c3": "local (cuda)"},
 "nodes": [{"url": "http://192.168.1.42:8010", "reachable": true,
            "hosts": {"reply": "auto / nf4"}, "loaded": {"reply": true}}]}
```

If `reachable` is false, check Windows Firewall on the node — it must allow
inbound TCP 8010 on the private network profile.

---

## 5. Moving a model back and forth

Each service is independent. Blank a URL in the profile and that model runs
locally again on the next restart; set one and it moves to the node. Nothing
else changes — the backend keeps making every routing, safety and escalation
decision either way, and a support node only ever returns a completion, a
transcript, a waveform or an appraisal.

Verified during development: a waveform sent to the node and the same waveform
run locally produce identical transcripts and appraisal values (all fields
agree to float32 precision), and a crisis message never reaches the node at all
because the safety template short-circuits before generation.

---

## 6. When something goes wrong

| Symptom | Cause | Fix |
| --- | --- | --- |
| Reply arrives but `source` is `template_fallback` | The node is unreachable or errored; the backend degraded rather than failing the turn | `GET /api/deployment`; check the backend log for the `Reply generation failed` warning, which names the error |
| `CUDA out of memory` on the node | history plus 4-bit weights exceeds the card | Check `quantization: nf4` is set, then lower `reply_generation.history_turns` (2 → 1). `GET /health` on the node shows the peak that failed |
| First reply takes ~40 s, later ones are fast | Models load lazily on first use | Set `preload: true` in the node's `config.yaml` |
| Node returns 503 | That service is `false` in the node's `services` | Enable it there, or blank the matching URL so it runs on the main machine |
| Node returns 401 | `auth_token` mismatch | Make the node's `auth_token` and the backend's `remote_services.auth_token` equal |
| UI in the browser cannot reach the API from another device | CORS | Add that address to `api.cors_origins`, and start Vite with `npm run dev -- --host` |

---

## 7. Two pre-existing issues worth knowing about

Neither is caused by the deployment work, and neither is fixed by it. Both were
found while testing on the development machine.

**Voice mode segfaults on this Windows machine.** The sequence
`decode_audio` (PyAV) → C2 wav2vec2 → Faster-Whisper in one process crashes
the interpreter (exit 139) inside `transcribe()`. It reproduces deterministically
and needs all three; any two of them are fine. That is exactly what
`POST /api/voice/chat` does, so voice mode is currently broken here. Note that
Setup C happens to sidestep it — with `stt` on the node, the backend runs PyAV
and C2 but never constructs a Whisper model, and the node runs Whisper but
never PyAV because it receives raw PCM. That is a side effect, not a fix; the
real cause is most likely an OpenMP/threading clash between PyAV and
CTranslate2 and deserves its own investigation.

**scikit-learn version mismatch.** The pickled C1 calibrators and the C2
appraisal engine were saved with scikit-learn 1.6.1; the installed version is
1.8.0. This raises `InconsistentVersionWarning` on every load and makes
`tests/test_c1.py` fail with `'SimpleImputer' object has no attribute
'_fill_dtype'`. Pinning `scikit-learn==1.6.1` on the deployment machines is the
quick fix; re-exporting the artifacts is the durable one. Worth settling before
copying to the laptops, so the same pin goes everywhere.
