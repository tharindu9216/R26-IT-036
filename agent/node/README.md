# Support node

A small model server that runs some of the app's heavy models on a **second
machine** and answers the backend over the local network. It exists so the
whole project can run on a 4 GB laptop: the 4B reply model moves off the
machine that runs the UI, C1, C3 and the emotion chain.

It is optional. With `remote_services` left null in `backend/config.yaml`,
nothing here is used and the app runs exactly as it always has.

## What it can host

| Service | Model | Why it can move |
| --- | --- | --- |
| `reply` | Qwen3-4B + ESConv LoRA | The only model that really needs a GPU — and the whole reason for the split |
| `stt` | Faster-Whisper | Library-only, no project code |
| `tts` | Kokoro | Library-only, no project code |
| `c2` | AudEERING ADV + fuzzy appraisal | Needs `backend/c2/appraisal_stress` as well (see below) |

**C1, C3 Stress/CBT and the current/next-emotion chain cannot move.** C1 reads
a serial port on the machine the wearable is plugged into, and the other three
are built on this repository's own model code rather than on a library, so
serving them would mean duplicating that code and keeping two copies in step.
They are also all CPU models in the default config, so they are not what is
filling the GPU.

## What this node is not

It runs models. It holds no conversation history and makes no decisions: the
backend detects crisis wording, chooses the route, builds the prompt, decides
whether the LoRA is active, and appends escalation contacts — then asks this
node only for the completion. That is deliberate. It means a two-machine
deployment produces the same reply, by the same route, as a one-machine one,
and there is no second place where the safety rules could drift.

## Setup on the second laptop

Copy two folders from the project, keeping them side by side:

```text
somewhere/
  node/     <- this folder
  model/    <- only the subfolders for the services you enable:
              qwen3-4b-instruct/  esconv_reply_adapter/   (reply,  7.8 GB)
              fasterwhisper/                              (stt,    464 MB)
              kokoro/                                     (tts,    347 MB)
              c2/                                         (c2,     631 MB)
```

Do **not** copy `backend/.venv` — it is a stale Linux virtualenv and is useless
on Windows.

Then:

```powershell
cd node
.\run.ps1                 # first run: creates .venv, installs requirements
.\run.ps1 -SkipInstall    # after that
```

`run.ps1` prints this machine's LAN addresses. Put one of them into the main
machine's `backend/profiles/split-main.yaml`.

On Linux/macOS use `./run.sh` instead.

If pip installs a CPU-only torch, install the CUDA build first and rerun:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Configuration

Everything is in [`config.yaml`](config.yaml). The parts that matter:

- `services` — which models this node hosts. Anything set to `false` is never
  loaded and returns **503** if asked for, so a node can host just the reply
  model and leave voice on the main machine.
- `reply.quantization` — `nf4` fits the 4B model into ~3.2 GB of VRAM, which is
  what makes a 4 GB card work. Use `none` on a 10 GB+ card for the exported
  bf16 weights.
- `preload` — `false` (default) loads each model on its first request; `true`
  loads them during startup so the first user turn is not slow.
- `auth_token` — optional shared secret, matched against `X-Auth-Token`. Set it
  on a shared network and put the same value in the backend's
  `remote_services.auth_token`.

Point at a different file with `$env:NODE_CONFIG = "other.yaml"`.

### Hosting C2

The C2 appraisal engine is a joblib artifact pickled against the backend's
`c2.appraisal_stress` package, so it can only load if that package is
importable. To host C2 here, also copy the `backend/` folder (its `.py` files
are about 1 MB — all the weights live in `model/`), set `services.c2: true`,
and leave `c2.appraisal_package_dir: ../backend`.

Otherwise leave `services.c2: false` and let C2 run on the main machine. It is
a CPU model there either way, so this costs little.

## Checking it works

From the main machine's browser or terminal:

```powershell
curl http://<node-ip>:8010/health
```

```json
{"ok": true, "hosts": {"reply": "auto / nf4"}, "loaded": {"reply": true}}
```

`loaded` shows which models are actually in memory — useful for telling a slow
first request apart from a broken one.

The backend has a matching check that probes every node it is configured to
use: `GET /api/deployment` on port 8005.

## API

All endpoints require `X-Auth-Token` when `auth_token` is set.

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/health` | — | hosted services and load state |
| POST | `/v1/reply/generate` | JSON: `messages`, `max_new_tokens`, `temperature`, `top_p`, `use_adapter` | `{"text": ...}` |
| POST | `/v1/stt/transcribe` | raw float32 LE mono PCM @ 16 kHz | `{"text", "language", "language_probability", "duration_seconds"}` |
| POST | `/v1/tts/synthesize` | JSON: `text`, optional `voice`, `speed` | `audio/wav` bytes |
| POST | `/v1/c2/appraise` | raw float32 LE mono PCM @ 16 kHz, `?file_name=` | the full appraisal dict |

Audio crosses as raw float32 because the backend has already decoded the
browser recording to exactly that form for C2 — no re-encode, and no ffmpeg
needed on this machine.

## Files

- [`server.py`](server.py) — FastAPI app, config loading, auth, endpoints
- [`models.py`](models.py) — the four model runners; imports nothing from `../backend`
- [`config.yaml`](config.yaml) — everything you normally edit
- [`requirements.txt`](requirements.txt) — a strict subset of the backend's
