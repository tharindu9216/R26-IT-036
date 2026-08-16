# Running the C4 demo on a second machine

Written for a laptop with an **RTX 3050 4 GB**, but the steps are the same for
any second machine; only the low-VRAM flag is card-specific.

Read this alongside the VRAM measurements in `README.md` → *Running on a small
GPU*. The short version: the default settings need ~3.5–3.7 GB and will not fit
4 GB reliably, so you set `C4_LOW_VRAM=1` and it drops to ~3.0–3.2 GB.

---

## 1. What to copy — one folder

Everything the demo needs is inside `c4_emotion_support/models/`. It is
gitignored, so it does not travel with the repository; copy it by hand (USB,
network share, cloud drive).

```text
models/                              8,819 MB total
├── qwen3-4b-instruct/               7,687 MB   reply generation base model
├── current_emotion_classifier/        479 MB   RoBERTa, 5 labels
├── next_emotion_forecaster/           259 MB   TextCNN / BiLSTM / DistilBERT checkpoints
├── distilbert-base-uncased/           256 MB   base for the DistilBERT forecaster
└── esconv_reply_adapter/              137 MB   the trained ESConv LoRA
```

With that folder in place the laptop needs **no Hugging Face download and no
network at all**. This was verified by running the full test suite with the
Hugging Face cache pointed at a non-existent directory and
`HF_HUB_OFFLINE=1` — all checks passed.

### Trimming it

| Drop | Saves | Cost |
|---|---|---|
| `distilbert-base-uncased/` | 256 MB | the DistilBERT forecaster option fails to load; TextCNN (the default) and BiLSTM still work |
| `next_emotion_forecaster/distilbert_forecast.pt` | 253 MB | same as above |
| `esconv_reply_adapter/` | 137 MB | replies come from the un-fine-tuned base model |

Dropping both DistilBERT pieces takes the transfer from 8.8 GB to **8.3 GB**.
The sidebar still lists DistilBERT and reports the load failure honestly if you
select it, so nothing breaks silently.

`qwen3-4b-instruct/` is not optional — without it there is no reply generation
at all, only templates.

### Regenerating the folder instead of copying

If the other machine has good internet, skip the 8.8 GB transfer. Copy only the
three small folders (`current_emotion_classifier`, `next_emotion_forecaster`,
`esconv_reply_adapter` — 875 MB), then run there:

```powershell
python vendor_models.py
```

That downloads Qwen3-4B and DistilBERT from Hugging Face and lays them out
locally. `--list` reports what is already present; `--skip-distilbert` omits the
optional one.

### Why not just copy the Hugging Face cache

Because it is twice the size and not portable. Windows cannot use symlinks in
the cache, so every file is stored twice — once under `blobs/` and again under
`snapshots/`. The Qwen3-4B cache entry is **15.3 GB on disk for 7.7 GB of
weights**, in hash-named directories that assume the machine that built them.
`vendor_models.py` copies the resolved snapshot into a plain directory that
`from_pretrained()` reads directly.

### How the paths are found

`config.py` resolves each model in order and falls back cleanly, so a machine
that has vendored nothing still works if it has network:

| Model | Order |
|---|---|
| Base model | `$env:C4_BASE_MODEL` → `models/qwen3-4b-instruct/` → hub `Qwen/Qwen3-4B-Instruct-2507` |
| Adapter | `$env:C4_ADAPTER_PATH` → `models/esconv_reply_adapter/` → `../../../qwen3_esconv_finetune/outputs/.../final_adapter/` |
| DistilBERT | `models/distilbert-base-uncased/` → hub `distilbert-base-uncased` |

Nothing needs configuring on the second machine — copying `models/` is enough.

---

## 2. Install

Python 3.12 and an NVIDIA driver recent enough for CUDA 12.x.

```powershell
cd R26-IT-036\ai_components\c4_emotion_support
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a CUDA build of PyTorch **first** — the plain `pip install torch` gives
a CPU-only wheel on Windows, and everything then silently falls back to
templates:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Confirm the GPU is visible before going further:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that prints `False`, fix it now. Nothing below will work as intended.

---

## 3. Configure for 4 GB

```powershell
$env:C4_LOW_VRAM = "1"
```

To make it permanent for your user account:

```powershell
setx C4_LOW_VRAM 1
```

(`setx` applies to new terminals, not the current one.)

---

## 4. Verify before demoing

```powershell
python smoke_test.py --llm
```

The smoke test honours `C4_LOW_VRAM`, so it exercises the configuration the app
will actually use. It should print `LOW-VRAM MODE: classifier and forecaster
pinned to cpu.` at the top and `All checks passed.` at the end.

If you copied `models/` there is no download — it starts immediately. To prove
the machine is genuinely self-contained before you travel with it, run it with
Hugging Face blocked:

```powershell
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
python smoke_test.py --llm --all
```

If that passes, nothing is silently reaching for the network. (This exact check
was run on the source machine, with the cache pointed at a non-existent
directory, and passed.) Clear the two variables afterwards, or leave them set —
the demo never needs the network once `models/` is populated.

Watch VRAM in a second terminal while it runs:

```powershell
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 2
```

Expect a peak in the region of 3.0–3.2 GB.

---

## 5. Run

```powershell
streamlit run streamlit_app.py
```

The sidebar states which configuration is live: classifier device, forecaster
checkpoint, whether the adapter loaded, and a low-VRAM notice. If a demo is
running on templates or without the adapter, the sidebar says so — check it
before presenting.

---

## Troubleshooting

**`CUDA out of memory` on load.** Something else is on the GPU. Close browsers
with hardware acceleration and any other Python process, then retry. On a
laptop, check in Windows Settings → Display → Graphics whether the display runs
on the Intel iGPU (leaves nearly all 4 GB free) or the dGPU (does not).

**Out of memory only after several messages.** The KV cache grows with
conversation length. Use **Clear conversation** in the sidebar, or lower
`LLM_MAX_CONTEXT_TURNS` in `config.py`.

**Sidebar says "base Qwen3-4B only — no trained adapter found".** The adapter
did not land where `config.py` looks. Check that
`models/esconv_reply_adapter/adapter_config.json` exists, or point at it
directly with `$env:C4_ADAPTER_PATH`.

**It starts downloading Qwen3-4B when you expected it not to.** The vendored
copy was not found, so it fell through to the hub. Confirm what resolved:

```powershell
python -c "from config import LLM_BASE_MODEL; print(LLM_BASE_MODEL)"
```

A local path means the vendored copy is in use; `Qwen/Qwen3-4B-Instruct-2507`
means it is about to download. The usual cause is `models/qwen3-4b-instruct/`
missing `config.json` — an interrupted copy. Re-copy, or run
`python vendor_models.py --force`.

**`vendor_models.py` says a model is NOT in the local cache.** It only copies
from an existing Hugging Face cache; it will not download on a machine that
never had the model. Run the demo once so the download happens, then vendor.

**Sidebar says "Qwen3-4B did not load".** The caption under it carries the real
exception. The usual causes are a CPU-only PyTorch wheel (see step 2) and a
bitsandbytes that will not import.

**Replies take tens of seconds.** Check the sidebar for "Running on the CPU".
That means `C4_ALLOW_CPU` is set and CUDA was not found — go back to step 2.

---

## Training on the laptop

Don't. QLoRA fine-tuning of Qwen3-4B peaked at **5.46 GB** on the tuned RTX 3080
profile, which does not fit 4 GB. Train on the 3080 (~1 h 50 m at `-Epochs 1`,
see `qwen3_esconv_finetune/README.md`), then re-vendor and copy across:

```powershell
Copy-Item ..\..\..\qwen3_esconv_finetune\outputs\qwen3-4b-esconv-qlora\final_adapter `
          models\esconv_reply_adapter -Recurse -Force
```

Only the 137 MB adapter folder changes — the 7.7 GB base model does not, so
retraining costs a small copy rather than a full re-transfer. That is exactly
why the adapter is kept separate from the base model instead of merged into it
with `merge_adapter.py`.
