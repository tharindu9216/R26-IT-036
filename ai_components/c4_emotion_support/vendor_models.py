#!/usr/bin/env python3
"""Copy the Hugging Face models this demo needs into `models/`, for migration.

Why not just copy the Hugging Face cache itself: on Windows the cache cannot use
symlinks, so every file is stored twice -- once under `blobs/` and again under
`snapshots/`. The Qwen3-4B cache entry is 15.3 GB on disk for 7.7 GB of actual
weights. Copying the resolved snapshot into a plain directory halves the
transfer and drops the cache's hash-named layout, which is not portable between
machines anyway.

The result is a directory `from_pretrained()` accepts directly, so a machine
with `models/` populated needs no Hugging Face download and no network at all.

    python vendor_models.py                  # Qwen3-4B + DistilBERT
    python vendor_models.py --skip-distilbert
    python vendor_models.py --voice          # + Whisper and Kokoro for voice mode
    python vendor_models.py --list           # show what is vendored already

Run it from this directory, after the models are in your Hugging Face cache
(they are, if the demo has run once).
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from config import (
    DISTILBERT_BASE,
    KOKORO_DIR,
    KOKORO_VOICES_FILE,
    LLM_BASE_MODEL,
    MODELS_DIR,
    STT_MODEL,
)

# (hub repo id, destination folder under models/, why it is needed)
TARGETS = [
    (LLM_BASE_MODEL, "qwen3-4b-instruct", "reply generation (required)"),
    (DISTILBERT_BASE, "distilbert-base-uncased", "distilbert forecaster (optional)"),
]

# Voice mode's two models, neither of which lives where the others do.
#
# Whisper is a normal Hugging Face repo, so it goes through the same snapshot
# copy as everything else -- but into models/faster-whisper/<size>, which is the
# download_root c4_pipeline/voice.py hands to WhisperModel.
#
# Kokoro is not on the hub at all; its ONNX builds are GitHub release assets and
# are fetched by URL.
#
# fp16 rather than the smaller int8: int8 measured ~4x SLOWER on the target
# laptop CPU (RTF 1.45 against 0.34), because ONNX int8 spends more in
# quantise/dequantise than it saves. See KOKORO_MODEL_PREFERENCE in config.py
# for the full measurement.
KOKORO_RELEASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
KOKORO_FILES = [
    ("kokoro-v1.0.fp16.onnx", "Kokoro-82M, fp16 (~160 MB)"),
    (KOKORO_VOICES_FILE, "Kokoro voice pack (~26 MB)"),
]


def _whisper_repo(model_size: str) -> str:
    """faster-whisper checkpoints are published under the Systran namespace."""
    return f"Systran/faster-whisper-{model_size}"


# Weights, tokenizer and config only. The repo's README, LICENSE and
# .gitattributes are not needed to load a model.
SKIP_SUFFIXES = {".md", ".gitattributes"}
SKIP_NAMES = {"LICENSE", "NOTICE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--skip-distilbert", action="store_true",
                        help="skip the 130 MB forecaster base model")
    parser.add_argument("--voice", action="store_true",
                        help="also vendor the Whisper and Kokoro voice models")
    parser.add_argument("--list", action="store_true",
                        help="report what is already vendored, copy nothing")
    parser.add_argument("--force", action="store_true",
                        help="re-copy even if the destination already exists")
    return parser.parse_args()


def directory_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 ** 2)


def resolve_snapshot(repo_id: str) -> Path | None:
    """Locate an already-downloaded snapshot without hitting the network."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:
        return None


def vendor(repo_id: str, folder: str, note: str, force: bool) -> bool:
    destination = MODELS_DIR / folder

    if destination.exists() and not force:
        print(f"  {folder:<26} already present ({directory_size_mb(destination):,.0f} MB) - skipping")
        return True

    source = resolve_snapshot(repo_id)
    if source is None:
        print(f"  {folder:<26} NOT in the local cache.")
        print(f"  {'':<26} Run the demo once so {repo_id} downloads, then retry.")
        return False

    print(f"  {folder:<26} copying from cache …", flush=True)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    for item in sorted(source.iterdir()):
        if not item.is_file():
            continue
        if item.suffix in SKIP_SUFFIXES or item.name in SKIP_NAMES:
            continue
        shutil.copy2(item, destination / item.name)

    print(f"  {folder:<26} done ({directory_size_mb(destination):,.0f} MB) - {note}")
    return True


def download(url: str, destination: Path, note: str, force: bool) -> bool:
    """Fetch one release asset, unless it is already on disk."""
    if destination.exists() and not force:
        size = destination.stat().st_size / (1024 ** 2)
        print(f"  {destination.name:<26} already present ({size:,.0f} MB) - skipping")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {destination.name:<26} downloading ...", flush=True)
    # Download to a .part file first, so an interrupted transfer cannot leave a
    # truncated model that later fails to load with a confusing ONNX error.
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(destination)
    except Exception as error:  # noqa: BLE001
        partial.unlink(missing_ok=True)
        print(f"  {destination.name:<26} FAILED - {type(error).__name__}: {error}")
        return False

    size = destination.stat().st_size / (1024 ** 2)
    print(f"  {destination.name:<26} done ({size:,.0f} MB) - {note}")
    return True


def fetch_whisper(force: bool) -> bool:
    """Put the Whisper checkpoint where voice.py expects it.

    Unlike the other targets this one may download. The rest of this script only
    copies what is already in the Hugging Face cache, but the Whisper checkpoint
    is small, and a `--voice` run that ends by telling you to go and run the demo
    first would not have prepared the offline machine it exists to prepare.
    """
    destination = MODELS_DIR / "faster-whisper" / STT_MODEL
    if destination.exists() and not force:
        size = directory_size_mb(destination)
        print(f"  {'faster-whisper/' + STT_MODEL:<26} already present ({size:,.0f} MB) - skipping")
        return True

    print(f"  {'faster-whisper/' + STT_MODEL:<26} downloading ...", flush=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            _whisper_repo(STT_MODEL),
            local_dir=str(destination),
            allow_patterns=["*.bin", "*.json", "*.txt", "*.model"],
        )
    except Exception as error:  # noqa: BLE001
        print(f"  {'faster-whisper/' + STT_MODEL:<26} FAILED - {type(error).__name__}: {error}")
        return False

    print(f"  {'faster-whisper/' + STT_MODEL:<26} done ({directory_size_mb(destination):,.0f} MB)")
    return True


def vendor_voice(force: bool) -> bool:
    """Whisper from the hub, Kokoro from its GitHub release."""
    print()
    print("Voice models:")
    whisper_ok = fetch_whisper(force)
    kokoro_ok = all(
        download(f"{KOKORO_RELEASE}/{name}", KOKORO_DIR / name, note, force)
        for name, note in KOKORO_FILES
    )
    if not kokoro_ok:
        # Not fatal: voice.py falls back to Windows SAPI, which needs no download.
        print("  Kokoro unavailable - voice mode will use the Windows SAPI fallback.")
    return whisper_ok


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    targets = TARGETS[:1] if args.skip_distilbert else TARGETS

    if args.list:
        print(f"Vendored models in {MODELS_DIR}:\n")
        total = 0.0
        for path in sorted(MODELS_DIR.iterdir()):
            if path.is_dir():
                size = directory_size_mb(path)
                total += size
                print(f"  {path.name:<30} {size:>9,.0f} MB")
        print(f"\n  {'TOTAL':<30} {total:>9,.0f} MB")
        return

    print(f"Vendoring into {MODELS_DIR}\n")
    ok = all(vendor(repo, folder, note, args.force) for repo, folder, note in targets)

    if args.voice:
        ok = vendor_voice(args.force) and ok

    total = sum(directory_size_mb(p) for p in MODELS_DIR.iterdir() if p.is_dir())
    print(f"\nmodels/ is now {total:,.0f} MB.")
    if ok:
        print("Copy the whole models/ folder to the other machine - it needs no network.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
