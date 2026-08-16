"""Qwen3-4B loader and text generation for the reply-generation stage.

One model object serves both routes. The base weights are loaded once in 4-bit
NF4 and the ESConv LoRA adapter is attached on top; PEFT's `disable_adapter()`
context manager then switches the adapter off for the positive/neutral route
instead of a second model being loaded. On a 12 GB card that distinction is the
difference between fitting and not: two separate 4-bit copies would cost ~6 GB
before the classifier and forecaster are counted, one copy plus an adapter costs
~3.2 GB.

Loading is lazy and every failure is captured rather than raised. A missing
adapter, an absent GPU, or a bitsandbytes build that will not import all
degrade to `available = False`, and the graph falls back to the template
responder. The demo must still start on a machine that cannot host a 4B model.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    LLM_ADAPTER_PATH,
    LLM_ALLOW_CPU,
    LLM_BASE_MODEL,
    LLM_LOAD_IN_4BIT,
    LLM_MAX_NEW_TOKENS,
    LLM_REPETITION_PENALTY,
    LLM_TEMPERATURE,
    LLM_TOP_P,
)

# Qwen3-4B (the original hybrid checkpoint) emits <think>…</think> before its
# answer unless thinking is disabled. The -Instruct-2507 checkpoint has no
# thinking mode and rejects the flag, so it is passed only for the hybrid.
#
# Matched on the basename rather than the full id, because the model may be a
# vendored local directory (models/qwen3-4b/) rather than a hub id.
_THINKING_BASENAMES = {"qwen3-4b"}


def _is_thinking_checkpoint(model_id: str) -> bool:
    basename = model_id.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return basename.lower() in _THINKING_BASENAMES
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class GenerationResult:
    text: str
    adapter_used: bool
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


class QwenReplyGenerator:
    """Lazily-loaded Qwen3-4B with an optional ESConv LoRA adapter."""

    def __init__(
        self,
        base_model: str = LLM_BASE_MODEL,
        adapter_path: Optional[Path] = LLM_ADAPTER_PATH,
        load_in_4bit: bool = LLM_LOAD_IN_4BIT,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.load_in_4bit = load_in_4bit

        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

        self.loaded = False
        self.load_error: Optional[str] = None
        self.adapter_loaded = False
        self.adapter_error: Optional[str] = None
        self.device = "unknown"
        self.dtype = "unknown"
        # Whether 4-bit actually applied. Requested != achieved on a CPU run.
        self.quantized = False

    # ------------------------------------------------------------------ status
    @property
    def available(self) -> bool:
        return self.loaded and self._model is not None

    def adapter_is_present(self) -> bool:
        """Whether a trained adapter exists on disk (checked without loading it)."""
        if self.adapter_path is None:
            return False
        return (self.adapter_path / "adapter_config.json").exists()

    def status(self) -> Dict[str, object]:
        return {
            "base_model": self.base_model,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "adapter_present": self.adapter_is_present(),
            "adapter_loaded": self.adapter_loaded,
            "adapter_error": self.adapter_error,
            "loaded": self.loaded,
            "load_error": self.load_error,
            "device": self.device,
            "dtype": self.dtype,
            "load_in_4bit": self.load_in_4bit,
            "quantized": self.quantized,
        }

    # ------------------------------------------------------------------ loading
    def load(self) -> bool:
        """Load the model once. Returns success; never raises."""
        if self.loaded:
            return True
        with self._lock:
            if self.loaded:
                return True
            try:
                self._load_unsafe()
                self.loaded = True
                self.load_error = None
            except Exception as error:  # noqa: BLE001 - degrade, never crash the demo
                self._model = None
                self._tokenizer = None
                self.loaded = False
                self.load_error = f"{type(error).__name__}: {error}"
        return self.loaded

    def _load_unsafe(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        has_cuda = torch.cuda.is_available()
        if not has_cuda and not LLM_ALLOW_CPU:
            raise RuntimeError(
                "CUDA is not available. Qwen3-4B reply generation needs an NVIDIA GPU; "
                "the pipeline will use template responses instead. Set C4_ALLOW_CPU=1 "
                "to run it on the CPU anyway (needs ~9 GB of RAM and is far slower)."
            )

        # 4-bit NF4 is a bitsandbytes CUDA kernel; there is no CPU equivalent, so
        # a CPU run loads the unquantized weights and pays for it in RAM.
        use_4bit = self.load_in_4bit and has_cuda
        if has_cuda:
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            compute_dtype = torch.float32

        # Prefer the adapter's own tokenizer: training saved it alongside the
        # adapter, so it carries the exact chat template the LoRA weights expect.
        tokenizer_source = (
            self.adapter_path
            if self.adapter_is_present() and (self.adapter_path / "tokenizer_config.json").exists()
            else self.base_model
        )
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "dtype": compute_dtype,
            "device_map": "auto" if has_cuda else "cpu",
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
        if use_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        model = AutoModelForCausalLM.from_pretrained(self.base_model, **model_kwargs)

        # The adapter is optional: without it this is still a working supportive
        # assistant, just an un-fine-tuned one, and the UI says so.
        if self.adapter_is_present():
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(self.adapter_path))
                self.adapter_loaded = True
                self.adapter_error = None
            except Exception as error:  # noqa: BLE001
                self.adapter_loaded = False
                self.adapter_error = f"{type(error).__name__}: {error}"
        else:
            self.adapter_loaded = False
            self.adapter_error = (
                f"No adapter at {self.adapter_path}. Train one with "
                "qwen3_esconv_finetune/run_train_rtx3080.ps1, or run base-model only."
            )

        model.eval()
        model.config.use_cache = True

        self._model = model
        self._tokenizer = tokenizer
        self.device = str(getattr(model, "device", "cuda" if has_cuda else "cpu"))
        self.dtype = str(compute_dtype)
        self.quantized = use_4bit

    # --------------------------------------------------------------- generation
    def generate(
        self,
        messages: List[Dict[str, str]],
        use_adapter: bool = True,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = LLM_TOP_P,
        repetition_penalty: float = LLM_REPETITION_PENALTY,
    ) -> GenerationResult:
        """Run one chat completion. Raises only if the model is not loaded."""
        import torch

        if not self.available:
            raise RuntimeError(self.load_error or "Generator is not loaded")

        model, tokenizer = self._model, self._tokenizer

        template_kwargs: Dict[str, Any] = {}
        if _is_thinking_checkpoint(self.base_model):
            template_kwargs["enable_thinking"] = False

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **template_kwargs,
        ).to(model.device)

        do_sample = temperature > 0
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=temperature, top_p=top_p)

        # `disable_adapter` only exists once a PeftModel is attached. When the
        # adapter failed to load or was never trained, both routes are the base
        # model and `adapter_used` reports that honestly.
        wants_base = not use_adapter and self.adapter_loaded
        started = time.perf_counter()
        with torch.inference_mode():
            if wants_base:
                with model.disable_adapter():
                    output_ids = model.generate(**inputs, **generation_kwargs)
            else:
                output_ids = model.generate(**inputs, **generation_kwargs)
        latency = time.perf_counter() - started

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, prompt_length:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        return GenerationResult(
            text=_strip_thinking(text).strip(),
            adapter_used=bool(use_adapter and self.adapter_loaded),
            prompt_tokens=int(prompt_length),
            completion_tokens=int(new_tokens.shape[0]),
            latency_seconds=latency,
            generation_kwargs=generation_kwargs,
        )

    def unload(self) -> None:
        """Release VRAM. Used by the Streamlit sidebar when switching models."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            self.loaded = False
            self.adapter_loaded = False
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - unload must not raise
                pass


def _strip_thinking(text: str) -> str:
    """Remove a <think> block if the hybrid checkpoint emitted one anyway."""
    cleaned = _THINK_BLOCK.sub("", text)
    # An unterminated block means generation hit the token budget mid-thought;
    # everything after the opening tag is reasoning, not a reply.
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned
