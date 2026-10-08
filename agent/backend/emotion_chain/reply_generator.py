"""Single-load Qwen3 base/ESConv adapter reply generation.

The model can live in this process (the single-machine default) or on a
support node reached over the LAN -- set ``remote_url`` and the class becomes
a thin HTTP client of ``node/server.py``. Routing, prompt construction,
crisis handling, escalation contacts and the template fallback are identical
either way: only the token-generation step moves, so the deployment split can
never change what the agent decides.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .core import select_reply_route
from .fusion import (
    DEFAULT_INFLUENCE_WEIGHTS,
    InfluenceWeights,
    weighted_influence,
)
from .support_contacts import (
    DEFAULT_SUPPORT_CONTACTS,
    SupportContact,
    append_contact_message,
    is_extremely_negative,
)


BASE_SYSTEM_PROMPT = """You are a warm, concise conversational companion.
Respond naturally to the user's latest message. Do not diagnose mental-health
conditions, claim to be a therapist, or invent facts. Ask at most one useful
follow-up question. Never write a phone number or the name of a helpline: the
application appends verified contact details itself when they are needed."""

SUPPORTIVE_SYSTEM_PROMPT = """You are a supportive emotional-support companion.
Respond in 3-4 sentences: acknowledge the user's feeling, briefly reflect back
what you heard, offer one concrete and realistic next step, then ask one gentle
follow-up question. Be warm, specific and non-judgmental. Do not diagnose,
shame, lecture, or claim to replace professional care. Never write a phone
number or the name of a helpline: the application appends verified contact
details itself when they are needed."""

INFLUENCE_PREAMBLE = (
    "Stress evidence for this turn, and how much each source should influence "
    "your reply (1.00 = primary evidence; lower = a weaker supporting hint that "
    "may be wrong, so lean on it less). Weigh them accordingly, and never name a "
    "source or quote a number to the user:"
)

LOGGER = logging.getLogger(__name__)

CRISIS_RESPONSE = (
    "I'm really sorry you're facing this. Your immediate safety matters most. "
    "Please move away from anything you could use to hurt yourself, contact a "
    "trusted person who can stay with you, and contact your local emergency or "
    "crisis service now. If you are in immediate danger, go to the nearest "
    "emergency department or call local emergency services."
)


def crisis_reply(
    contacts: Sequence[SupportContact] = DEFAULT_SUPPORT_CONTACTS,
) -> str:
    """The fixed crisis template with the configured contacts appended."""

    return append_contact_message(CRISIS_RESPONSE, contacts)


@dataclass(frozen=True)
class ReplySignals:
    current_emotion: str
    next_emotion: str
    next_emotion_confidence: float | None = None
    next_negative_min_confidence: float = 0.0
    # Turn-to-turn deviation (see deviation_tracker.py). "None" on the first
    # turn of a conversation, which is not the same as a calm one.
    previous_emotion: str | None = None
    deviation_level: str = "None"
    deviation_score: float = 0.0
    sensor_stress: bool = False
    # User-selected decision-level fusion of the C3 Stress + CBT headers.
    text_signal_flagged: bool = False
    # Voice-mode C2 appraisal/stress signal; C3 is not run in voice mode.
    voice_stress: bool = False
    questionnaire_stress: bool = False
    multimodal_stress_available: bool = False
    multimodal_stress: bool = False
    sensor_stress_probability: float | None = None
    text_signal_probability: float | None = None
    voice_stress_score: float | None = None
    questionnaire_stress_probability: float | None = None
    multimodal_stress_probability: float | None = None
    voice_appraisal_state: str | None = None
    voice_appraisal_uncertain: bool = False
    # Raw per-header probabilities are kept for logging/UI display.
    stress_probability: float | None = None
    cbt_probability: float | None = None
    safety_status: str = "normal"


@dataclass(frozen=True)
class ReplyGenerationResult:
    text: str
    route: str
    source: str
    # Contacts appended to ``text``; empty unless the turn was escalated.
    support_contacts: tuple[SupportContact, ...] = ()

    @property
    def escalated(self) -> bool:
        return bool(self.support_contacts)


class QwenReplyGenerator:
    """Attach the local ESConv LoRA once and enable/disable it per request."""

    def __init__(
        self,
        base_model_dir: str | Path,
        adapter_dir: str | Path,
        *,
        device: str = "auto",
        use_adapter: bool = True,
        max_new_tokens: int = 160,
        temperature: float = 0.7,
        top_p: float = 0.8,
        history_turns: int = 6,
        quantization: str = "none",
        influence_weights: InfluenceWeights = DEFAULT_INFLUENCE_WEIGHTS,
        allow_template_fallback: bool = True,
        support_contacts: Sequence[SupportContact] = DEFAULT_SUPPORT_CONTACTS,
        remote_url: str | None = None,
        remote_timeout: float = 180.0,
        remote_auth_token: str | None = None,
    ) -> None:
        self.base_model_dir = Path(base_model_dir).expanduser().resolve()
        self.adapter_dir = Path(adapter_dir).expanduser().resolve()
        self.device = device
        self.use_adapter = bool(use_adapter)
        self.support_contacts = tuple(support_contacts)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        # How many earlier turns are replayed into the prompt. Each one is
        # also KV cache: on a 4 GB card the full six can push peak VRAM past
        # what the GPU has, so it is a config knob rather than a constant.
        self.history_turns = max(0, int(history_turns))
        self.quantization = str(quantization or "none").lower()
        # How much the sensor / voice / text stress signals may sway the wording
        # of a reply. Deployment config rather than a per-turn signal, so it
        # sits here with the other generation knobs, not on ReplySignals.
        self.influence_weights = influence_weights
        self.allow_template_fallback = allow_template_fallback
        self.remote_url = remote_url.rstrip("/") if remote_url else None
        self.remote_timeout = float(remote_timeout)
        self.remote_auth_token = remote_auth_token
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

        # A remote generator never touches the local weights, so the machine
        # running it does not need the 7.6 GB Qwen directory on disk at all.
        if self.remote_url is None:
            if not (self.base_model_dir / "config.json").is_file():
                raise FileNotFoundError(
                    f"Qwen base model not found: {self.base_model_dir}"
                )
            if not (self.adapter_dir / "adapter_config.json").is_file():
                raise FileNotFoundError(f"ESConv adapter not found: {self.adapter_dir}")

    @property
    def is_remote(self) -> bool:
        return self.remote_url is not None

    @property
    def is_loaded(self) -> bool:
        if self.is_remote:
            return True
        return self._model is not None and self._tokenizer is not None

    def _quantization_config(self):
        """Translate ``quantization`` into a bitsandbytes config, or None."""

        if self.quantization in {"", "none", "full", "bf16", "fp16"}:
            return None
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - transformers is required
            raise RuntimeError("Quantized loading requires transformers") from exc
        if self.quantization in {"nf4", "4bit", "int4"}:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                # Quantizes the quantization constants too: ~0.37 bits per
                # parameter, about 190 MB off a 4B model, for no measurable
                # quality cost.
                bnb_4bit_use_double_quant=True,
            )
        if self.quantization in {"int8", "8bit"}:
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ValueError(
            f"Unsupported reply quantization: {self.quantization!r} "
            "(expected none, nf4 or int8)"
        )

    def load(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            try:
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen generation requires transformers, peft and accelerate"
                ) from exc

            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_dir,
                local_files_only=True,
            )
            load_kwargs = {
                "local_files_only": True,
                "torch_dtype": "auto",
                "low_cpu_mem_usage": True,
            }
            if self.device == "auto":
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["device_map"] = {"": self.device}
            quantization_config = self._quantization_config()
            if quantization_config is not None:
                # bitsandbytes keeps the non-quantized tensors (embeddings,
                # norms, the LoRA) in the compute dtype, so "auto" -- bf16 for
                # this checkpoint -- has to give way to the fp16 the 4-bit
                # kernels expect.
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["quantization_config"] = quantization_config
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_dir,
                **load_kwargs,
            )
            model = PeftModel.from_pretrained(
                base_model,
                self.adapter_dir,
                adapter_name="supportive",
                is_trainable=False,
                local_files_only=True,
            )
            model.eval()
            self._tokenizer = tokenizer
            self._model = model

    def _signal_influence(self, signals: ReplySignals) -> list[str]:
        """One line per stress source that ran, weighted by how far it counts.

        Each model already reports its own confidence; the configured weight
        caps how far that confidence is allowed to move the reply. A source
        that did not run this turn is left out entirely rather than reported
        as calm -- C3 in voice mode, C2 in text mode, C1 with no sensor.
        """

        weights = self.influence_weights
        raw_sources = (
            (
                "text (what the user wrote; stress+CBT fused)",
                weights.text,
                signals.text_signal_flagged,
                signals.text_signal_probability,
                "",
            ),
            (
                "body sensor (wearable)",
                weights.sensor,
                signals.sensor_stress,
                signals.sensor_stress_probability,
                "",
            ),
            (
                "voice tone (appraisal)",
                weights.voice,
                signals.voice_stress,
                signals.voice_stress_score,
                f" [appraisal state={signals.voice_appraisal_state}, "
                f"uncertain={signals.voice_appraisal_uncertain}]",
            ),
        )
        sources = (
            raw_sources[:1]
            + (
                (
                    "combined stress check (questionnaire 60% + wearable 40%)",
                    1.0,
                    signals.multimodal_stress,
                    signals.multimodal_stress_probability,
                    "",
                ),
            )
            if signals.multimodal_stress_available
            else raw_sources
        )
        lines = []
        for name, weight, flagged, probability, extra in sources:
            influence = weighted_influence(
                weight, flagged=flagged, probability=probability
            )
            if influence is None:
                continue
            state = "stress detected" if flagged else "no stress detected"
            lines.append(f"- {name}: {state}, influence {influence:.2f} of 1.00{extra}")
        return lines

    def _messages(
        self,
        user_text: str,
        signals: ReplySignals,
        route: str,
        history: Sequence[tuple[str, str]],
    ) -> list[dict[str, str]]:
        system_prompt = (
            SUPPORTIVE_SYSTEM_PROMPT if route == "supportive_adapter" else BASE_SYSTEM_PROMPT
        )
        signal_summary = (
            "Internal context (do not quote scores or labels to the user): "
            f"current emotion={signals.current_emotion}; "
            f"forecast state={signals.next_emotion}; "
            f"forecast confidence={signals.next_emotion_confidence}; "
            f"previous turn emotion={signals.previous_emotion}; "
            f"deviation since previous turn={signals.deviation_level}."
        )
        influence = self._signal_influence(signals)
        if influence:
            signal_summary += (
                "\n" + INFLUENCE_PREAMBLE + "\n" + "\n".join(influence)
            )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt + "\n\n" + signal_summary}
        ]
        # Two messages per turn (the user's and the reply), so the slice is
        # doubled: `history_turns` counts exchanges, not individual messages.
        window = history[-(self.history_turns * 2) :] if self.history_turns else ()
        for speaker, text in window:
            if not text.strip():
                continue
            normalized = speaker.strip().lower()
            role = "assistant" if normalized in {"assistant", "supporter"} else "user"
            messages.append({"role": role, "content": text.strip()})
        messages.append({"role": "user", "content": user_text.strip()})
        return messages

    def generate(
        self,
        user_text: str,
        signals: ReplySignals,
        history: Sequence[tuple[str, str]] = (),
    ) -> ReplyGenerationResult:
        if not user_text.strip():
            raise ValueError("user_text cannot be empty")
        if signals.safety_status.strip().lower() == "crisis":
            return ReplyGenerationResult(
                text=crisis_reply(self.support_contacts),
                route="crisis",
                source="safety_template",
                support_contacts=self.support_contacts,
            )
        route = select_reply_route(
            signals.current_emotion,
            signals.next_emotion,
            sensor_stress=(
                signals.sensor_stress if not signals.multimodal_stress_available else False
            ),
            text_signal_flagged=signals.text_signal_flagged,
            voice_stress=(
                signals.voice_stress if not signals.multimodal_stress_available else False
            ),
            multimodal_stress=(
                signals.multimodal_stress if signals.multimodal_stress_available else False
            ),
            next_emotion_confidence=signals.next_emotion_confidence,
            next_negative_min_confidence=signals.next_negative_min_confidence,
        )
        try:
            self.load()
            result = self._generate_model_reply(user_text, signals, route, history)
        except Exception as exc:
            if not self.allow_template_fallback:
                raise
            # Worth a warning rather than a silent degrade: with a support
            # node this is usually the node being down or unreachable, and the
            # only visible symptom is `source="template_fallback"` in the UI.
            LOGGER.warning(
                "Reply generation failed (%s: %s); using the routed template. "
                "Reply model: %s",
                type(exc).__name__,
                exc,
                self.remote_url or self.base_model_dir,
            )
            result = ReplyGenerationResult(
                text=self._fallback_reply(user_text, route),
                route=route,
                source="template_fallback",
            )
        return self._escalate(result, signals)

    def _escalate(
        self, result: ReplyGenerationResult, signals: ReplySignals
    ) -> ReplyGenerationResult:
        """Append the configured contacts when the turn reads as extreme.

        Applied after generation rather than inside it, so the template
        fallback and the model path escalate identically and the number itself
        is never something the model produced.
        """

        if not self.support_contacts:
            return result
        if not is_extremely_negative(
            signals.current_emotion,
            signals.next_emotion,
            deviation_level=signals.deviation_level,
            safety_status=signals.safety_status,
            sensor_stress=(
                signals.sensor_stress if not signals.multimodal_stress_available else False
            ),
            text_signal_flagged=signals.text_signal_flagged,
            voice_stress=(
                signals.voice_stress if not signals.multimodal_stress_available else False
            ),
            multimodal_stress=(
                signals.multimodal_stress if signals.multimodal_stress_available else False
            ),
        ):
            return result
        return ReplyGenerationResult(
            text=append_contact_message(result.text, self.support_contacts),
            route=result.route,
            source=result.source,
            support_contacts=self.support_contacts,
        )

    def _generate_model_reply(
        self,
        user_text: str,
        signals: ReplySignals,
        route: str,
        history: Sequence[tuple[str, str]],
    ) -> ReplyGenerationResult:
        messages = self._messages(user_text, signals, route, history)
        # `route` stays the routing decision (and picks the prompt above);
        # this flag only decides whether the LoRA weights are active for it.
        adapter_active = route == "supportive_adapter" and self.use_adapter
        text = (
            self._remote_completion(messages, adapter_active)
            if self.is_remote
            else self._local_completion(messages, adapter_active)
        )
        if not text:
            raise RuntimeError("Qwen generated an empty reply")
        return ReplyGenerationResult(
            text=text,
            route=route,
            source="qwen_remote" if self.is_remote else "qwen",
        )

    def _remote_completion(
        self, messages: list[dict[str, str]], adapter_active: bool
    ) -> str:
        """Ask the support node for the completion of an already-built prompt.

        Only the generated text crosses the network: every routing, safety and
        escalation decision has already been made here, and is made again the
        same way on a machine that runs the model locally.
        """

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "Offloading reply generation requires requests; "
                "install backend/requirements.txt"
            ) from exc

        headers = {"Content-Type": "application/json"}
        if self.remote_auth_token:
            headers["X-Auth-Token"] = self.remote_auth_token
        response = requests.post(
            f"{self.remote_url}/v1/reply/generate",
            json={
                "messages": messages,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "use_adapter": adapter_active,
            },
            headers=headers,
            timeout=self.remote_timeout,
        )
        response.raise_for_status()
        return str(response.json().get("text", "")).strip()

    def _local_completion(
        self, messages: list[dict[str, str]], adapter_active: bool
    ) -> str:
        with self._generation_lock:
            encoded = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_device = self._model.get_input_embeddings().weight.device
            # transformers >= 5 returns a BatchEncoding here; 4.x returned the
            # input_ids tensor directly. BatchEncoding is a UserDict, so this
            # has to test Mapping rather than dict.
            if isinstance(encoded, Mapping):
                model_inputs = {
                    key: value.to(input_device)
                    for key, value in encoded.items()
                    if key in {"input_ids", "attention_mask"}
                }
            else:
                model_inputs = {"input_ids": encoded.to(input_device)}
            input_ids = model_inputs["input_ids"]
            generation_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            if adapter_active:
                self._model.set_adapter("supportive")
                output = self._model.generate(**model_inputs, **generation_kwargs)
            else:
                with self._model.disable_adapter():
                    output = self._model.generate(
                        **model_inputs, **generation_kwargs
                    )
            generated = output[0, input_ids.shape[1] :]
            return self._tokenizer.decode(
                generated,
                skip_special_tokens=True,
            ).strip()

    @staticmethod
    def _fallback_reply(user_text: str, route: str) -> str:
        if route == "supportive_adapter":
            return (
                "That sounds difficult, and it makes sense that this is affecting you. "
                "Try taking one slow breath and choosing the smallest next step you can "
                "manage right now. What part feels hardest at this moment?"
            )
        return "Thanks for sharing that. What would you like to explore or do next?"
