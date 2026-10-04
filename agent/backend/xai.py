"""Shared Captum Integrated Gradients helpers for on-demand explanations.

Every classifier component (C1 sensor, C3 Stress/CBT headers, current
emotion, next intensity) exposes an ``.explain(...)`` method built on the two
functions here. The reply generator (Qwen) has no XAI: it is a free-form
generator, not a classifier, so there is no single target class to
attribute an output to.

Models must not be run under ``torch.inference_mode()`` before calling into
these -- Integrated Gradients needs a real backward pass, and inference
tensors cannot be used with autograd (the same constraint the C3 ensembles'
``predict_proba`` already documents).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients


@dataclass(frozen=True)
class TokenAttribution:
    token: str
    score: float


@dataclass(frozen=True)
class FeatureAttribution:
    feature: str
    score: float


def explain_tokens(
    *,
    forward_func: Callable[..., torch.Tensor],
    embedding_layer: torch.nn.Module,
    input_ids: torch.Tensor,
    baseline_ids: torch.Tensor,
    tokens: Sequence[str],
    additional_forward_args: tuple = (),
    n_steps: int = 32,
    internal_batch_size: int | None = 1,
) -> list[TokenAttribution]:
    """Layer Integrated Gradients attribution for one encoded text input.

    ``forward_func(input_ids, *additional_forward_args) -> Tensor[batch]``
    must return the target class's probability (or logit) for each item in
    the batch -- i.e. the target class is already baked into it by the
    caller, since each component's output shape (2-class binary heads vs.
    5/13-class classifiers) differs.
    """

    lig = LayerIntegratedGradients(forward_func, embedding_layer)
    attributions, _delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=additional_forward_args,
        n_steps=n_steps,
        # Captum otherwise evaluates all interpolation points together.  That
        # is fast on a large accelerator but can multiply transformer memory
        # by n_steps and make a laptop swap or run out of VRAM.
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    per_token = attributions.sum(dim=-1).squeeze(0)
    norm = per_token.norm()
    if norm > 1e-12:
        per_token = per_token / norm
    scores = per_token.detach().cpu().tolist()
    return [
        TokenAttribution(token=token, score=float(score))
        for token, score in zip(tokens, scores, strict=True)
    ]


def explain_features(
    *,
    forward_func: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    feature_names: Sequence[str],
    n_steps: int = 64,
    internal_batch_size: int | None = 1,
) -> list[FeatureAttribution]:
    """Plain Integrated Gradients attribution over continuous tabular features."""

    ig = IntegratedGradients(forward_func)
    attributions, _delta = ig.attribute(
        inputs=inputs,
        baselines=baseline,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    scores = attributions.squeeze(0).detach().cpu().tolist()
    return [
        FeatureAttribution(feature=name, score=float(score))
        for name, score in zip(feature_names, scores, strict=True)
    ]


def pad_preserved_baseline(
    input_ids: torch.Tensor,
    pad_token_id: int,
    special_tokens_mask: Sequence[int] | None = None,
) -> torch.Tensor:
    """An all-pad baseline that keeps special tokens (CLS/SEP/...) identical
    to the real input, so IG only attributes the actual content tokens."""

    baseline = torch.full_like(input_ids, pad_token_id)
    if special_tokens_mask is not None:
        mask = torch.tensor(special_tokens_mask, device=input_ids.device).bool()
        baseline[0][mask] = input_ids[0][mask]
    return baseline
