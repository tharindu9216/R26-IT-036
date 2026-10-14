"""Explainable AI layer for C4.

Three complementary explanation families, all implemented directly on top of
PyTorch so the demo has no dependency on captum / shap / lime (which are
awkward to install on Windows and add ~400 MB):

1. **Integrated Gradients** (Sundararajan et al., 2017) over the input
   embeddings. Signed, per-token, completeness-checked. This is the primary
   explanation for both models.
2. **Occlusion / leave-one-token-out**. Model-agnostic sanity check: mask a
   token, measure how far the predicted probability falls. Slower (one forward
   per token) but makes no assumption about gradients being faithful.
3. **Counterfactual probing** of the forecaster's auxiliary current-emotion
   feature: re-run the forecast under each of the 5 possible current emotions
   and report how much the answer moves. On the retrained next-state forecaster
   this sweep is the headline explanation rather than a footnote -- it makes
   visible that the current emotion, not the sentence, is doing the work, and
   it shows the transition constraint changing the admissible states.

`aggregate_to_words` folds subword pieces back into whole words, because
"unset" / "##tling" is not an explanation anyone can read.
"""

from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from config import IG_MAX_RELATIVE_GAP, IG_STEPS, XAI_TOP_K
from .forecast_models import PAD, simple_tokenize
from .label_mapping import UNKNOWN_AUX_ID

# Tokens that carry no explanatory content and only clutter the display.
_SPECIAL_DISPLAY = {"<s>", "</s>", "[CLS]", "[SEP]", "<pad>", "[PAD]", "<unk>"}


# --------------------------------------------------------------------- helpers
def _normalise(scores: Sequence[float]) -> List[float]:
    """Scale to [-1, 1] by peak magnitude, preserving sign."""
    peak = max((abs(float(s)) for s in scores), default=0.0)
    if peak == 0.0:
        return [0.0 for _ in scores]
    return [float(s) / peak for s in scores]


def _clean_piece(token: str) -> str:
    """Strip subword markers for display."""
    return token.replace("Ġ", "").replace("##", "").replace("▁", "")


def _is_continuation(token: str) -> bool:
    """True when this piece continues the previous word.

    RoBERTa BPE marks word *starts* with 'Ġ'; WordPiece marks continuations
    with '##'. The forecaster's word-level tokenizer uses neither, so it falls
    through to the caller's decision below.
    """
    if token.startswith("##"):
        return True
    if token.startswith("Ġ") or token.startswith("▁"):
        return False
    return None  # caller decides: no marker convention in this tokenizer


def aggregate_to_words(
    tokens: Sequence[str], scores: Sequence[float]
) -> List[Tuple[str, float]]:
    """Merge subword pieces into words, summing their attributions."""
    words: List[Tuple[str, float]] = []
    uses_gpt2_marker = any(t.startswith("Ġ") for t in tokens)
    for token, score in zip(tokens, scores):
        if token in _SPECIAL_DISPLAY:
            continue
        continuation = _is_continuation(token)
        if continuation is None:
            # No marker convention -> only merge when the tokenizer is BPE-style
            continuation = uses_gpt2_marker and bool(words)
        if continuation and words:
            previous_word, previous_score = words[-1]
            words[-1] = (previous_word + _clean_piece(token), previous_score + score)
        else:
            words.append((_clean_piece(token), float(score)))
    return [(word, score) for word, score in words if word.strip()]


def top_k(attributions: Sequence[Tuple[str, float]], k: int = XAI_TOP_K):
    """Highest-magnitude tokens first."""
    return sorted(attributions, key=lambda pair: abs(pair[1]), reverse=True)[:k]


# ------------------------------------------------------- integrated gradients
@contextmanager
def _rnn_backward_safe(model: nn.Module):
    """Allow backward through an LSTM that is in eval mode.

    cuDNN refuses `RNN backward ... in training mode` checks the other way
    round: its fused RNN kernel only keeps the workspace needed for a backward
    pass when the module is training, so autograd through an `eval()` LSTM
    raises. Switching the model to train() instead would enable dropout and
    silently randomise the attributions, so disable the cuDNN path for the
    duration and let the native (differentiable in eval) implementation run.
    """
    has_rnn = any(isinstance(module, nn.RNNBase) for module in model.modules())
    if not has_rnn or not torch.backends.cudnn.is_available():
        yield
        return
    previous = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = previous


def _integrated_gradients(
    forward_fn,
    input_embeddings: torch.Tensor,
    baseline_embeddings: torch.Tensor,
    target_index: int,
    steps: int = IG_STEPS,
) -> Tuple[torch.Tensor, float, float]:
    """Riemann-midpoint IG for a single example.

    `forward_fn(embeddings) -> logits [B, C]`. Returns per-token attributions,
    the absolute completeness gap |sum(attr) - (F(x) - F(baseline))| and the
    same gap relative to |F(x) - F(baseline)|. Completeness is the standard
    check that the Riemann approximation converged; the relative form is the
    one worth thresholding, because the absolute gap scales with the logit
    range and means nothing on its own.
    """
    delta = input_embeddings - baseline_embeddings
    alphas = (
        torch.arange(steps, device=input_embeddings.device, dtype=input_embeddings.dtype)
        + 0.5
    ) / steps

    total_gradient = torch.zeros_like(input_embeddings)
    for alpha in alphas:
        interpolated = (baseline_embeddings + alpha * delta).detach().requires_grad_(True)
        logits = forward_fn(interpolated)
        score = logits[0, target_index]
        (gradient,) = torch.autograd.grad(score, interpolated)
        total_gradient += gradient

    average_gradient = total_gradient / steps
    attributions = (delta * average_gradient).sum(dim=-1).squeeze(0)

    with torch.no_grad():
        end = float(forward_fn(input_embeddings)[0, target_index])
        start = float(forward_fn(baseline_embeddings)[0, target_index])
    span = end - start
    absolute_gap = abs(float(attributions.sum()) - span)
    relative_gap = absolute_gap / max(abs(span), 1e-6)
    return attributions.detach().cpu(), absolute_gap, relative_gap


def _integrated_gradients_converged(
    model: nn.Module,
    forward_fn,
    input_embeddings: torch.Tensor,
    baseline_embeddings: torch.Tensor,
    target_index: int,
    steps: int = IG_STEPS,
) -> Tuple[torch.Tensor, float, float, int]:
    """IG with one automatic refinement if completeness has not converged.

    An unconverged Riemann sum produces attributions that look plausible and
    are wrong, so rather than reporting the gap and moving on, retry once at
    double the step count. Returns the step count actually used.
    """
    with _rnn_backward_safe(model):
        attributions, absolute_gap, relative_gap = _integrated_gradients(
            forward_fn, input_embeddings, baseline_embeddings, target_index, steps
        )
        if relative_gap > IG_MAX_RELATIVE_GAP:
            steps *= 2
            attributions, absolute_gap, relative_gap = _integrated_gradients(
                forward_fn, input_embeddings, baseline_embeddings, target_index, steps
            )
    return attributions, absolute_gap, relative_gap, steps


# ------------------------------------------------------------------ classifier
def explain_classifier(
    classifier,
    text: str,
    target_label: Optional[str] = None,
    steps: int = IG_STEPS,
    include_occlusion: bool = True,
) -> Dict[str, object]:
    """Token attributions for the current-emotion classifier."""
    if classifier.fallback or classifier.model is None:
        return {
            "available": False,
            "reason": "no trained classifier loaded; keyword fallback has no attributions",
        }

    model, tokenizer = classifier.model, classifier.tokenizer
    inputs = classifier.encode(text)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits.squeeze(0), dim=-1)
    if target_label is None:
        target_index = int(torch.argmax(probabilities).item())
    else:
        target_index = next(
            (i for i, l in classifier.label_mapping.items() if l == target_label),
            int(torch.argmax(probabilities).item()),
        )
    target_label = classifier.label_mapping.get(target_index, str(target_index))

    embedding_layer = model.get_input_embeddings()
    input_embeddings = embedding_layer(input_ids).detach()

    # Baseline = the pad token everywhere except the special tokens that frame
    # the sequence; keeping <s>/</s> avoids explaining "the model saw a valid
    # sequence at all" as if it were evidence about the emotion.
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    baseline_ids = torch.full_like(input_ids, pad_id)
    special_mask = torch.tensor(
        tokenizer.get_special_tokens_mask(
            input_ids[0].tolist(), already_has_special_tokens=True
        ),
        device=input_ids.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    baseline_ids = torch.where(special_mask, input_ids, baseline_ids)
    baseline_embeddings = embedding_layer(baseline_ids).detach()

    def forward_fn(embeddings: torch.Tensor) -> torch.Tensor:
        return model(inputs_embeds=embeddings, attention_mask=attention_mask).logits

    attributions, absolute_gap, relative_gap, steps = _integrated_gradients_converged(
        model, forward_fn, input_embeddings, baseline_embeddings, target_index, steps
    )

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    word_attributions = aggregate_to_words(tokens, attributions.tolist())
    normalised = list(
        zip(
            [w for w, _ in word_attributions],
            _normalise([s for _, s in word_attributions]),
        )
    )

    result: Dict[str, object] = {
        "available": True,
        "method": "integrated_gradients",
        "steps": steps,
        "target_label": target_label,
        "target_probability": float(probabilities[target_index]),
        "completeness_gap": absolute_gap,
        "completeness_gap_relative": relative_gap,
        "token_attributions": normalised,
        "top_tokens": top_k(normalised),
    }

    if include_occlusion:
        result["occlusion"] = _occlusion_classifier(classifier, text, target_index)
    return result


def _occlusion_classifier(classifier, text: str, target_index: int) -> Dict[str, object]:
    """Mask each word in turn; the drop in target probability is its importance."""
    tokenizer = classifier.tokenizer
    words = text.split()
    if not words or len(words) > 60:  # one forward per word; cap the cost
        return {"available": False, "reason": "message too short or too long to occlude"}

    mask_token = tokenizer.mask_token or tokenizer.unk_token or ""
    variants = [text] + [
        " ".join(words[:i] + [mask_token] + words[i + 1:]) for i in range(len(words))
    ]
    probabilities = classifier.probabilities_for_texts(variants)
    base = float(probabilities[0, target_index])
    drops = [base - float(probabilities[i + 1, target_index]) for i in range(len(words))]
    scored = list(zip(words, _normalise(drops)))
    return {
        "available": True,
        "method": "occlusion",
        "base_probability": base,
        "token_attributions": scored,
        "top_tokens": top_k(scored),
    }


# ------------------------------------------------------------------ forecaster
def explain_forecaster(
    forecaster,
    context_text: str,
    aux_id: int,
    target_label: Optional[str] = None,
    steps: int = IG_STEPS,
) -> Dict[str, object]:
    """Token attributions over the utterance the forecaster reads.

    `context_text` is the single current utterance now, not a dialogue window --
    the retrained model's leakage-safe input is that utterance plus the current
    emotion. The parameter name is kept for continuity with existing callers.

    Read these attributions with the caveat in `EmotionForecaster`'s docstring
    in mind: on this corpus the text does not separate the reachable next
    states, so a large attribution marks a token the model reacts to, not
    evidence that the token predicts escalation. `aux_contribution` below is
    the honest comparison.
    """
    if forecaster.fallback or forecaster.model is None:
        return {
            "available": False,
            "reason": "rule-based fallback forecaster has no learned attributions",
        }

    model = forecaster.model
    aux = torch.tensor([aux_id], dtype=torch.long, device=forecaster.device)

    input_ids, lens = forecaster.encode_context(context_text)
    embedding_layer = model.emb
    inverse_vocab = {index: token for token, index in forecaster.vocab.items()}
    tokens = [inverse_vocab.get(i, "<unk>") for i in input_ids[0].tolist()]
    baseline_ids = torch.full_like(input_ids, PAD)

    def forward_fn(embeddings):
        return model.forward_from_embeddings(embeddings, input_ids, lens, aux)

    input_embeddings = embedding_layer(input_ids).detach()
    baseline_embeddings = embedding_layer(baseline_ids).detach()

    with torch.no_grad():
        probabilities = torch.softmax(forward_fn(input_embeddings).float().squeeze(0), dim=-1)
    if target_label is None:
        target_index = int(torch.argmax(probabilities).item())
    else:
        target_index = (
            forecaster.labels.index(target_label)
            if target_label in forecaster.labels
            else int(torch.argmax(probabilities).item())
        )

    attributions, absolute_gap, relative_gap, steps = _integrated_gradients_converged(
        model, forward_fn, input_embeddings, baseline_embeddings, target_index, steps
    )

    # Drop padding before display: the model sees it, but it explains nothing.
    scores = attributions.tolist()
    keep = [i for i, token_id in enumerate(input_ids[0].tolist()) if token_id != PAD]
    tokens = [tokens[i] for i in keep]
    scores = [scores[i] for i in keep]

    word_attributions = aggregate_to_words(tokens, scores)
    normalised = list(
        zip(
            [w for w, _ in word_attributions],
            _normalise([s for _, s in word_attributions]),
        )
    )

    return {
        "available": True,
        "method": "integrated_gradients",
        "steps": steps,
        "architecture": forecaster.architecture,
        "target_label": forecaster.labels[target_index],
        "target_probability": float(probabilities[target_index]),
        "completeness_gap": absolute_gap,
        "completeness_gap_relative": relative_gap,
        "token_attributions": normalised,
        "top_tokens": top_k(normalised),
        "aux_contribution": _aux_contribution(forecaster, context_text, aux_id, target_index),
    }


def _aux_contribution(
    forecaster, context_text: str, aux_id: int, target_index: int
) -> Dict[str, object]:
    """How much of the forecast comes from the current-emotion feature vs the text.

    Ablation: re-run with the aux feature set to the 'unknown current emotion'
    bucket the model was trained to handle, and report the change in the target
    probability.

    Both runs are UNMASKED -- no transition constraint is applied to either --
    so the comparison isolates the aux embedding rather than measuring the mask
    twice. The numbers here therefore will not match the masked `confidence`
    reported alongside the forecast, and that is deliberate.
    """
    from config import FORECAST_CURRENT_EMOTIONS

    target_label = forecaster.labels[target_index]
    with_aux = forecaster._model_probabilities(context_text, aux_id)
    without_aux = forecaster._model_probabilities(context_text, UNKNOWN_AUX_ID)
    top_without = max(without_aux, key=without_aux.__getitem__)
    return {
        "current_emotion_feature": (
            FORECAST_CURRENT_EMOTIONS[aux_id]
            if aux_id < len(FORECAST_CURRENT_EMOTIONS) else "unknown"
        ),
        "masked": False,
        "probability_with_feature": with_aux[target_label],
        "probability_without_feature": without_aux[target_label],
        "delta": with_aux[target_label] - without_aux[target_label],
        "prediction_without_feature": top_without,
        "prediction_flips_without_feature": top_without != target_label,
    }


def counterfactual_current_emotion(forecaster, context_text: str) -> Dict[str, object]:
    """Sweep the aux feature across all current emotions and summarise the spread.

    On the next-state forecaster this is the most informative panel in the app.
    The transition constraint means each assumed current emotion admits a
    different set of next states, so the sweep shows the structure the model
    actually encodes -- and, by contrast with the flat token attributions, how
    little the sentence itself moves the answer.
    """
    sweep = forecaster.counterfactual_by_current_emotion(context_text)
    if not sweep:
        return {"available": False, "reason": "no trained forecaster loaded"}
    predicted = {emotion: result["label"] for emotion, result in sweep.items()}
    distinct = sorted(set(predicted.values()))
    return {
        "available": True,
        "sweep": {
            emotion: {
                "label": result["label"],
                "confidence": result["confidence"],
                "trajectory": result.get("trajectory"),
            }
            for emotion, result in sweep.items()
        },
        "distinct_outcomes": distinct,
        "sensitivity": len(distinct) / max(1, len(predicted)),
        "note": (
            "The forecast changes with the assumed current emotion, so the "
            "classifier's output materially drives it."
            if len(distinct) > 1
            else "The forecast is the same under every assumed current emotion, "
                 "so it is driven by the utterance text alone."
        ),
    }


# ------------------------------------------------------------------ vocabulary
def vocabulary_coverage(forecaster, context_text: str) -> Dict[str, object]:
    """Which words the forecaster's word-level vocabulary does not know.

    Not an attribution method, but a necessary caveat for reading any
    forecaster explanation: an out-of-vocabulary word contributes only the
    generic `<unk>` embedding, so its attribution says nothing about that word.
    The retrained vocabulary is built from the training split of a ~10k-row
    corpus of real DailyDialog-style sentences, far broader than the 528-token
    synthetic vocabulary the previous forecaster shipped with, so OOV rates on
    ordinary chat input are much lower than they used to be.
    """
    if not forecaster.vocab:
        return {"available": False, "reason": "no word-level vocabulary loaded"}
    tokens = simple_tokenize(context_text)
    unknown = [token for token in tokens if token not in forecaster.vocab]
    return {
        "available": True,
        "vocab_size": len(forecaster.vocab),
        "n_tokens": len(tokens),
        "n_unknown": len(unknown),
        "unknown_rate": len(unknown) / max(1, len(tokens)),
        "unknown_tokens": sorted(set(unknown))[:25],
    }
