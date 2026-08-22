"""CBT names and safeguards around the shared local-XAI implementations."""

from __future__ import annotations

import sys
from pathlib import Path


XAI_DIR = Path(__file__).resolve().parent
SHARED_XAI_DIR = XAI_DIR.parents[1] / "Stress header" / "xai"
if str(SHARED_XAI_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_XAI_DIR))

from combined_explainer import StressCombinedXAI  # noqa: E402
from counterfactual_explainer import StressCounterfactual  # noqa: E402
from integrated_gradients import StressIG  # noqa: E402
from lime_explainer import StressLIME  # noqa: E402
from shap_explainer import StressSHAP  # noqa: E402


class CBTIG(StressIG):
    """Integrated Gradients for CBT Distortion/No Distortion models."""


class CBTSHAP(StressSHAP):
    """SHAP text explanations for CBT Distortion/No Distortion models."""


class CBTLIME(StressLIME):
    """LIME word explanations for CBT Distortion/No Distortion models."""


class CBTCombinedXAI(StressCombinedXAI):
    """Consensus SHAP + LIME + Integrated Gradients CBT explanation."""


class CBTCounterfactual(StressCounterfactual):
    """Minimal text edits that change a CBT binary model decision."""

    _NEUTRAL_PAIRS = (
        ("always", "sometimes"),
        ("never", "sometimes"),
        ("everyone", "some people"),
        ("nobody", "some people"),
        ("must", "could"),
        ("should", "could"),
        ("failure", "setback"),
        ("worthless", "imperfect"),
        ("hopeless", "uncertain"),
        ("impossible", "difficult"),
        ("terrible", "difficult"),
        ("ruined", "affected"),
    )

    def explain(self, *args, **kwargs) -> dict:
        result = super().explain(*args, **kwargs)
        result["disclaimer"] = (
            "Counterfactual text describes model sensitivity only; it is not "
            "a CBT assessment, diagnosis, or advice to change how someone "
            "describes their thoughts."
        )
        return result
