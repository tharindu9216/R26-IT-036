from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from .config import (
    RAW_FEATURES,
    REFERENCE_CONFIDENCE_QUANTILE,
    MembershipReference,
)
from .data_utils import validate_input
from .fuzzy import (
    decreasing_membership,
    increasing_membership,
    medium_membership,
    safe_scale,
)


class AdvancedAppraisalEngine:
    """
    Version 3 appraisal-theory-guided fuzzy pattern estimator.

    Main improvements:
    1. Stronger negative-valence requirement for negative appraisal states.
    2. New general Negative Activation state.
    3. Dominance is a soft modifier rather than a strict gate.
    4. Calm/Recovery is explicitly suppressed by negative evidence.
    5. Ambiguity is treated as interpretation uncertainty.
    6. The low-confidence threshold is derived from the reference speakers.
    """

    def __init__(
        self,
        confidence_quantile: float = REFERENCE_CONFIDENCE_QUANTILE,
    ):
        self.scaler = StandardScaler()
        self.covariance_model = LedoitWolf()

        self.membership_reference = None
        self.confidence_threshold = None

        self.confidence_quantile = float(confidence_quantile)
        self.is_fitted = False

    # ========================================================
    # 1. Fit reference data
    # ========================================================

    def fit_reference(
        self,
        reference_df: pd.DataFrame,
    ):
        """
        Fit ONLY on reference/training speakers.
        """

        validate_input(reference_df, "reference_df")

        X = (
            reference_df[RAW_FEATURES]
            .astype(float)
            .to_numpy()
        )

        # Reference-only normalization.
        Z = self.scaler.fit_transform(X)

        # Reference distribution for OOD diagnostics.
        self.covariance_model.fit(Z)

        q = {}
        scales = {}

        for i, name in enumerate(RAW_FEATURES):
            q33 = float(np.quantile(Z[:, i], 1 / 3))
            q67 = float(np.quantile(Z[:, i], 2 / 3))

            q[name] = (q33, q67)
            scales[name] = safe_scale(q33, q67)

        self.membership_reference = MembershipReference(
            arousal_q33=q["arousal"][0],
            arousal_q67=q["arousal"][1],

            dominance_q33=q["dominance"][0],
            dominance_q67=q["dominance"][1],

            valence_q33=q["valence"][0],
            valence_q67=q["valence"][1],

            arousal_scale=scales["arousal"],
            dominance_scale=scales["dominance"],
            valence_scale=scales["valence"],
        )

        # First pass: calculate reference interpretation confidence. At this
        # point confidence_threshold is still None, so the low-confidence flag
        # is not applied yet.
        reference_out = self.transform(
            reference_df,
        )

        confidence_values = (
            reference_out["interpretation_confidence"]
            .astype(float)
            .to_numpy()
        )

        # Data-driven uncertainty threshold.
        self.confidence_threshold = float(
            np.quantile(
                confidence_values,
                self.confidence_quantile,
            )
        )

        self.is_fitted = True
        return self

    # ========================================================
    # 2. Normalize A/D/V
    # ========================================================

    def normalize(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if self.membership_reference is None:
            raise RuntimeError(
                "Engine must be fitted first."
            )

        out = df.copy()

        X = (
            out[RAW_FEATURES]
            .astype(float)
            .to_numpy()
        )

        Z = self.scaler.transform(X)

        out["arousal_z"] = Z[:, 0]
        out["dominance_z"] = Z[:, 1]
        out["valence_z"] = Z[:, 2]

        return out

    # ========================================================
    # 3. Fuzzy memberships
    # ========================================================

    def add_fuzzy_memberships(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        out = df.copy()
        r = self.membership_reference

        A = out["arousal_z"].to_numpy(float)
        D = out["dominance_z"].to_numpy(float)
        V = out["valence_z"].to_numpy(float)

        # Arousal / activation
        low_A = decreasing_membership(
            A,
            r.arousal_q33,
            r.arousal_scale,
        )
        high_A = increasing_membership(
            A,
            r.arousal_q67,
            r.arousal_scale,
        )
        mid_A = medium_membership(
            low_A,
            high_A,
        )

        # Dominance used only as a control-related proxy.
        low_C = decreasing_membership(
            D,
            r.dominance_q33,
            r.dominance_scale,
        )
        high_C = increasing_membership(
            D,
            r.dominance_q67,
            r.dominance_scale,
        )
        mid_C = medium_membership(
            low_C,
            high_C,
        )

        # Valence
        unpleasant = decreasing_membership(
            V,
            r.valence_q33,
            r.valence_scale,
        )
        pleasant = increasing_membership(
            V,
            r.valence_q67,
            r.valence_scale,
        )
        neutral_V = medium_membership(
            unpleasant,
            pleasant,
        )

        out["m_low_activation"] = low_A
        out["m_medium_activation"] = mid_A
        out["m_high_activation"] = high_A

        out["m_low_control"] = low_C
        out["m_medium_control"] = mid_C
        out["m_high_control"] = high_C

        out["m_unpleasant"] = unpleasant
        out["m_neutral_valence"] = neutral_V
        out["m_pleasant"] = pleasant

        return out

    # ========================================================
    # 4. Appraisal-inspired states
    # ========================================================

    def add_appraisal_states(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        out = df.copy()

        HA = out["m_high_activation"].to_numpy(float)
        MA = out["m_medium_activation"].to_numpy(float)
        LA = out["m_low_activation"].to_numpy(float)

        LC = out["m_low_control"].to_numpy(float)
        HC = out["m_high_control"].to_numpy(float)

        NEG = out["m_unpleasant"].to_numpy(float)
        NEU = out["m_neutral_valence"].to_numpy(float)
        POS = out["m_pleasant"].to_numpy(float)

        # Strong negative evidence:
        # weak unpleasantness is heavily attenuated.
        strong_negative = NEG ** 2

        # ----------------------------------------------------
        # THREAT descriptor
        # High activation + strong unpleasantness + low control proxy.
        # ----------------------------------------------------
        threat = (
            HA
            * strong_negative
            * LC
        )

        # ----------------------------------------------------
        # CHALLENGE descriptor
        # High activation + strong unpleasantness + high control proxy.
        # ----------------------------------------------------
        challenge = (
            HA
            * strong_negative
            * HC
        )

        # ----------------------------------------------------
        # WITHDRAWAL / DISTRESS
        # Low activation + strong unpleasantness + low control proxy.
        # ----------------------------------------------------
        withdrawal_distress = (
            LA
            * strong_negative
            * LC
        )

        # ----------------------------------------------------
        # NEGATIVE ACTIVATION
        #
        # General negative activation state.
        # Dominance is only a SOFT modifier, not a strict gate.
        #
        # This preserves negative-activation evidence even when dominance is
        # high.
        # ----------------------------------------------------
        activation_strength = np.maximum(
            HA,
            MA,
        )

        control_risk_modifier = (
            0.75
            + 0.25 * LC
        )

        negative_activation = (
            activation_strength
            * strong_negative
            * control_risk_modifier
        )

        # ----------------------------------------------------
        # POSITIVE ACTIVATION
        #
        # High activation + pleasantness.
        # No high-dominance requirement.
        # ----------------------------------------------------
        positive_activation = (
            HA
            * POS
        )

        # ----------------------------------------------------
        # CALM / RECOVERY
        #
        # Low activation + neutral/positive affect,
        # but explicitly suppressed by negative evidence.
        #
        # This is the key V3 change for sadness-like cases.
        # ----------------------------------------------------
        non_negative_affect = np.maximum(
            NEU,
            POS,
        )

        negative_suppression = (
            (1.0 - NEG) ** 1.5
        )

        calm_recovery = (
            LA
            * non_negative_affect
            * negative_suppression
        )

        # ----------------------------------------------------
        # AMBIGUITY
        #
        # Used only for uncertainty.
        # ----------------------------------------------------
        known_strength = np.maximum.reduce(
            [
                threat,
                challenge,
                withdrawal_distress,
                negative_activation,
                positive_activation,
                calm_recovery,
            ]
        )

        ambiguous = np.clip(
            1.0 - known_strength,
            0.0,
            1.0,
        )

        out["state_threat"] = threat
        out["state_challenge"] = challenge
        out["state_withdrawal_distress"] = withdrawal_distress
        out["state_negative_activation"] = negative_activation
        out["state_positive_activation"] = positive_activation
        out["state_calm_recovery"] = calm_recovery

        out["state_ambiguous"] = ambiguous
        out["appraisal_ambiguity"] = ambiguous
        out["appraisal_coverage"] = np.clip(
            1.0 - ambiguous,
            0.0,
            1.0,
        )

        return out

    # ========================================================
    # 5. Dominant appraisal pattern
    # ========================================================

    def add_dominant_appraisal_pattern(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        out = df.copy()

        state_columns = [
            "state_threat",
            "state_challenge",
            "state_withdrawal_distress",
            "state_negative_activation",
            "state_positive_activation",
            "state_calm_recovery",
        ]

        state_names = {
            "state_threat": "threat",
            "state_challenge": "challenge",
            "state_withdrawal_distress": "withdrawal_distress",
            "state_negative_activation": "negative_activation",
            "state_positive_activation": "positive_activation",
            "state_calm_recovery": "calm_recovery",
        }

        state_matrix = out[state_columns].to_numpy(dtype=float)

        dominant_indices = np.argmax(
            state_matrix,
            axis=1,
        )

        best = state_matrix[
            np.arange(len(out)),
            dominant_indices,
        ]

        second = np.partition(
            state_matrix,
            -2,
            axis=1,
        )[:, -2]

        out["dominant_appraisal_state"] = [
            state_names[state_columns[index]]
            for index in dominant_indices
        ]

        out["dominant_appraisal_weight"] = np.max(
            state_matrix,
            axis=1,
        )

        out["appraisal_margin"] = np.clip(
            best - second,
            0.0,
            1.0,
        )

        # State strengths are the primary predictions. These normalized
        # contributions are retained only to support transparent ranking in
        # XAI displays; no additional psychological consequents are applied.
        state_sum = np.maximum(state_matrix.sum(axis=1), 1e-9)
        out["recognized_rule_strength"] = state_matrix.sum(axis=1)

        for index, column in enumerate(state_columns):
            state_name = state_names[column]
            out[f"xai_firing_{state_name}"] = state_matrix[:, index]
            out[f"xai_contribution_{state_name}"] = (
                state_matrix[:, index] / state_sum
            )

        return out

    # ========================================================
    # 6. OOD + interpretation confidence
    # ========================================================

    def add_reference_diagnostics(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        out = df.copy()

        Z = out[
            [
                "arousal_z",
                "dominance_z",
                "valence_z",
            ]
        ].to_numpy(float)

        diff = (
            Z
            - self.covariance_model.location_
        )

        precision = (
            self.covariance_model.precision_
        )

        mahalanobis_sq = np.einsum(
            "ij,jk,ik->i",
            diff,
            precision,
            diff,
        )

        # Reference-distribution similarity in 3D A/D/V space.
        reference_similarity = chi2.sf(
            mahalanobis_sq,
            df=3,
        )

        out["ood_mahalanobis_sq"] = (
            mahalanobis_sq
        )

        out["reference_similarity"] = (
            np.clip(
                reference_similarity,
                0.0,
                1.0,
            )
        )

        out["ood_flag"] = (
            out["reference_similarity"]
            < 0.05
        )

        # ----------------------------------------------------
        # Interpretation confidence
        #
        # Combines:
        # 1. reference similarity
        # 2. dominant-state margin
        # 3. appraisal coverage
        #
        # This is NOT a calibrated predictive probability.
        # ----------------------------------------------------
        ref_term = np.clip(
            out["reference_similarity"].to_numpy(float),
            0.0,
            1.0,
        )

        margin_term = np.clip(
            out["appraisal_margin"].to_numpy(float),
            0.0,
            1.0,
        )

        coverage_term = np.clip(
            out["appraisal_coverage"].to_numpy(float),
            0.0,
            1.0,
        )

        out["interpretation_confidence"] = (
            ref_term
            * margin_term
            * coverage_term
        ) ** (1.0 / 3.0)

        # During the first fit_reference pass the threshold
        # does not exist yet.
        if self.confidence_threshold is None:
            out["low_interpretation_confidence"] = False
        else:
            out["low_interpretation_confidence"] = (
                out["interpretation_confidence"]
                < self.confidence_threshold
            )

        out["interpretation_uncertain"] = (
            out["ood_flag"].to_numpy(bool)
            |
            out["low_interpretation_confidence"].to_numpy(bool)
        )

        return out

    # ========================================================
    # 7. Exact XAI explanation text
    # ========================================================

    def add_xai_text(
        self,
        df: pd.DataFrame,
        top_k: int = 3,
    ) -> pd.DataFrame:

        out = df.copy()

        states = [
            "threat",
            "challenge",
            "withdrawal_distress",
            "negative_activation",
            "positive_activation",
            "calm_recovery",
        ]

        explanations = []

        for _, row in out.iterrows():

            values = [
                (
                    state,
                    float(
                        row[
                            f"state_{state}"
                        ]
                    ),
                )
                for state in states
            ]

            values.sort(
                key=lambda x: abs(x[1]),
                reverse=True,
            )

            top = values[:top_k]

            text = "; ".join(
                f"{name}={value:.3f}"
                for name, value in top
            )

            text += (
                f"; ambiguity="
                f"{float(row['appraisal_ambiguity']):.3f}"
            )

            text += (
                f"; confidence="
                f"{float(row['interpretation_confidence']):.3f}"
            )

            if self.confidence_threshold is not None:
                text += (
                    f"; confidence_threshold="
                    f"{self.confidence_threshold:.3f}"
                )

            if bool(row["ood_flag"]):
                text += (
                    "; WARNING="
                    "outside_reference_distribution"
                )

            if bool(
                row["low_interpretation_confidence"]
            ):
                text += (
                    "; WARNING="
                    "low_interpretation_confidence"
                )

            explanations.append(text)

        out[
            "xai_top_appraisal_contributions"
        ] = explanations

        return out

    # ========================================================
    # 8. Full transform
    # ========================================================

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        validate_input(
            df,
            "input_df",
        )

        if self.membership_reference is None:
            raise RuntimeError(
                "Engine has not been fitted."
            )

        out = self.normalize(df)
        out = self.add_fuzzy_memberships(out)
        out = self.add_appraisal_states(out)
        out = self.add_dominant_appraisal_pattern(out)
        out = self.add_reference_diagnostics(out)
        out = self.add_xai_text(out)

        return out

    # ========================================================
    # 9. Save / load
    # ========================================================

    def save(
        self,
        path: str | Path,
    ):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self,
            path,
        )

    @staticmethod
    def load(
        path: str | Path,
    ):
        return joblib.load(path)
