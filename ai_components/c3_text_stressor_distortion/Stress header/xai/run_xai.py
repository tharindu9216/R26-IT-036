from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path


XAI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = XAI_DIR.parents[3]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "c3_text_stressor_distortion"
    / "Stress header"
    / "evaluation"
    / "xai_outputs"
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(XAI_DIR))

from backend.c3_text_stressor_distortion.app.stress_model import (  # noqa: E402
    StressPredictor,
)
from combined_explainer import StressCombinedXAI  # noqa: E402
from counterfactual_explainer import StressCounterfactual  # noqa: E402
from integrated_gradients import StressIG  # noqa: E402
from lime_explainer import StressLIME  # noqa: E402
from shap_explainer import StressSHAP  # noqa: E402


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"/?r/[A-Za-z0-9_]+|/?u/[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"&#x200B;|\u200b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def expand_common_contractions(text: str) -> str:
    replacements = {
        "can't": "can not",
        "cannot": "can not",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'t": " not",
        "'ve": " have",
        "'m": " am",
    }
    lowered = text.lower()
    for src, dst in replacements.items():
        lowered = lowered.replace(src, dst)
    return lowered


def preprocess_for_transformers(text: str) -> str:
    return expand_common_contractions(clean_text(text))


def serialise_result(method: str, result: dict, plot_path: Path | None = None) -> dict:
    return {
        "method": method,
        "prediction": result["predicted_label"],
        "confidence": result["confidence"],
        "rationales": result.get("rationales", []),
        "plot_path": str(plot_path) if plot_path else None,
    }


def print_rationales(method: str, result: dict) -> None:
    rationales = result.get("rationales", [])
    if not rationales:
        print(f"{method} rationales: none extracted")
        return
    print(f"{method} rationale spans:")
    for index, rationale in enumerate(rationales, start=1):
        print(f"  {index}. {rationale['text']} (score={rationale['score']:.4f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run XAI explanations for the trained Stress Header model."
    )
    parser.add_argument(
        "--text",
        default="I can't handle my exam deadlines anymore",
        help="Text sample to explain.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        choices=[0, 1],
        default=None,
        help="Optional class to explain: 0=Not Stressed, 1=Stressed. Defaults to prediction.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help=(
            "Skip standalone SHAP. The combined method still uses SHAP when "
            "explicitly enabled."
        ),
    )
    parser.add_argument(
        "--with-combined-xai",
        action="store_true",
        help="Combine normalized SHAP, LIME, and Integrated Gradients evidence.",
    )
    parser.add_argument(
        "--with-lime",
        action="store_true",
        help="Also compute a word-level LIME local surrogate explanation.",
    )
    parser.add_argument(
        "--with-counterfactuals",
        action="store_true",
        help="Also search for minimal class-flipping text edits.",
    )
    parser.add_argument(
        "--lime-samples",
        type=int,
        default=1000,
        help="Number of perturbed texts used to fit the LIME surrogate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ig_dir = OUTPUT_DIR / "ig_plots"
    shap_dir = OUTPUT_DIR / "shap_plots"
    lime_dir = OUTPUT_DIR / "lime_plots"
    combined_dir = OUTPUT_DIR / "combined_xai"
    ig_dir.mkdir(parents=True, exist_ok=True)
    shap_dir.mkdir(parents=True, exist_ok=True)
    if args.with_combined_xai:
        combined_dir.mkdir(parents=True, exist_ok=True)
    if args.with_lime:
        lime_dir.mkdir(parents=True, exist_ok=True)

    predictor = StressPredictor()
    predictor.load()

    model = predictor._model
    tokenizer = predictor._tokenizer
    labels = predictor._labels
    max_len = predictor._max_len

    if model is None or tokenizer is None:
        raise RuntimeError("StressPredictor did not load the model/tokenizer.")

    prepared_text = preprocess_for_transformers(args.text)
    report = {
        "original_text": args.text,
        "preprocessed_text": prepared_text,
        "results": [],
    }

    if not args.with_combined_xai:
        ig = StressIG(model, tokenizer, labels, max_length=max_len)
        ig_result = ig.explain(prepared_text, target_class=args.target_class)
        ig_path = ig_dir / "ig_sample.png"
        ig.save_plot(ig_result, ig_path)

        print(f"IG prediction: {ig_result['predicted_label']}")
        print(f"IG confidence: {ig_result['confidence']:.4f}")
        print(f"IG plot: {ig_path}")
        print_rationales("IG", ig_result)
        report["results"].append(
            serialise_result("Integrated Gradients", ig_result, ig_path)
        )

    if not args.skip_shap and not args.with_combined_xai:
        shap_xai = StressSHAP(model, tokenizer, labels, max_length=max_len)
        shap_result = shap_xai.explain(
            prepared_text,
            target_class=args.target_class,
        )
        shap_png = shap_dir / "shap_sample.png"
        shap_html = shap_dir / "shap_sample.html"
        shap_xai.save_bar_plot(shap_result, shap_png)
        shap_xai.save_text_plot(shap_result, shap_html)

        print(f"SHAP prediction: {shap_result['predicted_label']}")
        print(f"SHAP confidence: {shap_result['confidence']:.4f}")
        print(f"SHAP plot: {shap_png}")
        print(f"SHAP HTML: {shap_html}")
        print_rationales("SHAP", shap_result)
        report["results"].append(serialise_result("SHAP", shap_result, shap_png))

    if args.with_lime:
        lime_xai = StressLIME(model, tokenizer, labels, max_length=max_len)
        lime_result = lime_xai.explain(
            prepared_text,
            target_class=args.target_class,
            num_samples=args.lime_samples,
        )
        lime_png = lime_dir / "lime_sample.png"
        lime_html = lime_dir / "lime_sample.html"
        lime_xai.save_plot(lime_result, lime_png)
        lime_xai.save_html(lime_result, lime_html)

        print(f"LIME prediction: {lime_result['predicted_label']}")
        print(
            f"LIME explained class: {lime_result['explained_label']} "
            f"(probability={lime_result['explained_probability']:.4f})"
        )
        print(f"LIME local R²: {lime_result['local_r2']:.4f}")
        print(f"LIME plot: {lime_png}")
        print(f"LIME HTML: {lime_html}")
        print_rationales("LIME", lime_result)
        lime_serialised = serialise_result("LIME", lime_result, lime_png)
        lime_serialised["explained_class"] = lime_result["explained_label"]
        lime_serialised["explained_probability"] = lime_result[
            "explained_probability"
        ]
        lime_serialised["local_r2"] = lime_result["local_r2"]
        lime_serialised["html_path"] = str(lime_html)
        report["results"].append(lime_serialised)

    if args.with_combined_xai:
        combined_xai = StressCombinedXAI(
            model,
            tokenizer,
            labels,
            max_length=max_len,
        )
        combined_result = combined_xai.explain(
            prepared_text,
            target_class=args.target_class,
            lime_samples=args.lime_samples,
        )
        combined_path = combined_dir / "combined_sample.png"
        combined_xai.save_plot(combined_result, combined_path)

        print(
            f"Combined XAI prediction: {combined_result['predicted_label']} "
            f"({combined_result['confidence']:.4f})"
        )
        print(f"Combined XAI explained class: {combined_result['explained_label']}")
        print(f"Methods used: {', '.join(combined_result['methods_used'])}")
        print(f"Combined XAI plot: {combined_path}")
        combined_serialised = serialise_result(
            "Combined SHAP + LIME + Integrated Gradients",
            combined_result,
            combined_path,
        )
        combined_serialised["methods_used"] = combined_result["methods_used"]
        combined_serialised["ranked_features"] = combined_result["ranked_features"]
        report["results"].append(combined_serialised)

    if args.with_counterfactuals:
        counterfactual_xai = StressCounterfactual(
            model,
            tokenizer,
            labels,
            max_length=max_len,
        )
        counterfactual_result = counterfactual_xai.explain(prepared_text)
        best = counterfactual_result["best_counterfactual"]
        if counterfactual_result["counterfactual_found"] and best:
            print(f"Counterfactual: {best['text']}")
            print(
                f"Counterfactual prediction: {best['predicted_label']} "
                f"({best['confidence']:.4f})"
            )
        elif best:
            print(f"No flip found; closest counterfactual candidate: {best['text']}")
        else:
            print("No counterfactual candidate could be generated.")
        report["results"].append(
            {
                "method": "Counterfactual explanations",
                "prediction": counterfactual_result["predicted_label"],
                "confidence": counterfactual_result["confidence"],
                "target": counterfactual_result["target_label"],
                "counterfactual_found": counterfactual_result[
                    "counterfactual_found"
                ],
                "counterfactuals": counterfactual_result["counterfactuals"],
                "disclaimer": counterfactual_result["disclaimer"],
            }
        )

    report_path = OUTPUT_DIR / "rationale_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Rationale report: {report_path}")


if __name__ == "__main__":
    main()
