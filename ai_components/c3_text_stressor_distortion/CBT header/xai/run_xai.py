"""Run local explanations for trained CBT binary transformer models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


XAI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = XAI_DIR.parents[3]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "c3_text_stressor_distortion"
    / "CBT header"
    / "evaluation"
    / "xai_outputs"
)
if str(XAI_DIR) not in sys.path:
    sys.path.insert(0, str(XAI_DIR))

from model_loader import load_cbt_xai_resources  # noqa: E402
from preprocessing import preprocess_for_model  # noqa: E402
from shared_explainers import (  # noqa: E402
    CBTCombinedXAI,
    CBTCounterfactual,
    CBTIG,
    CBTLIME,
    CBTSHAP,
)


MODEL_CHOICES = ("BERT", "MentalBERT", "DeBERTa-v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explain a trained CBT Distortion vs No Distortion transformer."
        )
    )
    parser.add_argument(
        "--text",
        default=(
            "If I make one mistake, everyone will think I am a complete failure."
        ),
        help="Text sample to explain.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="DeBERTa-v3",
        help="Trained CBT transformer checkpoint to explain.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        choices=(0, 1),
        default=None,
        help=(
            "Optional class to explain: 0=No Distortion, 1=Distortion. "
            "Defaults to the thresholded model prediction."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Optional device override.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip standalone SHAP.",
    )
    parser.add_argument(
        "--with-lime",
        action="store_true",
        help="Also compute a word-level LIME local surrogate.",
    )
    parser.add_argument(
        "--with-combined-xai",
        action="store_true",
        help="Run combined SHAP + LIME + Integrated Gradients.",
    )
    parser.add_argument(
        "--with-counterfactuals",
        action="store_true",
        help="Search for minimal edits that change the binary decision.",
    )
    parser.add_argument(
        "--lime-samples",
        type=int,
        default=1000,
        help="Perturbed texts used by LIME (default: 1000).",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=50,
        help="Integrated Gradients approximation steps (default: 50).",
    )
    return parser.parse_args()


def serialise_result(
    method: str,
    result: dict,
    plot_path: Path | None = None,
) -> dict:
    return {
        "method": method,
        "prediction": result["predicted_label"],
        "confidence": result["confidence"],
        "explained_class": result.get("explained_label"),
        "explained_probability": result.get("explained_probability"),
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
        print(
            f"  {index}. {rationale['text']} "
            f"(score={rationale['score']:.4f})"
        )


def common_kwargs(resources) -> dict:
    return {
        "label_names": resources.labels,
        "device": str(resources.device),
        "max_length": resources.max_length,
        "decision_threshold": resources.decision_threshold,
    }


def main() -> None:
    args = parse_args()
    if not args.text.strip():
        raise ValueError("--text must not be empty.")
    if args.lime_samples < 2:
        raise ValueError("--lime-samples must be at least 2.")
    if args.ig_steps < 2:
        raise ValueError("--ig-steps must be at least 2.")

    resources = load_cbt_xai_resources(args.model, device=args.device)
    prepared_text = preprocess_for_model(args.text, args.model)
    if not prepared_text:
        raise ValueError("Text was empty after CBT preprocessing.")

    model_slug = args.model.lower().replace("-", "_")
    run_dir = OUTPUT_DIR / model_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_slug}_{int(time.time())}"
    common = common_kwargs(resources)
    report = {
        "task": "CBT binary cognitive-distortion detection",
        "model": args.model,
        "checkpoint": str(resources.checkpoint_path),
        "decision_threshold": resources.decision_threshold,
        "original_text": args.text,
        "preprocessed_text": prepared_text,
        "results": [],
    }

    print(f"Model: {args.model}")
    print(f"Checkpoint: {resources.checkpoint_path}")
    print(f"Decision threshold: {resources.decision_threshold:.2f}")
    print(f"Device: {resources.device}")

    if not args.with_combined_xai:
        ig = CBTIG(resources.model, resources.tokenizer, **common)
        ig_result = ig.explain(
            prepared_text,
            target_class=args.target_class,
            n_steps=args.ig_steps,
        )
        ig_path = run_dir / f"{stem}_ig.png"
        ig.save_plot(ig_result, ig_path)
        print(
            f"IG: predicted {ig_result['predicted_label']} "
            f"({ig_result['confidence']:.4f}); explaining "
            f"{ig_result['explained_label']}"
        )
        print_rationales("IG", ig_result)
        report["results"].append(
            serialise_result("Integrated Gradients", ig_result, ig_path)
        )

    if not args.skip_shap and not args.with_combined_xai:
        shap_xai = CBTSHAP(resources.model, resources.tokenizer, **common)
        shap_result = shap_xai.explain(
            prepared_text,
            target_class=args.target_class,
        )
        shap_png = run_dir / f"{stem}_shap.png"
        shap_html = run_dir / f"{stem}_shap.html"
        shap_xai.save_bar_plot(shap_result, shap_png)
        shap_xai.save_text_plot(shap_result, shap_html)
        print(
            f"SHAP: predicted {shap_result['predicted_label']} "
            f"({shap_result['confidence']:.4f}); explaining "
            f"{shap_result['explained_label']}"
        )
        print_rationales("SHAP", shap_result)
        item = serialise_result("SHAP", shap_result, shap_png)
        item["html_path"] = str(shap_html)
        report["results"].append(item)

    if args.with_lime:
        lime_xai = CBTLIME(resources.model, resources.tokenizer, **common)
        lime_result = lime_xai.explain(
            prepared_text,
            target_class=args.target_class,
            num_samples=args.lime_samples,
        )
        lime_png = run_dir / f"{stem}_lime.png"
        lime_html = run_dir / f"{stem}_lime.html"
        lime_xai.save_plot(lime_result, lime_png)
        lime_xai.save_html(lime_result, lime_html)
        print(
            f"LIME: predicted {lime_result['predicted_label']} "
            f"({lime_result['confidence']:.4f}); local R²="
            f"{lime_result['local_r2']:.4f}"
        )
        print_rationales("LIME", lime_result)
        item = serialise_result("LIME", lime_result, lime_png)
        item["local_r2"] = lime_result["local_r2"]
        item["html_path"] = str(lime_html)
        report["results"].append(item)

    if args.with_combined_xai:
        combined_xai = CBTCombinedXAI(
            resources.model,
            resources.tokenizer,
            **common,
        )
        combined_result = combined_xai.explain(
            prepared_text,
            target_class=args.target_class,
            lime_samples=args.lime_samples,
            ig_steps=args.ig_steps,
        )
        combined_path = run_dir / f"{stem}_combined_xai.png"
        combined_xai.save_plot(combined_result, combined_path)
        print(
            f"Combined XAI: predicted {combined_result['predicted_label']} "
            f"({combined_result['confidence']:.4f}); explaining "
            f"{combined_result['explained_label']}"
        )
        item = serialise_result(
            "Combined SHAP + LIME + Integrated Gradients",
            combined_result,
            combined_path,
        )
        item["methods_used"] = combined_result["methods_used"]
        item["ranked_features"] = combined_result["ranked_features"]
        report["results"].append(item)

    if args.with_counterfactuals:
        counterfactual_xai = CBTCounterfactual(
            resources.model,
            resources.tokenizer,
            **common,
        )
        counterfactual_result = counterfactual_xai.explain(prepared_text)
        best = counterfactual_result["best_counterfactual"]
        if counterfactual_result["counterfactual_found"] and best:
            print(
                f"Counterfactual: {best['text']} -> "
                f"{best['predicted_label']} ({best['confidence']:.4f})"
            )
        elif best:
            print(f"No flip found; closest candidate: {best['text']}")
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

    report_path = run_dir / f"{stem}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
