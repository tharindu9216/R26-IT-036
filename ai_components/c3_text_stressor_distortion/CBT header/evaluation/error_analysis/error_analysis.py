"""CBT binary head (No Distortion vs Distortion) error analysis on the
held-out cbt_test.csv split.

Mirrors ai_components/c3_text_stressor_distortion/Stress header/evaluation/
error_analysis/error_analysis.ipynb for the CBT header. Where the Stress
header breaks errors down by subreddit, this breaks them down by the
`Dominant Distortion` type since the deployed CBT models are binary-only
(no distortion-type head).

Run from the repository root:
    python3 "ai_components/c3_text_stressor_distortion/CBT header/evaluation/error_analysis/error_analysis.py"
"""
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path("/mnt/storage/SLIIT/research/R26-IT-036")
DATA_DIR = BASE_DIR / "data/CBT header/processed"
MODELS_DIR = BASE_DIR / "models/c3_text_stressor_distortion/CBT header"
OUTPUT_DIR = BASE_DIR / "reports/c3_text_stressor_distortion/CBT header/evaluation/error_analysis"
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

XAI_DIR = BASE_DIR / "ai_components/c3_text_stressor_distortion/CBT header/xai"
TRAIN_DIR = BASE_DIR / "ai_components/c3_text_stressor_distortion/CBT header/train"
sys.path.insert(0, str(XAI_DIR))
sys.path.insert(0, str(TRAIN_DIR))
from model_loader import load_binary_config, load_cbt_xai_resources  # noqa: E402
from utils import save_results  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORMER_NAMES = ["BERT", "MentalBERT", "DeBERTa-v3"]
BASELINE_NAMES = ["LR", "SVM"]

print(" Imports loaded")
print(f" Data dir   : {DATA_DIR}")
print(f" Models dir : {MODELS_DIR}")
print(f" Output dir : {OUTPUT_DIR}")

config = load_binary_config()

test_df = pd.read_csv(DATA_DIR / "cbt_test.csv")
test_df["binary_label"] = (test_df["label"].astype(int) != 0).astype(int)
print(f" Test set shape: {test_df.shape}")
print("\nLabel distribution:")
print(test_df["binary_label"].value_counts().rename("count"))
print("\nDominant Distortion distribution:")
print(test_df["Dominant Distortion"].value_counts().rename("count"))


def classify_error_type(true_label, pred_label):
    if true_label == 1 and pred_label == 0:
        return "false_negative"
    if true_label == 0 and pred_label == 1:
        return "false_positive"
    return "correct"


def compute_binary_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["roc_auc"] = np.nan
    return metrics


def evaluate_transformer(model_name, df, batch_size=16):
    resources = load_cbt_xai_resources(model_name=model_name, device=DEVICE)
    model = resources.model
    tokenizer = resources.tokenizer
    max_length = resources.max_length
    threshold = resources.decision_threshold
    text_col = config["models"][model_name]["training_text_column"]

    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start + batch_size]
            texts = batch[text_col].fillna("").tolist()
            encoded = tokenizer(
                texts, max_length=max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(DEVICE)
            attention_mask = encoded["attention_mask"].to(DEVICE)
            logits, _ = model(input_ids, attention_mask)
            probs_all = torch.softmax(logits, dim=1)
            prob_pos = probs_all[:, 1].cpu().numpy()
            conf = probs_all.max(dim=1).values.cpu().numpy()

            for i, idx in enumerate(batch.index):
                row = df.loc[idx].to_dict()
                pred = int(prob_pos[i] >= threshold)
                row.update({
                    "model_pred": pred,
                    "model_prob": float(prob_pos[i]),
                    "model_confidence": float(conf[i]),
                })
                rows.append(row)

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    pred_df = pd.DataFrame(rows)
    pred_df["error_type"] = pred_df.apply(
        lambda r: classify_error_type(r["binary_label"], r["model_pred"]), axis=1)
    pred_df["is_correct"] = pred_df["error_type"] == "correct"
    pred_df["is_error"] = ~pred_df["is_correct"]
    pred_df["error_margin"] = np.abs(pred_df["model_prob"] - threshold) * 2
    return pred_df, threshold


def evaluate_baseline(model_name, df):
    path = MODELS_DIR / f"binary_baseline_{model_name}.pkl"
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    threshold = bundle.get("binary_threshold", 0.5)
    text_col = bundle.get("input_text_column", "Patient Question")

    X = vectorizer.transform(df[text_col].fillna("").tolist())
    raw_proba = np.asarray(model.predict_proba(X), dtype=float)
    probs = np.zeros(len(df), dtype=float)
    for source_col, class_id in enumerate(model.classes_):
        if int(class_id) == 1:
            probs = raw_proba[:, source_col]
    conf = raw_proba.max(axis=1)
    preds = (probs >= threshold).astype(int)

    pred_df = df.copy()
    pred_df["model_pred"] = preds
    pred_df["model_prob"] = probs
    pred_df["model_confidence"] = conf
    pred_df["error_type"] = pred_df.apply(
        lambda r: classify_error_type(r["binary_label"], r["model_pred"]), axis=1)
    pred_df["is_correct"] = pred_df["error_type"] == "correct"
    pred_df["is_error"] = ~pred_df["is_correct"]
    pred_df["error_margin"] = np.abs(pred_df["model_prob"] - threshold) * 2
    return pred_df, threshold


def summarize_errors(pred_df):
    total = len(pred_df)
    errors = int(pred_df["is_error"].sum())
    fp = int((pred_df["error_type"] == "false_positive").sum())
    fn = int((pred_df["error_type"] == "false_negative").sum())
    y_true = pred_df["binary_label"].tolist()
    y_pred = pred_df["model_pred"].tolist()
    y_prob = pred_df["model_prob"].tolist()
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    metrics.update({
        "total_samples": total,
        "errors": errors,
        "error_rate": errors / total if total else 0.0,
        "false_positives": fp,
        "false_negatives": fn,
        "mean_wrong_confidence": float(pred_df.loc[pred_df["is_error"], "model_confidence"].mean()) if errors else 0.0,
        "mean_correct_confidence": float(pred_df.loc[pred_df["is_correct"], "model_confidence"].mean()) if total - errors else 0.0,
    })
    return metrics


analysis_results = {}
all_pred_frames = {}
thresholds = {}

for name in TRANSFORMER_NAMES:
    print("\n" + "=" * 80)
    print(f"Evaluating {name} [transformer]")
    print("=" * 80)
    pred_df, threshold = evaluate_transformer(name, test_df)
    metrics = summarize_errors(pred_df)
    thresholds[name] = threshold
    analysis_results[name] = {
        "model_name": name, "model_type": "transformer", "threshold": threshold,
        "metrics": metrics, "error_counts": pred_df["error_type"].value_counts().to_dict(),
    }
    all_pred_frames[name] = pred_df
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"F1-Score   : {metrics['f1']:.4f}")
    print(f"MCC        : {metrics['mcc']:.4f}")
    if not np.isnan(metrics.get("roc_auc", np.nan)):
        print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"Errors     : {metrics['errors']} / {metrics['total_samples']} ({metrics['error_rate']:.2%})")
    print(f"FP / FN    : {metrics['false_positives']} / {metrics['false_negatives']}")

for name in BASELINE_NAMES:
    print("\n" + "=" * 80)
    print(f"Evaluating {name} [baseline]")
    print("=" * 80)
    pred_df, threshold = evaluate_baseline(name, test_df)
    metrics = summarize_errors(pred_df)
    thresholds[name] = threshold
    analysis_results[name] = {
        "model_name": name, "model_type": "baseline", "threshold": threshold,
        "metrics": metrics, "error_counts": pred_df["error_type"].value_counts().to_dict(),
    }
    all_pred_frames[name] = pred_df
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"F1-Score   : {metrics['f1']:.4f}")
    print(f"MCC        : {metrics['mcc']:.4f}")
    if not np.isnan(metrics.get("roc_auc", np.nan)):
        print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"Errors     : {metrics['errors']} / {metrics['total_samples']} ({metrics['error_rate']:.2%})")
    print(f"FP / FN    : {metrics['false_positives']} / {metrics['false_negatives']}")

combined_df = test_df.copy().reset_index(drop=True)
for name, pred_df in all_pred_frames.items():
    pred_df = pred_df.reset_index(drop=True)
    combined_df[f"{name}_pred"] = pred_df["model_pred"]
    combined_df[f"{name}_prob"] = pred_df["model_prob"]
    combined_df[f"{name}_conf"] = pred_df["model_confidence"]
    combined_df[f"{name}_error"] = pred_df["is_error"]
    combined_df[f"{name}_error_type"] = pred_df["error_type"]

print("\n Combined prediction frame created")
print(" Shape:", combined_df.shape)

save_results(analysis_results, OUTPUT_DIR / "error_analysis_results.json")
combined_df.to_csv(OUTPUT_DIR / "combined_predictions.csv", index=False)
print(" Saved: error_analysis_results.json")
print(" Saved: combined_predictions.csv")

# ===== Summary tables =====
summary_rows = []
for name, result in analysis_results.items():
    m = result["metrics"]
    summary_rows.append({
        "Model": name, "Type": result["model_type"], "Accuracy": m["accuracy"],
        "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
        "MCC": m["mcc"], "ROC-AUC": m.get("roc_auc", np.nan),
        "Errors": m["errors"], "Error Rate": m["error_rate"],
        "FP": m["false_positives"], "FN": m["false_negatives"],
        "Wrong Confidence": m["mean_wrong_confidence"],
    })
summary_df = pd.DataFrame(summary_rows).sort_values("F1", ascending=False).reset_index(drop=True)
print("\n" + "=" * 100)
print("ERROR ANALYSIS SUMMARY (CBT DISTORTION DETECTION)")
print("=" * 100)
print(summary_df.to_string(index=False))
summary_df.to_csv(OUTPUT_DIR / "error_summary.csv", index=False)
print("\n Saved: error_summary.csv")

# Distortion-type breakdown (CBT analog of the Stress header's subreddit breakdown)
distortion_rows = []
for name, pred_df in all_pred_frames.items():
    for distortion_type, group in pred_df.groupby("Dominant Distortion"):
        distortion_rows.append({
            "Model": name,
            "Dominant Distortion": distortion_type,
            "Support": len(group),
            "Error Rate": float(group["is_error"].mean()),
            "Errors": int(group["is_error"].sum()),
        })
distortion_df = pd.DataFrame(distortion_rows)
distortion_df.to_csv(OUTPUT_DIR / "distortion_type_error_summary.csv", index=False)
print(" Saved: distortion_type_error_summary.csv")

# ===== Visualizations =====
# 1. Error rate comparison
fig, ax = plt.subplots(figsize=(12, 6))
plot_df = summary_df.sort_values("Error Rate", ascending=False)
colors = ["#d62728" if t == "baseline" else "#1f77b4" for t in plot_df["Type"]]
bars = ax.bar(plot_df["Model"], plot_df["Error Rate"], color=colors, edgecolor="black", alpha=0.85)
ax.set_title("Model Error Rate Comparison", fontsize=14, fontweight="bold")
ax.set_ylabel("Error Rate")
ax.set_xlabel("Model")
ax.set_ylim(0, max(plot_df["Error Rate"].max() * 1.2, 0.05))
ax.tick_params(axis="x", rotation=25)
for bar, val in zip(bars, plot_df["Error Rate"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.2%}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "1_error_rate_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 2. FP vs FN
fig, ax = plt.subplots(figsize=(12, 6))
fp = summary_df["FP"].values
fn = summary_df["FN"].values
x = np.arange(len(summary_df))
width = 0.35
ax.bar(x - width / 2, fp, width, label="False Positives", color="#ff7f0e")
ax.bar(x + width / 2, fn, width, label="False Negatives", color="#2ca02c")
ax.set_xticks(x)
ax.set_xticklabels(summary_df["Model"], rotation=25)
ax.set_title("False Positives and False Negatives by Model", fontsize=14, fontweight="bold")
ax.set_ylabel("Count")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "2_fp_fn_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 3. Confusion matrices
n_models = len(all_pred_frames)
ncols = min(3, n_models)
nrows = (n_models + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows))
axes = np.array(axes).reshape(-1)
for i, (name, pred_df) in enumerate(all_pred_frames.items()):
    cm = confusion_matrix(pred_df["binary_label"], pred_df["model_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i],
                xticklabels=["No Distortion", "Distortion"],
                yticklabels=["No Distortion", "Distortion"])
    axes[i].set_title(f"{name} | F1={analysis_results[name]['metrics']['f1']:.4f}")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("True")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Confusion Matrices by Model", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "3_confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 4. Confidence distributions
fig, axes = plt.subplots(len(all_pred_frames), 1, figsize=(12, 4 * len(all_pred_frames)), sharex=True)
if len(all_pred_frames) == 1:
    axes = [axes]
for ax, (name, pred_df) in zip(axes, all_pred_frames.items()):
    correct_conf = pred_df.loc[pred_df["is_correct"], "model_confidence"]
    wrong_conf = pred_df.loc[pred_df["is_error"], "model_confidence"]
    if len(correct_conf):
        sns.kdeplot(correct_conf, fill=True, ax=ax, label="Correct", color="#2ca02c", alpha=0.35)
    if len(wrong_conf):
        sns.kdeplot(wrong_conf, fill=True, ax=ax, label="Wrong", color="#d62728", alpha=0.35)
    ax.set_title(f"Prediction Confidence Distribution — {name}")
    ax.set_xlabel("Model Confidence")
    ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "4_confidence_distributions.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 5. Error agreement matrix
model_names = list(all_pred_frames.keys())
overlap = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
for m1 in model_names:
    e1 = all_pred_frames[m1]["is_error"].astype(int).values
    for m2 in model_names:
        e2 = all_pred_frames[m2]["is_error"].astype(int).values
        overlap.loc[m1, m2] = np.mean(e1 == e2)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(overlap, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
ax.set_title("Error Agreement Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "5_error_agreement_matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6. Error rate by distortion type (CBT-specific, replaces subreddit plot)
fig, ax = plt.subplots(figsize=(14, 7))
pivot = distortion_df.pivot(index="Dominant Distortion", columns="Model", values="Error Rate")
pivot = pivot.reindex(columns=list(all_pred_frames.keys()))
pivot.plot(kind="bar", ax=ax, width=0.85)
ax.set_title("Error Rate by Dominant Distortion Type", fontsize=14, fontweight="bold")
ax.set_ylabel("Error Rate")
ax.set_xlabel("Dominant Distortion")
ax.legend(title="Model", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "6_error_rate_by_distortion_type.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(" Saved visualization files to:", PLOTS_DIR)

# ===== Hardest / easiest samples =====
error_cols = [f"{name}_error" for name in all_pred_frames]
combined_df["all_models_wrong"] = combined_df[error_cols].all(axis=1)
combined_df["all_models_correct"] = ~combined_df[error_cols].any(axis=1)
combined_df["wrong_model_count"] = combined_df[error_cols].sum(axis=1)
combined_df.to_csv(OUTPUT_DIR / "agreement_analysis.csv", index=False)
print(" Saved: agreement_analysis.csv")

# ===== Final report =====
report_lines = []
report_lines.append("ERROR ANALYSIS REPORT")
report_lines.append("=" * 80)
report_lines.append(f"Test samples: {len(test_df)}")
report_lines.append("")
report_lines.append("MODEL SUMMARY")
report_lines.append("-" * 80)
for _, row in summary_df.iterrows():
    report_lines.append(
        f"{row['Model']:12s} | {row['Type']:11s} | "
        f"Acc={row['Accuracy']:.4f} | F1={row['F1']:.4f} | "
        f"Err={row['Error Rate']:.2%} | FP={int(row['FP'])} | FN={int(row['FN'])}"
    )
report_lines.append("")
report_lines.append("DISTORTION-TYPE ERROR SUMMARY (mean error rate across models)")
report_lines.append("-" * 80)
mean_by_type = distortion_df.groupby("Dominant Distortion").agg(
    Support=("Support", "first"), Mean_Error_Rate=("Error Rate", "mean")
).sort_values("Mean_Error_Rate", ascending=False)
for distortion_type, row in mean_by_type.iterrows():
    report_lines.append(f"{distortion_type:26s} | n={int(row['Support']):3d} | mean error rate={row['Mean_Error_Rate']:.2%}")
report_lines.append("")
report_lines.append("HARD MODEL AGREEMENT STATS")
report_lines.append("-" * 80)
report_lines.append(f"All models correct: {int(combined_df['all_models_correct'].sum())}")
report_lines.append(f"All models wrong   : {int(combined_df['all_models_wrong'].sum())}")
report_lines.append(f"Avg wrong-model count per sample: {combined_df['wrong_model_count'].mean():.2f}")

report_path = OUTPUT_DIR / "error_analysis_report.txt"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f" Saved: {report_path}")
print("\nDone.")
