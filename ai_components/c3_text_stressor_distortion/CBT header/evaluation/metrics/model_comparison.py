"""CBT binary head (No Distortion vs Distortion) model comparison on the
held-out cbt_test.csv split.

Mirrors ai_components/c3_text_stressor_distortion/Stress header/evaluation/
metrics/model_comparism.ipynb for the CBT header. Loads each transformer
through the canonical CBT XAI loader (ai_components/.../CBT header/xai/
model_loader.py) so architecture, dropout, MentalBERT runtime scaffold, and
per-model decision thresholds always match binary_config.json exactly, and
evaluates the two TF-IDF baselines from their saved pickles.

Run from the repository root:
    python3 "ai_components/c3_text_stressor_distortion/CBT header/evaluation/metrics/model_comparison.py"
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
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path("/mnt/storage/SLIIT/research/R26-IT-036")
DATA_DIR = BASE_DIR / "data/CBT header/processed"
MODELS_DIR = BASE_DIR / "models/c3_text_stressor_distortion/CBT header"
RESULTS_DIR = BASE_DIR / "reports/c3_text_stressor_distortion/CBT header/evaluation/metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

XAI_DIR = BASE_DIR / "ai_components/c3_text_stressor_distortion/CBT header/xai"
sys.path.insert(0, str(XAI_DIR))
from model_loader import load_binary_config, load_cbt_xai_resources  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORMER_NAMES = ["BERT", "MentalBERT", "DeBERTa-v3"]
BASELINE_NAMES = ["LR", "SVM"]

print(f" Base dir: {BASE_DIR}")
print(f" Data dir: {DATA_DIR}")
print(f" Models dir: {MODELS_DIR}")
print(f" Results dir: {RESULTS_DIR}")
print(f" Device: {DEVICE}")

config = load_binary_config()

test_df = pd.read_csv(DATA_DIR / "cbt_test.csv")
test_df["binary_label"] = (test_df["label"].astype(int) != 0).astype(int)
print(f"\nLoaded test_df: {len(test_df)} rows from {DATA_DIR / 'cbt_test.csv'}")
print(test_df["binary_label"].value_counts().rename("count"))


def compute_all_metrics(preds, labels, probs=None):
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
    }
    if probs is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(labels, probs)
        except Exception:
            metrics["roc_auc"] = np.nan
    return metrics


def evaluate_transformer(model_name, df, batch_size=16):
    print(f"\n→ Evaluating {model_name}...")
    resources = load_cbt_xai_resources(model_name=model_name, device=DEVICE)
    model = resources.model
    tokenizer = resources.tokenizer
    max_length = resources.max_length
    threshold = resources.decision_threshold
    text_col = config["models"][model_name]["training_text_column"]

    probs = []
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
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    probs = np.array(probs)
    preds = (probs >= threshold).astype(int)
    print(f"   Threshold={threshold} | Acc={accuracy_score(df['binary_label'], preds):.4f}")
    return probs, preds, threshold, text_col


def evaluate_baseline(model_name, df):
    print(f"\n→ Evaluating {model_name}...")
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
    preds = (probs >= threshold).astype(int)
    print(f"   Threshold={threshold} | Acc={accuracy_score(df['binary_label'], preds):.4f}")
    return probs, preds, threshold, text_col


all_results = {}

for name in TRANSFORMER_NAMES:
    probs, preds, threshold, text_col = evaluate_transformer(name, test_df)
    metrics = compute_all_metrics(preds, test_df["binary_label"].values, probs)
    all_results[name] = {
        "model": name, "type": "Transformer", "threshold": threshold,
        "metrics": metrics, "preds": preds, "probs": probs,
        "labels": test_df["binary_label"].values,
    }

for name in BASELINE_NAMES:
    probs, preds, threshold, text_col = evaluate_baseline(name, test_df)
    metrics = compute_all_metrics(preds, test_df["binary_label"].values, probs)
    all_results[name] = {
        "model": name, "type": "Baseline", "threshold": threshold,
        "metrics": metrics, "preds": preds, "probs": probs,
        "labels": test_df["binary_label"].values,
    }

# ===== Comparison table =====
metrics_list = []
for name, result in all_results.items():
    row = result["metrics"].copy()
    row["Model"] = name
    row["Type"] = result["type"]
    metrics_list.append(row)

comparison_df = pd.DataFrame(metrics_list)
col_order = ["Model", "Type", "accuracy", "precision", "recall", "f1", "mcc", "roc_auc"]
comparison_df = comparison_df[[c for c in col_order if c in comparison_df.columns]]
comparison_df = comparison_df.rename(columns={
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1-Score", "mcc": "MCC", "roc_auc": "ROC-AUC",
})

print("\n" + "=" * 70)
print("CBT DISTORTION DETECTION METRICS COMPARISON")
print("=" * 70)
print(comparison_df.to_string(index=False))

comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
print(f"\n Saved: {RESULTS_DIR / 'model_comparison.csv'}")

# ===== Detailed classification reports =====
detailed_reports = {}
for name, result in all_results.items():
    report = classification_report(
        result["labels"], result["preds"],
        target_names=["No Distortion", "Distortion"], digits=4,
    )
    detailed_reports[name] = report

with open(RESULTS_DIR / "detailed_classification_reports.txt", "w") as f:
    for name, report in detailed_reports.items():
        f.write(f"\n{'=' * 70}\n")
        f.write(f"Model: {name}\n")
        f.write(f"{'=' * 70}\n")
        f.write(report)
        f.write("\n")
print(f" Saved: {RESULTS_DIR / 'detailed_classification_reports.txt'}")

# ===== All results JSON =====
all_results_json = {
    name: {"type": result["type"], "threshold": result["threshold"],
           "distortion_detection_metrics": result["metrics"]}
    for name, result in all_results.items()
}
with open(RESULTS_DIR / "all_evaluation_results.json", "w") as f:
    json.dump(all_results_json, f, indent=2, default=str)
print(f" Saved: {RESULTS_DIR / 'all_evaluation_results.json'}")

# ===== Plots =====
# 1. Metrics comparison bar chart
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
plot_df = comparison_df[["Model"] + metrics_to_plot].set_index("Model")
fig, ax = plt.subplots(figsize=(14, 6))
plot_df.plot(kind="bar", ax=ax, width=0.8)
ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_xlabel("Model", fontsize=12, fontweight="bold")
ax.set_title("Model Performance Comparison - CBT Distortion Detection", fontsize=14, fontweight="bold", pad=20)
ax.legend(title="Metrics", fontsize=10, title_fontsize=11)
ax.set_ylim([0, 1.05])
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=1.0, color="red", linestyle="--", linewidth=2, alpha=0.3)
plt.xticks(rotation=45, ha="right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "1_metrics_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(" Saved: 1_metrics_comparison.png")

# 2. Confusion matrices
n_models = len(all_results)
ncols = min(3, n_models)
nrows = (n_models + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
axes = np.array(axes).reshape(-1)
for idx, (name, result) in enumerate(all_results.items()):
    cm = confusion_matrix(result["labels"], result["preds"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                xticklabels=["No Distortion", "Distortion"],
                yticklabels=["No Distortion", "Distortion"],
                cbar_kws={"label": "Count"},
                annot_kws={"fontsize": 12, "fontweight": "bold"})
    axes[idx].set_title(f"{name}\nAccuracy: {result['metrics']['accuracy']:.4f}", fontsize=11, fontweight="bold")
    axes[idx].set_ylabel("True Label", fontsize=10, fontweight="bold")
    axes[idx].set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
for idx in range(n_models, len(axes)):
    axes[idx].set_visible(False)
plt.suptitle("Confusion Matrices - CBT Distortion Detection", fontsize=14, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "2_confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(" Saved: 2_confusion_matrices.png")

# 3. ROC curves
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
for color, (name, result) in zip(colors, all_results.items()):
    try:
        fpr, tpr, _ = roc_curve(result["labels"], result["probs"])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.5, label=f"{name} (AUC = {roc_auc_val:.4f})", color=color)
    except Exception as exc:
        print(f"   Could not plot ROC for {name}: {exc}")
ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier", alpha=0.5)
ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
ax.set_title("ROC Curves - CBT Distortion Detection", fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(PLOTS_DIR / "3_roc_curves.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(" Saved: 3_roc_curves.png")

# 4. F1 & Accuracy comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
f1_data = comparison_df.sort_values("F1-Score", ascending=True)
colors_f1 = ["#ff7f0e" if t == "Baseline" else "#1f77b4" for t in f1_data["Type"]]
ax1.barh(f1_data["Model"], f1_data["F1-Score"], color=colors_f1)
ax1.set_xlabel("F1-Score", fontsize=11, fontweight="bold")
ax1.set_title("F1-Score Comparison", fontsize=12, fontweight="bold")
ax1.set_xlim([0, 1])
ax1.grid(axis="x", alpha=0.3)
for idx, (model, val) in enumerate(zip(f1_data["Model"], f1_data["F1-Score"])):
    ax1.text(val + 0.01, idx, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

acc_data = comparison_df.sort_values("Accuracy", ascending=True)
colors_acc = ["#ff7f0e" if t == "Baseline" else "#1f77b4" for t in acc_data["Type"]]
ax2.barh(acc_data["Model"], acc_data["Accuracy"], color=colors_acc)
ax2.set_xlabel("Accuracy", fontsize=11, fontweight="bold")
ax2.set_title("Accuracy Comparison", fontsize=12, fontweight="bold")
ax2.set_xlim([0, 1])
ax2.grid(axis="x", alpha=0.3)
for idx, (model, val) in enumerate(zip(acc_data["Model"], acc_data["Accuracy"])):
    ax2.text(val + 0.01, idx, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#1f77b4", label="Transformer"), Patch(facecolor="#ff7f0e", label="Baseline")]
fig.legend(handles=legend_elements, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=10)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "4_f1_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(" Saved: 4_f1_accuracy_comparison.png")

# 5. Precision vs recall
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
for idx, (name, color) in enumerate(zip(comparison_df["Model"], colors)):
    row = comparison_df[comparison_df["Model"] == name].iloc[0]
    marker = "o" if row["Type"] == "Transformer" else "s"
    size = 200 if row["Type"] == "Transformer" else 100
    ax.scatter(row["Recall"], row["Precision"], s=size, c=[color], marker=marker,
               label=name, alpha=0.7, edgecolors="black", linewidth=1.5)
    ax.annotate(name, (row["Recall"], row["Precision"]), xytext=(10, 5), textcoords="offset points", fontsize=9)
ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
ax.set_title("Precision vs Recall - CBT Distortion Detection", fontsize=14, fontweight="bold", pad=20)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="Transformer"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=10, label="Baseline"),
]
ax.legend(handles=legend_elements, loc="best", fontsize=10)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "5_precision_recall.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(" Saved: 5_precision_recall.png")

# 6. Metrics heatmap
heatmap_data = comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "MCC"]].set_index("Model")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="RdYlGn", ax=ax,
            cbar_kws={"label": "Score"}, vmin=0, vmax=1, linewidths=0.5, linecolor="gray",
            annot_kws={"fontsize": 10, "fontweight": "bold"})
ax.set_title("Metrics Heatmap - CBT Distortion Detection", fontsize=14, fontweight="bold", pad=20)
ax.set_ylabel("Model", fontsize=12, fontweight="bold")
ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "6_metrics_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(" Saved: 6_metrics_heatmap.png")

# ===== Summary =====
print("\n" + "=" * 70)
print("SUMMARY STATISTICS & KEY FINDINGS")
print("=" * 70)
metrics_to_check = ["Accuracy", "Precision", "Recall", "F1-Score", "MCC", "ROC-AUC"]
for metric in metrics_to_check:
    if metric in comparison_df.columns:
        best_idx = comparison_df[metric].idxmax()
        row = comparison_df.loc[best_idx]
        print(f"  {metric:15s}: {row['Model']:15s} ({row['Type']:12s}) = {row[metric]:.4f}")

ranked = comparison_df.sort_values("F1-Score", ascending=False).reset_index(drop=True)
print("\n MODEL RANKINGS (by F1-Score):")
for idx, row in ranked.iterrows():
    print(f"  {idx + 1:2d}. {row['Model']:15s} - F1: {row['F1-Score']:.4f}, Acc: {row['Accuracy']:.4f}")

print("\nDone.")
