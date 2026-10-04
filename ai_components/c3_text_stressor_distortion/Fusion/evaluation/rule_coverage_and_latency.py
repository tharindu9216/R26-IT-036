"""Rule-coverage and latency report for the decision-level fusion layer.

The fusion layer (backend/c3_text_stressor_distortion/app/fusion/) has no
learned parameters, so it has no accuracy/F1 to report (see
reports/c3_text_stressor_distortion/Fusion/README.md) -- there is no dataset
with both a real stress label and a real distortion label on the same text
to check a "joint prediction" against. What CAN be reported honestly:

1. Rule coverage: given a sample of real text, how often does each of the
   4 fusion states fire (no_signal / stress_only / distortion_only /
   stress_with_distortion)?
2. Latency: how long does POST /api/fusion/predict take end to end (both
   ensembles + the deterministic rule), and how long does the heavier
   on-demand /explain endpoint (SHAP + LIME + Integrated Gradients
   consensus for both heads) take on a small sample?

This calls the exact production code path (app.predictors.fusion_service /
fusion_explanation_service) in-process, so the numbers reflect the real
deployed behaviour rather than a re-implementation.

Run from the repository root:
    python3 "ai_components/c3_text_stressor_distortion/Fusion/evaluation/rule_coverage_and_latency.py"
"""
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path("/mnt/storage/SLIIT/research/R26-IT-036")
BACKEND_DIR = BASE_DIR / "backend/c3_text_stressor_distortion"
OUTPUT_DIR = BASE_DIR / "reports/c3_text_stressor_distortion/Fusion"
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))
from app.predictors import fusion_explanation_service, fusion_service  # noqa: E402

RANDOM_SEED = 42
N_PER_SOURCE = 30
N_EXPLAIN_SAMPLES = 4

random.seed(RANDOM_SEED)

stress_test = pd.read_csv(BASE_DIR / "data/Stress header/processed/dreaddit_test.csv")
cbt_test = pd.read_csv(BASE_DIR / "data/CBT header/processed/cbt_test.csv")

stress_sample = stress_test["text"].dropna().sample(
    n=N_PER_SOURCE, random_state=RANDOM_SEED).tolist()
cbt_sample = cbt_test["Patient Question"].dropna().sample(
    n=N_PER_SOURCE, random_state=RANDOM_SEED).tolist()

samples = (
    [{"source": "stress_test (Dreaddit)", "text": t} for t in stress_sample]
    + [{"source": "cbt_test (patient questions)", "text": t} for t in cbt_sample]
)
random.shuffle(samples)

print(f"Sample size: {len(samples)} ({N_PER_SOURCE} from each source)")

# ===== Warm-up (loads both ensembles; excluded from latency) =====
print("\nWarming up (loading Stress + CBT ensembles)...")
warm_start = time.perf_counter()
_ = fusion_service.analyze(samples[0]["text"])
warm_elapsed = time.perf_counter() - warm_start
print(f" Warm-up (incl. model load) took {warm_elapsed:.2f}s")

# ===== Rule coverage + predict latency =====
rows = []
predict_latencies_ms = []

print("\nRunning fusion predictions...")
for i, item in enumerate(samples):
    start = time.perf_counter()
    analysis = fusion_service.analyze(item["text"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    predict_latencies_ms.append(elapsed_ms)

    result = analysis.as_dict()
    rows.append({
        "source": item["source"],
        "text": item["text"],
        "text_preview": item["text"][:120].replace("\n", " "),
        "stress_detected": result["stress"]["detected"],
        "stress_confidence": result["stress"]["confidence"],
        "distortion_detected": result["distortion"]["detected"],
        "distortion_confidence": result["distortion"]["confidence"],
        "fusion_state": result["fusion"]["state"],
        "fusion_label": result["fusion"]["label"],
        "latency_ms": elapsed_ms,
    })
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(samples)} done")

coverage_df = pd.DataFrame(rows)
coverage_df.to_csv(OUTPUT_DIR / "rule_coverage_samples.csv", index=False)
print(f"\n Saved: {OUTPUT_DIR / 'rule_coverage_samples.csv'}")

state_counts = coverage_df["fusion_state"].value_counts()
all_states = ["no_signal", "stress_only", "distortion_only", "stress_with_distortion"]
state_dist = pd.DataFrame({
    "state": all_states,
    "count": [int(state_counts.get(s, 0)) for s in all_states],
})
state_dist["percentage"] = (state_dist["count"] / len(coverage_df) * 100).round(2)
state_dist.to_csv(OUTPUT_DIR / "state_distribution.csv", index=False)
print(f" Saved: {OUTPUT_DIR / 'state_distribution.csv'}")
print("\nSTATE DISTRIBUTION")
print(state_dist.to_string(index=False))

latency_stats = {
    "n_samples": len(predict_latencies_ms),
    "warmup_seconds_incl_model_load": warm_elapsed,
    "mean_ms": float(np.mean(predict_latencies_ms)),
    "median_ms": float(np.median(predict_latencies_ms)),
    "p95_ms": float(np.percentile(predict_latencies_ms, 95)),
    "min_ms": float(np.min(predict_latencies_ms)),
    "max_ms": float(np.max(predict_latencies_ms)),
    "std_ms": float(np.std(predict_latencies_ms)),
}
print("\nPREDICT LATENCY (/api/fusion/predict, warm)")
for k, v in latency_stats.items():
    print(f"  {k}: {v}")

# ===== Explain latency (small sample -- SHAP+LIME+IG on both heads is heavy) =====
print(f"\nRunning /explain latency on {N_EXPLAIN_SAMPLES} samples "
      f"(one per fusion state where available)...")

explain_rows = []
picked_states = set()
explain_candidates = []
for state in all_states:
    match = coverage_df[coverage_df["fusion_state"] == state]
    if len(match):
        explain_candidates.append(match.iloc[0])
while len(explain_candidates) < N_EXPLAIN_SAMPLES and len(explain_candidates) < len(coverage_df):
    extra = coverage_df.sample(n=1, random_state=RANDOM_SEED + len(explain_candidates)).iloc[0]
    explain_candidates.append(extra)
explain_candidates = explain_candidates[:N_EXPLAIN_SAMPLES]

from types import SimpleNamespace

for row in explain_candidates:
    stress_result = SimpleNamespace(is_stressed=bool(row["stress_detected"]))
    cbt_result = SimpleNamespace(has_distortion=bool(row["distortion_detected"]))
    start = time.perf_counter()
    try:
        fusion_explanation_service.explain(row["text"], stress_result, cbt_result)
        ok = True
    except Exception as exc:
        ok = False
        print(f"   explain() failed: {exc}")
    elapsed_ms = (time.perf_counter() - start) * 1000
    explain_rows.append({
        "fusion_state": row["fusion_state"],
        "latency_ms": elapsed_ms,
        "succeeded": ok,
    })
    print(f"  state={row['fusion_state']:24s} latency={elapsed_ms / 1000:.2f}s succeeded={ok}")

explain_df = pd.DataFrame(explain_rows)
explain_df.to_csv(OUTPUT_DIR / "explain_latency_samples.csv", index=False)
print(f" Saved: {OUTPUT_DIR / 'explain_latency_samples.csv'}")

explain_latency_stats = {
    "n_samples": len(explain_rows),
    "mean_ms": float(explain_df["latency_ms"].mean()) if len(explain_df) else None,
    "min_ms": float(explain_df["latency_ms"].min()) if len(explain_df) else None,
    "max_ms": float(explain_df["latency_ms"].max()) if len(explain_df) else None,
    "note": (
        f"Small sample (n={len(explain_rows)}) because SHAP + LIME + "
        "Integrated Gradients consensus runs for both heads sequentially "
        "under a lock; not representative of p95/p99 under load."
    ),
}
print("\nEXPLAIN LATENCY (/api/diary/{id}/explain, warm)")
for k, v in explain_latency_stats.items():
    print(f"  {k}: {v}")

with open(OUTPUT_DIR / "latency_report.json", "w") as f:
    json.dump({
        "predict_endpoint": latency_stats,
        "explain_endpoint": explain_latency_stats,
    }, f, indent=2)
print(f"\n Saved: {OUTPUT_DIR / 'latency_report.json'}")

# ===== Plots =====
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728"]
bars = ax.bar(state_dist["state"], state_dist["count"], color=colors, edgecolor="black", alpha=0.85)
ax.set_title(f"Fusion Rule Coverage (n={len(coverage_df)} sampled diary-style texts)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
ax.set_xlabel("Fusion state")
plt.xticks(rotation=20, ha="right")
for bar, count, pct in zip(bars, state_dist["count"], state_dist["percentage"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{count} ({pct}%)", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "1_state_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(" Saved: 1_state_distribution.png")

fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(predict_latencies_ms, bins=20, color="#1f77b4", edgecolor="black", alpha=0.85)
ax.axvline(latency_stats["median_ms"], color="red", linestyle="--", label=f"median={latency_stats['median_ms']:.1f}ms")
ax.axvline(latency_stats["p95_ms"], color="orange", linestyle="--", label=f"p95={latency_stats['p95_ms']:.1f}ms")
ax.set_title("POST /api/fusion/predict Latency (warm)", fontsize=13, fontweight="bold")
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "2_predict_latency_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(" Saved: 2_predict_latency_distribution.png")

print("\nDone.")
