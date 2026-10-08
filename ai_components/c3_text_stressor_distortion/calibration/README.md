# C3 probability calibration

This pipeline fits affine binary temperature scaling (temperature plus a
learned bias, equivalently Platt/logistic scaling) to each deployed ensemble's
positive-class log-odds. The bias corrects an ensemble prior/prevalence offset
that slope-only temperature scaling cannot represent. It reports ECE, Brier
score, negative log-likelihood, and a reliability diagram before and after
calibration.

The original held-out test split is deterministically repartitioned into a
20% calibration partition and an 80% final evaluation partition. Checkpoints,
weights, and thresholds are locked first. Temperature is fitted only on the
calibration partition, while calibration quality is reported only on the
disjoint evaluation partition. The original threshold is transformed through
the same monotonic calibration function, preserving every binary decision.

Run from the repository root:

```bash
python ai_components/c3_text_stressor_distortion/calibration/evaluate_and_fit.py
```

Runtime artifacts are written beside the model checkpoints; reports and
reliability diagrams are written under
`reports/c3_text_stressor_distortion/Calibration/`.
