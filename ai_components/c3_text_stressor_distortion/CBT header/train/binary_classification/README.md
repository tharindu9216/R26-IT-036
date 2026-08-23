# Binary classification

Task:

```text
0 = No Distortion
1 = Distortion
```

Original labels 1 through 10 are merged into the Distortion class.

Models:

- TF-IDF + class-weighted Logistic Regression baseline
- TF-IDF + calibrated class-weighted Linear SVM baseline
- BERT
- MentalBERT
- DeBERTa-v3
- Out-of-fold validated weighted probability ensemble

The two traditional ML models are research baselines. Only BERT, MentalBERT,
and DeBERTa-v3 participate in the deployment ensemble.

Selection and evaluation:

- Optuna hyperparameters and the fixed epoch count are selected on the
  dedicated validation split.
- Five-fold predictions are generated only from the real training split. Each
  outer fold is trained for the fixed epoch count, so its validation labels do
  not select its checkpoint.
- Ensemble weights and the binary threshold are selected from those
  out-of-fold predictions using macro-F1.
- Each ML baseline also fits TF-IDF independently inside five training folds
  and selects its threshold from training-only out-of-fold probabilities.
- The final models are refit on train plus validation data. The test split is
  used only for the final evaluation.
- Reports include balanced accuracy, macro-F1, Distortion precision/recall/F1,
  specificity, MCC, PR-AUC, and ROC-AUC.

Run on Modal:

```bash
modal run train_modal.py
```

Results are stored in the Modal volume under `binary_v4_outputs/`.
The saved artifacts include `binary_baseline_LR.pkl` and
`binary_baseline_SVM.pkl` alongside the transformer checkpoints and reports.
