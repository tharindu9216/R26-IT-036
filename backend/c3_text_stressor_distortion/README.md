# Component 3 — Stress Assistant Backend

FastAPI backend for:

- Equal-weight BERT + DeBERTa-v3 Stress probability ensemble.
- OOF-weighted BERT + MentalBERT + DeBERTa-v3 CBT probability ensemble.
- Training-equivalent, task/model-specific preprocessing before tokenization.
- Affine temperature-calibrated probabilities with decision-preserving
  transformed thresholds.
- Transparent four-state decision-level fusion.
- On-demand ensemble-weighted SHAP + LIME + Integrated Gradients consensus
  and rule-trace explanations.

## Run

```bash
pip install -r backend/c3_text_stressor_distortion/requirements.txt
uvicorn app.main:app --reload --app-dir backend/c3_text_stressor_distortion --host 0.0.0.0 --port 8005
```

Environment variables:

- Stress checkpoints default to the reported final-refit artifacts:
  `BERT_final.pt` and `DeBERTa-v3_final.pt`.
- Calibration artifacts default to `stress_ensemble_calibration.json` and
  `binary_ensemble_calibration.json` beside the task checkpoints.
- `STRESS_DEVICE`, default `auto`

## Fusion endpoints

- `POST /api/fusion/predict` runs both independent heads and returns their
  original results plus the four-state combined decision.
- `POST /api/diary/{entry_id}/explain` explains every deployed member, then
  combines normalized word-level SHAP + LIME + Integrated Gradients evidence
  with the same weights used by each probability ensemble. It also explains
  the deterministic rule that produced the combined state.
- `GET /api/health` reports checkpoint and probability-calibration metadata.

Fusion is not a trained severity or diagnostic model. Consensus evidence
describes model behaviour and is generated only when requested.
