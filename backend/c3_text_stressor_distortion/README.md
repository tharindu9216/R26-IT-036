# Component 3 — Stress Assistant Backend

FastAPI backend for:

- Equal-weight BERT + DeBERTa-v3 Stress probability ensemble.
- OOF-weighted BERT + MentalBERT + DeBERTa-v3 CBT probability ensemble.
- Transparent four-state decision-level fusion.
- On-demand hierarchical SHAP + LIME + Integrated Gradients consensus and
  rule-trace explanations.

## Run

```bash
pip install -r backend/c3_text_stressor_distortion/requirements.txt
uvicorn app.main:app --reload --app-dir backend/c3_text_stressor_distortion --host 0.0.0.0 --port 8005
```

Environment variables:

- `DEBERTA_CHECKPOINT`, default `models/c3_text_stressor_distortion/Stress header/DeBERTa-v3_best.pt`
- `STRESS_DEVICE`, default `auto`

## Fusion endpoints

- `POST /api/fusion/predict` runs both independent heads and returns their
  original results plus the four-state combined decision.
- `POST /api/diary/{entry_id}/explain` generates representative DeBERTa-v3
  SHAP + LIME + Integrated Gradients consensus evidence for each head and
  explains the deterministic rule that produced the combined state.

Fusion is not a trained severity or diagnostic model. Consensus evidence
describes model behaviour and is generated only when requested.
