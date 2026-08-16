# Superseded — forecaster training moved out of this folder

`dataset_builder.py` and `train_forecaster.py` are from the first C4 prototype,
when the forecaster was a scikit-learn classifier over a 15-float one-hot
feature vector saved to `forecaster.joblib`.

That design is gone. The forecaster is now a neural model over dialogue-context
text, trained in `../../../../forcasting/`:

| Then (this folder) | Now (`forcasting/`) |
|---|---|
| 7 labels (`neutral, joy, sadness, anger, fear, surprise, disgust`) | 8 labels (`angry, anxious, calm, excited, happy, neutral, sad, stressed`) |
| 15-float one-hot of current/previous emotion + deviation | previous 3 dialogue turns as text + current emotion as an aux embedding |
| sklearn `GradientBoosting` / `RandomForest` / `LogReg` → `forecaster.joblib` | TextCNN / BiLSTM-attention / DistilBERT → `*_forecast.pt` |
| random row splits | `GroupShuffleSplit` on `conversation_id` (no conversation crosses splits) |
| no baselines | majority + persistence baselines logged before any model runs |

Neither script runs against the current code: they call
`EmotionForecaster._build_features()`, which no longer exists, and
`EmotionForecaster` no longer loads joblib files at all. They also need
`scikit-learn` and `joblib`, which the demo's `requirements.txt` no longer
installs.

They are kept only as a record of the earlier approach. **They can be deleted**
without affecting the demo or the smoke test.

## To retrain the forecaster

```bash
cd forcasting/
python preprocess.py --task forecast --context 3
python train_deep.py --task forecast --model all --trials 30
# then copy artifacts/ into ../R26-IT-036/ai_components/c4_emotion_support/models/next_emotion_forecaster/
```

## To retrain the current-emotion classifier

Run `classification/training.ipynb`, then copy
`classification/models/optuna_comparison/roberta-base/` into
`models/current_emotion_classifier/`. See `../models/README.md`.
