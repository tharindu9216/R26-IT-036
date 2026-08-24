# Superseded — forecaster training moved out of this folder

`dataset_builder.py` and `train_forecaster.py` are from the first C4 prototype,
when the forecaster was a scikit-learn classifier over a 15-float one-hot
feature vector saved to `forecaster.joblib`.

That design is gone twice over. The forecaster is now a neural model that
predicts the next emotional **state**, trained in
`../../../../emotion_forecasting_pipeline/`:

| Then (this folder) | Now (`emotion_forecasting_pipeline/`) |
|---|---|
| 7 emotion labels (`neutral, joy, sadness, anger, fear, surprise, disgust`) | 13 **state** labels (`neutral`, `joy`/`sadness`/`anger`/`fear`, `low_*`/`high_*`) |
| 15-float one-hot of current/previous emotion + deviation | the current utterance as text + the current emotion as an aux embedding |
| sklearn `GradientBoosting` / `RandomForest` / `LogReg` → `forecaster.joblib` | TextCNN / BiLSTM / BiGRU / CNN-BiLSTM → `*_state_forecast.pt`, plus TF-IDF LogReg / LinearSVM |
| random row splits | stratified splits on the target state |
| no baselines | majority + prior-by-current-emotion baselines logged before any model runs |
| — | `context_text` excluded as leakage (it predicts the target at 1.0000 accuracy) |

The intermediate design — an 8-label next-*emotion* model over 3 turns of
dialogue context, trained in `../../../../forcasting/` — is also superseded.
Its checkpoints are parked in
`../models/next_emotion_forecaster/archive_8label/`.

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
