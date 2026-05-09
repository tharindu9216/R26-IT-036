# C3 Text Stressor Distortion Streamlit App

This Streamlit app compares all saved C3 stress detection models on one user text input:

- BERT
- MentalBERT
- DeBERTa-v3
- TF-IDF + Logistic Regression
- TF-IDF + SVM

## Run

From the repository root:

```bash
streamlit run "streamlit_app/c3_text_stressor_distortion/stressor head/app.py"
```

## Expected Model Files

The app expects the trained artifacts in:

```text
models/c3_text_stressor_distortion/Stress header/
```

Required files:

```text
BERT_best.pt
MentalBERT_best.pt
DeBERTa-v3_best.pt
baseline_LR.pkl
baseline_SVM.pkl
```

## Output

For each model, the app shows:

- Stress prediction
- Stress probability
- Confidence
- Transformer subreddit/category head output where available
- Loading or prediction status

The app also shows majority vote and the highest-confidence model for quick comparison.
