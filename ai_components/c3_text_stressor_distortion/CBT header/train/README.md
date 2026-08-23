# CBT training tasks

The CBT component is separated into two research tasks:

```text
train/
├── binary_classification/
│   ├── train.py
│   ├── train_modal.py
│   └── training.py
├── distortion_type/
│   ├── config.py
│   ├── prepare_sft_data.py
│   └── README.md
├── config.py
├── dataset.py
├── model.py
├── utils.py
└── augmentation.py
```

- `binary_classification/` predicts only No Distortion vs Distortion using
  TF-IDF Logistic Regression and Linear SVM research baselines, plus BERT,
  MentalBERT, DeBERTa-v3, and their validated weighted ensemble.
- `distortion_type/` contains the independent SLM research pipeline for the
  ten distortion types. It must only receive examples already identified as
  distorted.
- Files in the root of `train/` are shared transformer/data/metric utilities
  or retained legacy experiment modules.

Run the binary task:

```bash
cd binary_classification
modal run train_modal.py
```
