# C3 semantic model tests

Each header contains one balanced corpus of 50 independently labelled cases:
25 positive and 25 negative. The same corpus is evaluated by all five trained
models and the research ensemble:

1. TF-IDF + Logistic Regression
2. TF-IDF + calibrated Linear SVM
3. BERT
4. MentalBERT
5. DeBERTa-v3
6. Ensemble

This produces 300 model checks per header and 600 checks in total.

The cases are intentionally separate from the training and held-out CSV files.
They cover clear positives, clear negatives, negation, uncertainty, coping,
and boundary language. Each failed assertion includes the model probability,
case category, and human rationale.

From the repository root, evaluate all 600 model/case combinations with:

```bash
python test/c3_text_stressor_distortion/run_model_tests.py
```

The command prints a metrics table for each header and writes these files under
`test/c3_text_stressor_distortion/reports/{cbt,stress}/`:

- `metrics.csv` and `metrics.json`
- `predictions.csv`
- `metrics_comparison.png`
- `confusion_matrices.png`
- `roc_pr_curves.png`

When all CBT models are evaluated, the runner also compares five fixed
ensemble strategies without rerunning model inference:

1. Saved OOF-weighted soft vote over BERT, MentalBERT, and DeBERTa-v3
2. Equal soft vote over the three transformers
3. Hard majority vote over the three transformers
4. Equal soft vote over all five individual models
5. Hard majority vote over all five individual models

The comparison is saved as `ensemble_metrics.csv`, `ensemble_metrics.json`,
`ensemble_predictions.csv`, `ensemble_metrics_comparison.png`,
`ensemble_confusion_matrices.png`, and `ensemble_roc_pr_curves.png` in the CBT
report directory. These are evaluation comparisons, not newly selected
deployment settings; only the saved OOF-weighted ensemble was selected without
using the 50 semantic test cases.

Run one 50-case model evaluation with, for example:

```bash
python test/c3_text_stressor_distortion/run_model_tests.py \
  --header cbt --model ensemble
```

Valid headers are `cbt`, `stress`, and `all`. Valid model choices are `lr`,
`svm`, `bert`, `mentalbert`, `deberta`, `ensemble`, and `all`. The runner uses
the configured `STRESS_DEVICE`; set `STRESS_DEVICE=cpu` if GPU inference is not
desired.

The Stress ensemble reproduces the training pipeline's unweighted BERT +
DeBERTa-v3 probability average at threshold 0.50. The CBT ensemble uses the
saved OOF-selected BERT/MentalBERT/DeBERTa-v3 weights (0.10/0.40/0.50) and
threshold 0.37. The two TF-IDF baselines remain comparison models and are not
silently added to either trained ensemble.

The evaluation command exits successfully after producing its reports even
when a model misclassifies semantic cases. Add `--strict` if any
misclassification should result in a non-zero exit code. Ordinary `unittest`
or `pytest` discovery sees all cases but skips heavyweight inference unless
`RUN_C3_MODEL_TESTS=1` is set. Required checkpoints must exist under
`models/c3_text_stressor_distortion`, and the dependencies in the root
`requirements.txt` must be installed before an inference run.

## BERTopic tests

Each header also has a separate 50-case BERTopic corpus and a corresponding
`test_bertopic.py`. Run both saved topic models with:

```bash
python test/c3_text_stressor_distortion/run_bertopic_tests.py
```

Use `--header stress` to test only the BERTopic integration used by the React
diary or `--header cbt` to test only the CBT research model. The runner writes
these files under
`test/c3_text_stressor_distortion/reports/{cbt,stress}/bertopic/`:

- `topic_metrics.csv` and `topic_metrics.json`
- `topic_predictions.csv`
- `topic_distribution.png`
- `similarity_distribution.png`
- `clarity_and_latency.png`

BERTopic is an unsupervised semantic clustering model. Therefore its report
uses valid-assignment rate, topic coverage, clear/weak/ambiguous match rates,
cosine similarity, repeatability, and inference latency. Accuracy, precision,
recall, F1, and confusion matrices are intentionally not reported because the
50 cases do not have independently human-annotated ground-truth topic IDs.
Weak or ambiguous results are valid uncertainty behavior and do not fail a
test. Stress cases call the exact `TopicPredictor` used by the React diary with
its `0.48` clarity threshold. CBT cases reproduce the Streamlit research
interface with its `0.45` similarity threshold and `0.05` ambiguity margin.

Ordinary discovery skips the heavyweight topic encoder. To execute the
individual unittest methods directly, set `RUN_C3_BERTOPIC_TESTS=1`.
