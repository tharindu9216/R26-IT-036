# C3 Stress and CBT header unit tests

This directory contains one sample-text test module for each deployed C3
classification header. Each test sends balanced sample sentences through the
real trained transformer ensemble and compares its prediction with the expected
Stress or CBT cognitive-distortion label.

Run the tests and regenerate all report files and charts with:

```bash
python test/c3_text_stressor_distortion/generate_report.py
```

Compare every deployed transformer member with its final weighted ensemble and
generate a separate chart for each header with:

```bash
python test/c3_text_stressor_distortion/generate_ensemble_charts.py
```

The two charts are saved under `report/ensemble/`.

The run requires the backend ML dependencies and trained checkpoints. Generated
predictions, metrics, and PNG accuracy, confusion-matrix, and probability
charts are saved under `report/`.
