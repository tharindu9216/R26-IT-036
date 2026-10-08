# Optional reviewed synthetic type data

Place the reviewed file at:

`data/CBT header/processed/cbt_synthetic_reviewed.csv`

Required CSV columns:

```csv
Patient Question,label,human_reviewed
```

Rules:

- `human_reviewed` must be `true`, `1`, `yes`, or `y`.
- `label` must be an integer from 1 through 10.
- Do not generate label 0 (`No Distortion`) for this file.
- Do not copy or paraphrase validation/test examples.
- Review the dominant label manually before marking a row as reviewed.
- Exact duplicates of any real train, validation, or test text are removed.
- These rows are used only by the conditional type classifier. They are
  never added to validation/test or to the binary transformer training data.

Label mapping:

| Label | Distortion type |
|---:|---|
| 1 | All-or-nothing thinking |
| 2 | Overgeneralization |
| 3 | Mental filter |
| 4 | Should statements |
| 5 | Labeling |
| 6 | Personalization |
| 7 | Magnification |
| 8 | Emotional Reasoning |
| 9 | Mind Reading |
| 10 | Fortune-telling |
