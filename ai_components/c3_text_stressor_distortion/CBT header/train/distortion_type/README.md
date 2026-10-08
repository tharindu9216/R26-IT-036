# Distortion-type SLM research task

This folder is independent from binary classification. It receives only rows
where a distortion is present and predicts one of ten distortion types.

## Recommended model and stages

Primary SLM:

```text
microsoft/Phi-4-mini-instruct
```

Training sequence:

```text
Pretrained Phi-4-mini-instruct
        ↓
QLoRA supervised fine-tuning (SFT)
        ↓
Retrieval of similar CBT examples
        ↓
DPO using human-reviewed chosen/rejected responses
        ↓
Optional GRPO research ablation
```

DPO is the recommended post-training method for the available small dataset.
It is preference optimization rather than online policy-gradient RL. GRPO
should be added only when the research explicitly requires true RL and the
reward can be calculated reliably.

## Prepare SFT data

```bash
python prepare_sft_data.py
```

Generated files:

- `prepared_data/type_train_sft.jsonl`
- `prepared_data/type_val_sft.jsonl`
- `prepared_data/type_test_evaluation.jsonl`
- `prepared_data/dataset_manifest.json`

The test file has no assistant training response. Its reference is stored
separately to reduce the risk of accidentally training on test labels.

## DPO preference schema

Preference examples must be human-reviewed:

```json
{
  "prompt": "Patient text plus the CBT taxonomy",
  "chosen": "Correct label, grounded evidence, cautious explanation",
  "rejected": "Incorrect label or unsupported explanation",
  "human_reviewed": true
}
```

Do not create preference pairs from validation or test rows.

## Suggested reward for optional GRPO

```text
+1.00 exact label
+0.40 evidence overlap with the annotated Distorted part
+0.10 valid JSON schema
+0.10 appropriate uncertainty/abstention
-1.00 incorrect label
-0.50 unsupported evidence
-0.25 invalid output
```

Use train rewards only. Use validation for model selection and test once for
the final comparison.
