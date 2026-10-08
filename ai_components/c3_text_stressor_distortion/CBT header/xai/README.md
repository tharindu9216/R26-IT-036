# CBT Header XAI

This folder explains the trained binary CBT models that predict:

- `0` — No Distortion
- `1` — Distortion

Supported transformer checkpoints:

- BERT
- MentalBERT
- DeBERTa-v3

The loader reads architecture, preprocessing, maximum length, and each
model's validated decision threshold from
`models/c3_text_stressor_distortion/CBT header/binary_config.json`.

## Methods

- Integrated Gradients
- SHAP Text/Partition
- LIME
- Combined SHAP + LIME + Integrated Gradients
- Counterfactual explanations

These methods explain the binary decision only. They do not infer which of
the ten cognitive-distortion types is present.

For consumer GPUs, Integrated Gradients tokenizes to the input's actual length
instead of padding every sample to the CBT model's 448/512-token maximum. Its
interpolation batch size is one, and SHAP/LIME/counterfactual predictions use
small dynamically padded batches. This trades some runtime for bounded GPU
memory without changing the trained checkpoint or decision threshold.

## Run

From the repository root:

```bash
python3 "ai_components/c3_text_stressor_distortion/CBT header/xai/run_xai.py" \
  --model DeBERTa-v3 \
  --text "If I make one mistake, everyone will think I am a failure."
```

Run LIME and counterfactuals too:

```bash
python3 "ai_components/c3_text_stressor_distortion/CBT header/xai/run_xai.py" \
  --model DeBERTa-v3 \
  --text "If I make one mistake, everyone will think I am a failure." \
  --with-lime \
  --with-counterfactuals
```

Run the combined explanation:

```bash
python3 "ai_components/c3_text_stressor_distortion/CBT header/xai/run_xai.py" \
  --model DeBERTa-v3 \
  --text "If I make one mistake, everyone will think I am a failure." \
  --with-combined-xai
```

Outputs are saved under:

```text
reports/c3_text_stressor_distortion/CBT header/evaluation/xai_outputs/
```

The first run may download the selected Hugging Face base model if it is not
already cached. XAI output describes model behavior and is not a CBT
assessment or diagnosis.

MentalBERT inference does not require access to the gated MentalBERT
repository. Its BERT-compatible public tokenizer/architecture scaffold is
created from `bert-base-uncased`, then the complete saved MentalBERT binary
checkpoint is loaded strictly over that scaffold.
