# C3 Text Stressor Distortion Streamlit App

This Streamlit app has two independent top-level tabs:

- **Stress Header** — Stress vs Not Stressed
- **CBT Header** — Distortion vs No Distortion

Each tab compares three transformer models and two TF-IDF baselines:

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

The CBT tab expects:

```text
models/c3_text_stressor_distortion/CBT header/
├── BERT_binary_final.pt
├── MentalBERT_binary_final.pt
├── DeBERTa-v3_binary_final.pt
├── binary_baseline_LR.pkl
├── binary_baseline_SVM.pkl
├── binary_config.json
└── BERTopic/
```

## Output

For each model, the app shows:

- Stress prediction
- Stress probability
- Confidence
- Transformer subreddit/category head output where available
- Loading or prediction status

The app also shows majority vote and the highest-confidence model for quick comparison.

The CBT comparison applies the validated model-specific thresholds saved in
`binary_config.json` and shows the threshold beside each distortion probability.
MentalBERT uses the public `bert-base-uncased` tokenizer/architecture scaffold
at runtime and then strictly loads the complete fine-tuned MentalBERT
checkpoint, so the Streamlit app does not require gated repository access.

## BERTopic semantic-theme testing

The **Stress BERTopic — semantic theme testing** panel uses the saved lightweight
topic model from:

```text
models/c3_text_stressor_distortion/Stress header/BERTopic/
```

and topic metadata from:

```text
reports/c3_text_stressor_distortion/Stress header/BERTopic/
```

The testing tab loads the embedding model recorded in the saved BERTopic
configuration. Newly trained Stress and CBT topic models use
`all-mpnet-base-v2`. It embeds the current text with that same encoder,
compares it with all saved non-outlier topic embeddings, and displays one
primary topic candidate plus four ranked alternatives. Alternatives are not
additional topic assignments. The result is interpreted as:

- **Clear match** when similarity is at least 0.45 and the first topic leads
  the second by at least 0.05.
- **Ambiguous match** when the two leading themes are too close.
- **Weak match** when the best similarity is below 0.45.

The panel displays:

- Closest semantic topic and cosine similarity
- Top topic terms
- A separate, collapsed historical-statistics section containing topic size
  and the post-training Stress/Not-Stressed distribution
- Most represented source communities
- Four alternative topics

The overview tab displays the saved topic metrics, largest topics, and static
topic plots. Topic similarity is thematic metadata; it is not a stress
probability, diagnostic confidence, or replacement for the supervised models.

## CBT BERTopic semantic-theme testing

The CBT tab has an equivalent **CBT BERTopic — semantic theme testing** panel.
Its topic space is learned from the short annotated `Distorted part` excerpts,
not the much longer full patient questions. It uses the same one-candidate,
ranked-alternatives, and uncertainty display. A weak result is shown as
**No reliable CBT topic found** and its historical statistics are hidden. For
clear or ambiguous candidates, the panel displays terms, alternatives, and a
separate collapsed section for
training/all-split sizes, post-training distortion-status distribution, and
represented distortion labels. Metrics and plots are loaded from:

```text
reports/c3_text_stressor_distortion/CBT header/BERTopic/
```

Numeric label values are not passed to BERTopic, although excerpt availability
selects the topic-forming corpus. Label distributions are descriptive joins
added after training, not binary or distortion-type predictions.

## Explainable AI

The **XAI explanations** panel supports five local explanation methods:

- **Integrated Gradients**: signed token contributions relative to a padding baseline.
- **SHAP Text/Partition**: token contributions estimated by masking token groups.
- **LIME**: word-level contributions from a sparse local surrogate fitted to
  randomly masked versions of the selected text. The panel reports the local
  surrogate's R² fidelity score and saves both PNG and interactive HTML outputs.
- **Combined SHAP + LIME + Integrated Gradients**: aligns and normalizes the
  three methods' word contributions, then averages them into a consensus score
  with a per-word agreement indicator.
- **Counterfactual explanations**: bounded beam search for minimal lexical edits
  that move the prediction to the opposite class.

Counterfactuals describe the trained model's sensitivity. They must not be
interpreted as a diagnosis or as advice to change how a person expresses distress.

The CBT tab exposes the same five methods through **CBT XAI explanations**.
They explain only the binary Distortion/No Distortion head and use each CBT
transformer's saved decision threshold; they do not predict the ten distortion
types.

On limited-memory GPUs, the app clears cached Stress, CBT, and topic-embedding
resources before a model batch or XAI job. Only the models required by the
current action are then loaded lazily. The BERTopic text encoder stays on CPU
because it embeds only one short input at a time, preserving CUDA memory for
gradient-based explainers. PyTorch expandable CUDA segments are also enabled
to reduce allocator fragmentation.

The advanced methods are also available from the command-line runner:

```bash
python3 "ai_components/c3_text_stressor_distortion/Stress header/xai/run_xai.py" \
  --text "I cannot handle my exam deadlines anymore" \
  --with-combined-xai \
  --with-counterfactuals
```
