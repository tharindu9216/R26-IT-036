# CBT Header BERTopic

This component discovers semantic themes in short CBT distortion excerpts
using a stronger semantic topic-modeling pipeline:

```text
all-mpnet-base-v2 embeddings
        ↓
UMAP dimensionality reduction
        ↓
HDBSCAN clustering
        ↓
optional semantic topic reduction
        ↓
BM25 c-TF-IDF + KeyBERT/MMR representation
```

Numeric distortion labels are not passed into BERTopic. Corpus selection is
annotation-scoped, however: only training rows with a non-empty
`Distorted part` excerpt form the learned topic space. The default CBT terms
receive only a small 1.2x representation-weight boost; they do not change
document embeddings, clustering, or the data-driven topic count. Disable it with
`CBT_BERTOPIC_USE_GUIDED_TERMS=false`. Embedding-level guided BERTopic remains
an optional experiment (`CBT_BERTOPIC_USE_SEED_TOPICS=true`) but is off by
default so training and inference embeddings follow the same path. It
complements the CBT classifier; it does not replace binary prediction.

## Data policy

The model fits only on `cbt_train.csv`. It then assigns the learned topics to
`cbt_val.csv` and `cbt_test.csv`, preventing validation/test content from
influencing the learned topic space.

The short `Distorted part` text is used for topic discovery so training matches
the short CBT statements entered in Streamlit. `Patient Question` is retained
only as a transform/reporting fallback for rows without an excerpt. The binary
status and original 11-class label values are never passed into BERTopic; they
are joined afterward for descriptive summaries.

Before fitting, rows without distortion excerpts are excluded, followed by
conservative removal of exact duplicates and obvious research boilerplate.
Excluded training rows remain in the final reports and are transformed using
their fallback topic text. Audit flags, text source, and exclusion reasons are
included in `document_topics.csv`.

## Run locally

From this directory:

```bash
python3 -m pip install -r ../../../requirements.txt
python3 -m pip install -r requirements.txt
python3 train.py
```

Optional CBT-specific settings (shared `BERTOPIC_*` names are also accepted):

```bash
export CBT_BERTOPIC_BATCH_SIZE=64
export CBT_BERTOPIC_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
export CBT_BERTOPIC_MIN_TOPIC_SIZE=15
export CBT_BERTOPIC_HDBSCAN_MIN_SAMPLES=3
export CBT_BERTOPIC_HDBSCAN_SELECTION_METHOD=leaf
export CBT_BERTOPIC_TOP_N_WORDS=10
export CBT_BERTOPIC_UMAP_NEIGHBORS=15
export CBT_BERTOPIC_NR_TOPICS=none
export CBT_BERTOPIC_VECTORIZER_MIN_DF=3
export CBT_BERTOPIC_FILTER_BOILERPLATE=true
export CBT_BERTOPIC_DEDUPLICATE_FIT_DOCUMENTS=true
export CBT_BERTOPIC_USE_GUIDED_TERMS=true
export CBT_BERTOPIC_USE_SEED_TOPICS=false
export OUTPUT_DIR=/path/to/output
```

The stable-topic defaults require at least 15 fitting documents per base
cluster and use a broader 15-document UMAP neighbourhood, so groups like the
old eight-document quiz/psychopathy topic cannot become standalone themes.
HDBSCAN `leaf` selection avoids over-collapsing the data. Automatic reduction
is disabled by default after it proved too aggressive on the Stress dataset.
Set `CBT_BERTOPIC_NR_TOPICS=auto` to experiment with automatic reduction or
provide an integer to request a fixed maximum.

## Run on Modal

The public MPNet embedding model does not require a Hugging Face token. It is
cached in the Modal image before the GPU training function starts. If
`HF_TOKEN` exists in the project `.env`, Modal injects it as a secret but does
not upload the `.env` file.

```bash
modal run train_modal.py
```

The default is a B200 GPU with a CUDA 12.8 PyTorch build. You can set it
explicitly in `.env`:

```env
CBT_BERTOPIC_GPU=B200
```

`bertopic==0.16.4` and `sentence-transformers==3.0.1` are intentionally pinned
together. BERTopic 0.17.4 is incompatible with that Sentence Transformers
version because it imports the newer `StaticEmbedding` class.

Download the completed results:

```bash
modal volume get cbt-bertopic-outputs \
  cbt_bertopic_outputs ./cbt_bertopic_results/
```

## Outputs

```text
cbt_bertopic_outputs/
├── topic_model/                           # Saved BERTopic model
├── topic_info.csv                         # Topic names and sizes
├── topic_terms.csv                        # Ranked c-TF-IDF terms
├── document_topics.csv                    # Topic per train/val/test row
├── document_topic_probabilities.npy       # Full probability matrix
├── probability_matrix_metadata.json       # Probability column mapping
├── topic_by_distortion_status.csv          # Distortion/No-Distortion by topic
├── topic_by_distortion_type.csv            # Original 11 classes by topic
├── topic_metrics.json                      # Diversity/outlier/silhouette
├── dataset_summary.json                    # Split and class counts
├── fit_filter_summary.json                 # Fit-time quality-control audit
├── run_config.json                         # Reproducibility metadata
├── training.log
└── plots/
    ├── topic_sizes.png
    ├── topic_distortion_status_distribution.png
    ├── topic_distortion_type_distribution.png
    ├── topics.html
    ├── topic_barchart.html
    ├── topic_heatmap.html
    └── topic_hierarchy.html
```

Topic `-1` is the HDBSCAN outlier group. It represents documents that were not
confidently assigned to a learned topic.
