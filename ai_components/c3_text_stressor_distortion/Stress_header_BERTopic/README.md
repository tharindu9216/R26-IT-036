# Stress Header BERTopic

This component discovers semantic themes in the Stress Header text using:

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

It is a class-label-free, lightly domain-guided topic-discovery component. The
default domain terms receive only a small 1.2x representation-weight boost;
they do not change document embeddings, clustering, the data-driven topic
count, or use Stress/Not-Stressed labels. Disable that boost with
`BERTOPIC_USE_GUIDED_TERMS=false`. Embedding-level guided BERTopic remains an
optional experiment (`BERTOPIC_USE_SEED_TOPICS=true`) but is off by default so
training and inference embeddings follow the same path. This component does
not replace the Stress/Not-Stressed classifier.

## Data policy

By default, BERTopic is fitted only on `dreaddit_train.csv`. The learned topic
space then assigns topics to `dreaddit_val.csv` and `dreaddit_test.csv`. This
prevents validation/test documents from influencing topics if topic features
are later used by a predictive model.

The raw `text` column is used for semantic embeddings and human-readable topic
reports. Stress labels and subreddit names are used only for post-training
descriptive summaries; they are never passed into BERTopic.

Before fitting, conservative quality controls remove exact duplicate texts and
obvious survey/research recruitment boilerplate. Removed training rows are not
discarded from the reports: the learned model assigns them afterward, and
`document_topics.csv` records why each row was excluded from topic discovery.

## Run locally

From this directory:

```bash
python3 -m pip install -r ../../../requirements.txt
python3 -m pip install -r requirements.txt
python3 train.py
```

Optional environment settings:

```bash
export BERTOPIC_BATCH_SIZE=64
export BERTOPIC_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
export BERTOPIC_MIN_TOPIC_SIZE=15
export BERTOPIC_HDBSCAN_MIN_SAMPLES=3
export BERTOPIC_HDBSCAN_SELECTION_METHOD=leaf
export BERTOPIC_TOP_N_WORDS=10
export BERTOPIC_UMAP_NEIGHBORS=15
export BERTOPIC_NR_TOPICS=none
export BERTOPIC_VECTORIZER_MIN_DF=3
export BERTOPIC_FILTER_BOILERPLATE=true
export BERTOPIC_DEDUPLICATE_FIT_DOCUMENTS=true
export BERTOPIC_USE_GUIDED_TERMS=true
export BERTOPIC_USE_SEED_TOPICS=false
export OUTPUT_DIR=/path/to/output
```

The stable-topic defaults require at least 15 fitting documents per base
cluster and use a broader 15-document UMAP neighbourhood. HDBSCAN `leaf`
selection avoids the two-topic collapse observed with `eom`. Automatic topic
reduction is disabled by default because it reduced 34 discovered groups to
only 14 non-outlier topics on this dataset. Set `BERTOPIC_NR_TOPICS=auto` to
experiment with automatic reduction or provide an integer for a fixed maximum.

## Run on Modal

The public MPNet model does not require a Hugging Face token. It is cached in
the Modal image before the GPU training function starts. If `HF_TOKEN`
is present in the project `.env`, Modal injects it as a secret without
uploading the `.env` file.

```bash
modal run train_modal.py
```

The default is a B200 GPU with a CUDA 12.8 PyTorch build. Override it by
placing a different supported GPU in `.env`:

```env
BERTOPIC_GPU=B200
```

`bertopic==0.16.4` and `sentence-transformers==3.0.1` are intentionally pinned
together. BERTopic 0.17.4 is incompatible with that Sentence Transformers
version because it imports the newer `StaticEmbedding` class.

Download the results:

```bash
modal volume get stress-bertopic-outputs \
  stress_bertopic_outputs ./stress_bertopic_results/
```

## Outputs

```text
stress_bertopic_outputs/
├── topic_model/                        # Saved BERTopic model
├── topic_info.csv                    # Topic names and sizes
├── topic_terms.csv                   # Ranked c-TF-IDF terms
├── document_topics.csv               # Topic per train/val/test row
├── document_topic_probabilities.npy  # Full probability matrix
├── probability_matrix_metadata.json  # Probability column mapping
├── topic_by_stress_label.csv         # Descriptive label summary
├── topic_by_subreddit.csv            # Descriptive subreddit summary
├── topic_metrics.json                 # Diversity/outlier/silhouette
├── fit_filter_summary.json            # Fit-time quality-control audit
├── run_config.json                    # Reproducibility metadata
├── training.log
└── plots/
    ├── topic_sizes.png
    ├── topic_stress_distribution.png
    ├── topics.html
    ├── topic_barchart.html
    ├── topic_heatmap.html
    └── topic_hierarchy.html
```

Topic `-1` represents documents that HDBSCAN considers outliers.
