# Dreaddit Stress Dataset

This folder contains the processed Dreaddit dataset used for stress detection experiments.

Source dataset: https://huggingface.co/datasets/andreagasparini/dreaddit

Original paper: https://aclanthology.org/D19-6213/

## Dataset Overview

Dreaddit is a Reddit-based dataset for stress analysis in social media. It contains text segments from multiple Reddit communities, annotated for whether the text expresses stress. The dataset was introduced by Turcan and McKeown (2019) to support supervised stress detection on longer, multi-domain social media posts.

In this repository, the dataset is prepared for:

- Binary stress classification: stressed vs. not stressed
- Subreddit/category classification
- Transformer-based experiments
- Traditional machine learning baselines using TF-IDF features

## Files

The processed data is stored under `processed/`.

| File | Rows | Description |
| --- | ---: | --- |
| `processed/dreaddit_train.csv` | 2,258 | Training split |
| `processed/dreaddit_val.csv` | 566 | Validation split |
| `processed/dreaddit_test.csv` | 715 | Test split |
| `processed/metadata.json` | - | Dataset, label, model, preprocessing, and class-weight metadata |

CSV row counts exclude the header row.

## Columns

| Column | Description |
| --- | --- |
| `text` | Original text used for machine learning baselines |
| `text_for_bert` | Preprocessed text for BERT-based models |
| `text_for_mentalbert` | Preprocessed text for MentalBERT |
| `text_for_deberta` | Preprocessed text for DeBERTa-v3 |
| `label` | Binary stress label: `0` = not stressed, `1` = stressed |
| `subreddit` | Source subreddit/category name |
| `subreddit_label` | Numeric subreddit/category label |
| `confidence` | Annotation confidence score |
| `sentiment` | Sentiment score included with the processed data |

## Labels

### Binary Stress Label

| Value | Meaning |
| ---: | --- |
| `0` | Not stressed |
| `1` | Stressed |

### Subreddit Labels

| Value | Subreddit |
| ---: | --- |
| `0` | `almosthomeless` |
| `1` | `anxiety` |
| `2` | `assistance` |
| `3` | `domesticviolence` |
| `4` | `food_pantry` |
| `5` | `homeless` |
| `6` | `ptsd` |
| `7` | `relationships` |
| `8` | `stress` |
| `9` | `survivorsofabuse` |

## Preprocessing Notes

The processed dataset includes model-specific text columns. According to `processed/metadata.json`, preprocessing includes:

- Contraction expansion
- Preserved negation
- Repeated character handling
- Punctuation normalization
- No stopword removal
- No lemmatization

For traditional ML baselines, use the raw `text` column. For transformer models, use the matching model-specific text column listed in `metadata.json`.

## Intended Use

This data should be used for research and experimentation on stress detection in social media text. Because the content comes from Reddit communities related to sensitive personal experiences, handle the data carefully and avoid attempts to identify users or communities beyond the labels already provided.

## Citation

If you use this dataset, cite the original Dreaddit paper:

```bibtex
@inproceedings{turcan-mckeown-2019-dreaddit,
    title = "{D}readdit: A {R}eddit Dataset for Stress Analysis in Social Media",
    author = "Turcan, Elsbeth  and
      McKeown, Kathy",
    editor = "Holderness, Eben  and
      Jimeno Yepes, Antonio  and
      Lavelli, Alberto  and
      Minard, Anne-Lyse  and
      Pustejovsky, James  and
      Rinaldi, Fabio",
    booktitle = "Proceedings of the Tenth International Workshop on Health Text Mining and Information Analysis (LOUHI 2019)",
    month = nov,
    year = "2019",
    address = "Hong Kong",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-6213/",
    doi = "10.18653/v1/D19-6213",
    pages = "97--107",
    abstract = "Stress is a nigh-universal human experience, particularly in the online world. While stress can be a motivator, too much stress is associated with many negative health outcomes, making its identification useful across a range of domains. However, existing computational research typically only studies stress in domains such as speech, or in short genres such as Twitter. We present Dreaddit, a new text corpus of lengthy multi-domain social media data for the identification of stress. Our dataset consists of 190K posts from five different categories of Reddit communities; we additionally label 3.5K total segments taken from 3K posts using Amazon Mechanical Turk. We present preliminary supervised learning methods for identifying stress, both neural and traditional, and analyze the complexity and diversity of the data and characteristics of each category."
}
```

## License and Terms

Check the upstream Hugging Face dataset page and original paper for the latest dataset license and usage terms before redistributing or publishing derived artifacts.
