"""Configuration for CBT Header BERTopic topic discovery."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _setting(name, default):
    """Prefer a CBT-specific setting, then the shared BERTopic setting."""
    return os.getenv(f'CBT_{name}', os.getenv(name, default))


def _positive_int(name, default):
    value = int(_setting(name, default))
    if value <= 0:
        raise ValueError(f'CBT_{name} must be a positive integer')
    return value


def _boolean(name, default):
    value = _setting(name, str(default)).strip().lower()
    if value in {'1', 'true', 'yes', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(f'CBT_{name} must be true or false')


def _topic_reduction(name, default='none'):
    value = _setting(name, default).strip().lower()
    if value == 'auto':
        return 'auto'
    if value in {'none', 'off', 'false', '0'}:
        return None
    topic_count = int(value)
    if topic_count < 2:
        raise ValueError(
            f'CBT_{name} must be auto, none, or an integer >= 2')
    return topic_count


class Config:
    DATA_DIR = Path(os.getenv(
        'DATA_DIR',
        PROJECT_ROOT / 'data/CBT header/processed',
    ))
    OUTPUT_DIR = Path(os.getenv(
        'OUTPUT_DIR',
        PROJECT_ROOT
        / 'models/c3_text_stressor_distortion/CBT_header_BERTopic',
    ))

    SEED = 42
    # The topic geometry is learned from the short, annotated CBT excerpts.
    # Full questions remain available only as a reporting/transform fallback
    # for rows that do not have an excerpt (the No Distortion rows).
    TEXT_COLUMN = 'Patient Question'
    DISTORTED_TEXT_COLUMN = 'Distorted part'
    TOPIC_TEXT_COLUMN = 'topic_text'
    TOPIC_TEXT_SOURCE_COLUMN = 'topic_text_source'
    EMBEDDING_MODEL = _setting(
        'BERTOPIC_EMBEDDING_MODEL',
        'sentence-transformers/all-mpnet-base-v2',
    )
    EMBEDDING_BATCH_SIZE = _positive_int('BERTOPIC_BATCH_SIZE', 64)

    # Learn the topic space from training data only. Validation and test data
    # are assigned to the already learned topics to prevent data leakage.
    FIT_SPLITS = ('train',)
    TRANSFORM_SPLITS = ('val', 'test')

    # Stable-topic defaults. These stop tiny groups of unusual questions from
    # becoming standalone themes while retaining enough detail for CBT use.
    # Automatic reduction remains optional but is disabled by default so it
    # cannot over-collapse useful CBT themes after stable clustering.
    MIN_TOPIC_SIZE = _positive_int('BERTOPIC_MIN_TOPIC_SIZE', 15)
    HDBSCAN_MIN_SAMPLES = _positive_int(
        'BERTOPIC_HDBSCAN_MIN_SAMPLES', 3)
    HDBSCAN_SELECTION_METHOD = _setting(
        'BERTOPIC_HDBSCAN_SELECTION_METHOD', 'leaf').strip().lower()
    if HDBSCAN_SELECTION_METHOD not in {'eom', 'leaf'}:
        raise ValueError(
            'CBT_BERTOPIC_HDBSCAN_SELECTION_METHOD must be eom or leaf')
    TOP_N_WORDS = _positive_int('BERTOPIC_TOP_N_WORDS', 10)
    UMAP_N_NEIGHBORS = _positive_int('BERTOPIC_UMAP_NEIGHBORS', 15)
    UMAP_N_COMPONENTS = _positive_int('BERTOPIC_UMAP_COMPONENTS', 5)
    MAX_PLOT_TOPICS = _positive_int('BERTOPIC_MAX_PLOT_TOPICS', 20)
    NR_TOPICS = _topic_reduction('BERTOPIC_NR_TOPICS')
    VECTORIZER_MIN_DF = _positive_int('BERTOPIC_VECTORIZER_MIN_DF', 3)
    FILTER_BOILERPLATE = _boolean(
        'BERTOPIC_FILTER_BOILERPLATE', True)
    DEDUPLICATE_FIT_DOCUMENTS = _boolean(
        'BERTOPIC_DEDUPLICATE_FIT_DOCUMENTS', True)
    USE_GUIDED_TERMS = _boolean('BERTOPIC_USE_GUIDED_TERMS', True)
    USE_SEED_TOPICS = _boolean('BERTOPIC_USE_SEED_TOPICS', False)

    # Domain terms improve representations without using the CBT labels. By
    # default they receive a mild c-TF-IDF boost, not geometry-level seeding.
    SEED_TOPIC_LIST = (
        ('failure', 'mistake', 'worthless', 'inadequate', 'self esteem'),
        ('perfectionism', 'standards', 'expectations', 'good enough'),
        ('rejection', 'abandonment', 'relationship', 'breakup', 'lonely'),
        ('judgment', 'embarrassed', 'social', 'people think', 'criticize'),
        ('work', 'school', 'exam', 'performance', 'career'),
        ('future', 'catastrophe', 'worst', 'uncertainty', 'fortune'),
        ('health', 'illness', 'body', 'symptoms', 'diagnosis'),
        ('guilt', 'blame', 'responsibility', 'fault', 'regret'),
        ('anxiety', 'panic', 'fear', 'worry', 'overthinking'),
        ('anger', 'argument', 'conflict', 'resentment', 'frustrated'),
        ('control', 'should', 'must', 'rules', 'demand'),
        ('comparison', 'success', 'achievement', 'better than', 'inferior'),
    )

    SPLIT_FILES = {
        'train': 'cbt_train.csv',
        'val': 'cbt_val.csv',
        'test': 'cbt_test.csv',
    }

    DISTORTION_LABELS = {
        0: 'No Distortion',
        1: 'All-or-nothing thinking',
        2: 'Overgeneralization',
        3: 'Mental filter',
        4: 'Should statements',
        5: 'Labeling',
        6: 'Personalization',
        7: 'Magnification',
        8: 'Emotional Reasoning',
        9: 'Mind Reading',
        10: 'Fortune-telling',
    }

    @staticmethod
    def device():
        try:
            import torch
        except ModuleNotFoundError:
            return 'cpu'
        return 'cuda' if torch.cuda.is_available() else 'cpu'
