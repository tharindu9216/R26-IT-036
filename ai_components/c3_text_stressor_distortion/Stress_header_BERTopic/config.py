"""Configuration for Stress Header BERTopic discovery."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _positive_int(name, default):
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _boolean(name, default):
    value = os.getenv(name, str(default)).strip().lower()
    if value in {'1', 'true', 'yes', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(f'{name} must be true or false')


def _topic_reduction(name, default='none'):
    value = os.getenv(name, default).strip().lower()
    if value == 'auto':
        return 'auto'
    if value in {'none', 'off', 'false', '0'}:
        return None
    topic_count = int(value)
    if topic_count < 2:
        raise ValueError(f'{name} must be auto, none, or an integer >= 2')
    return topic_count


class Config:
    DATA_DIR = Path(os.getenv(
        'DATA_DIR',
        PROJECT_ROOT / 'data/Stress header/processed',
    ))
    OUTPUT_DIR = Path(os.getenv(
        'OUTPUT_DIR',
        PROJECT_ROOT
        / 'models/c3_text_stressor_distortion/Stress_header_BERTopic',
    ))

    SEED = 42
    TEXT_COLUMN = 'text'
    EMBEDDING_MODEL = os.getenv(
        'BERTOPIC_EMBEDDING_MODEL',
        'sentence-transformers/all-mpnet-base-v2',
    )
    EMBEDDING_BATCH_SIZE = _positive_int(
        'BERTOPIC_BATCH_SIZE', 64)

    # Fit topics on training documents only. Validation and test documents are
    # transformed afterward, so they cannot change the learned topic space.
    FIT_SPLITS = ('train',)
    TRANSFORM_SPLITS = ('val', 'test')

    # Stable-topic defaults. A larger neighbourhood and cluster floor prevent
    # a handful of unusual posts from becoming misleading standalone topics.
    # Leaf selection is retained because EOM collapsed this dataset too far;
    # Automatic reduction remains available as an experiment, but is disabled
    # by default because it merged 34 discovered groups down to only 14 topics.
    MIN_TOPIC_SIZE = _positive_int('BERTOPIC_MIN_TOPIC_SIZE', 15)
    HDBSCAN_MIN_SAMPLES = _positive_int(
        'BERTOPIC_HDBSCAN_MIN_SAMPLES', 3)
    HDBSCAN_SELECTION_METHOD = os.getenv(
        'BERTOPIC_HDBSCAN_SELECTION_METHOD', 'leaf').strip().lower()
    if HDBSCAN_SELECTION_METHOD not in {'eom', 'leaf'}:
        raise ValueError(
            'BERTOPIC_HDBSCAN_SELECTION_METHOD must be eom or leaf')
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

    # These are representation hints, not prediction labels. By default they
    # receive a mild c-TF-IDF boost without changing document clustering.
    SEED_TOPIC_LIST = (
        ('work', 'job', 'deadline', 'manager', 'career'),
        ('money', 'debt', 'rent', 'financial', 'bills'),
        ('family', 'parent', 'child', 'home', 'conflict'),
        ('relationship', 'partner', 'breakup', 'dating', 'marriage'),
        ('school', 'college', 'exam', 'studying', 'grades'),
        ('anxiety', 'panic', 'worry', 'fear', 'overwhelmed'),
        ('depression', 'lonely', 'hopeless', 'sad', 'motivation'),
        ('trauma', 'abuse', 'ptsd', 'flashback', 'safety'),
        ('health', 'illness', 'pain', 'doctor', 'treatment'),
        ('grief', 'death', 'loss', 'bereavement', 'funeral'),
        ('housing', 'homeless', 'landlord', 'eviction', 'shelter'),
        ('caregiving', 'disabled', 'elderly', 'support', 'responsibility'),
    )

    SPLIT_FILES = {
        'train': 'dreaddit_train.csv',
        'val': 'dreaddit_val.csv',
        'test': 'dreaddit_test.csv',
    }

    @staticmethod
    def device():
        try:
            import torch
        except ModuleNotFoundError:
            return 'cpu'
        return 'cuda' if torch.cuda.is_available() else 'cpu'
