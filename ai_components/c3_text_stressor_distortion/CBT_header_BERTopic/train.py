"""Train CBT BERTopic on annotated distortion excerpts without leakage."""

import importlib.metadata
import logging
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
)
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from config import Config
from reporting import (
    assigned_probabilities,
    calculate_topic_metrics,
    save_interactive_plots,
    save_json,
    save_static_plots,
    topic_terms_frame,
)


_RESEARCH_MARKERS = (
    re.compile(
        r'\b(?:surveys?|questionnaires?|research (?:study|project)|'
        r'focus group|online study|academic study)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:participants?|participation|participat(?:e|ing)|'
        r'recruit(?:ing|ment)?)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:consent form|informed consent|institutional review|'
        r'ethics approval)\b',
        re.IGNORECASE,
    ),
)
_SOLICITATION_MARKER = re.compile(
    r"\b(?:looking for|seeking|we are conducting|we're conducting|"
    r'we are running|you are invited|voluntary|eligible|qualify|'
    r'compensated|compensation|gift card|anonymous|confidential)\b',
    re.IGNORECASE,
)


def build_logger(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('cbt_bertopic')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s')
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / 'training.log', mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_splits(data_dir):
    """Load and validate the original processed CBT data splits."""
    frames = {}
    expected_labels = set(Config.DISTORTION_LABELS)
    required = {
        'Id_Number',
        Config.TEXT_COLUMN,
        'label',
        'Dominant Distortion',
        Config.DISTORTED_TEXT_COLUMN,
        'secondary_distortion',
    }
    for split, filename in Config.SPLIT_FILES.items():
        path = Path(data_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing CBT split: {path}')
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f'{path.name} is missing columns: {sorted(missing)}')

        frame = frame.copy()
        frame[Config.TEXT_COLUMN] = (
            frame[Config.TEXT_COLUMN].fillna('').astype(str).str.strip())
        frame[Config.DISTORTED_TEXT_COLUMN] = (
            frame[Config.DISTORTED_TEXT_COLUMN]
            .fillna('')
            .astype(str)
            .str.strip()
        )
        empty_count = int((frame[Config.TEXT_COLUMN] == '').sum())
        if empty_count:
            raise ValueError(
                f'{path.name} contains {empty_count} empty documents')
        frame['label'] = pd.to_numeric(
            frame['label'], errors='raise').astype(int)
        unexpected = set(frame['label'].unique()) - expected_labels
        if unexpected:
            raise ValueError(
                f'{path.name} has unknown CBT labels: {sorted(unexpected)}')

        frame['has_distorted_excerpt'] = (
            frame[Config.DISTORTED_TEXT_COLUMN] != '')
        frame[Config.TOPIC_TEXT_COLUMN] = frame[
            Config.DISTORTED_TEXT_COLUMN].where(
                frame['has_distorted_excerpt'],
                frame[Config.TEXT_COLUMN],
            )
        frame[Config.TOPIC_TEXT_SOURCE_COLUMN] = np.where(
            frame['has_distorted_excerpt'],
            Config.DISTORTED_TEXT_COLUMN,
            f'{Config.TEXT_COLUMN} fallback',
        )

        frame['split'] = split
        frame['source_row'] = np.arange(len(frame), dtype=int)
        frames[split] = frame
    return frames


def is_research_boilerplate(document):
    """Conservatively identify recruitment boilerplate, not user stories."""
    marker_count = sum(
        bool(pattern.search(document)) for pattern in _RESEARCH_MARKERS)
    return marker_count >= 2 or (
        marker_count >= 1
        and bool(_SOLICITATION_MARKER.search(document))
    )


def prepare_fit_documents(frames):
    """Select distortion excerpts, then remove boilerplate and duplicates."""
    for frame in frames.values():
        frame['is_research_boilerplate'] = frame[
            Config.TOPIC_TEXT_COLUMN].map(is_research_boilerplate)
        frame['is_duplicate_for_topic_fit'] = False
        frame['excluded_from_topic_fit'] = False
        frame['topic_fit_exclusion_reason'] = ''

    candidates = pd.concat(
        [frames[split] for split in Config.FIT_SPLITS],
        ignore_index=True,
    )
    normalized = (
        candidates[Config.TOPIC_TEXT_COLUMN]
        .str.lower()
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    eligible_excerpt = candidates['has_distorted_excerpt']
    candidates['is_duplicate_for_topic_fit'] = False
    candidates.loc[
        eligible_excerpt, 'is_duplicate_for_topic_fit'
    ] = normalized[eligible_excerpt].duplicated(keep='first')
    excluded_missing_excerpt = ~eligible_excerpt
    excluded_boilerplate = (
        candidates['is_research_boilerplate']
        & Config.FILTER_BOILERPLATE
    )
    excluded_duplicate = (
        candidates['is_duplicate_for_topic_fit']
        & Config.DEDUPLICATE_FIT_DOCUMENTS
    )
    candidates['excluded_from_topic_fit'] = (
        excluded_missing_excerpt
        | excluded_boilerplate
        | excluded_duplicate
    )
    candidates['topic_fit_exclusion_reason'] = [
        '+'.join(reason for reason, applies in (
            ('missing_distorted_excerpt', bool(missing)),
            ('research_boilerplate', bool(boilerplate)),
            ('duplicate', bool(duplicate)),
        ) if applies)
        for missing, boilerplate, duplicate in zip(
            excluded_missing_excerpt,
            excluded_boilerplate,
            excluded_duplicate,
        )
    ]

    for split in Config.FIT_SPLITS:
        split_audit = candidates[candidates['split'] == split].set_index(
            'source_row')
        for column in (
            'is_duplicate_for_topic_fit',
            'excluded_from_topic_fit',
            'topic_fit_exclusion_reason',
        ):
            frames[split][column] = frames[split]['source_row'].map(
                split_audit[column])

    fit_frame = candidates[
        ~candidates['excluded_from_topic_fit']].copy().reset_index(drop=True)
    if len(fit_frame) < Config.MIN_TOPIC_SIZE * 2:
        raise ValueError(
            'Too few documents remain after BERTopic fit filtering: '
            f'{len(fit_frame)}')

    audit = {
        'candidate_documents': int(len(candidates)),
        'eligible_distorted_excerpts': int(eligible_excerpt.sum()),
        'included_documents': int(len(fit_frame)),
        'excluded_documents': int(
            candidates['excluded_from_topic_fit'].sum()),
        'missing_distorted_excerpt': int(
            excluded_missing_excerpt.sum()),
        'research_boilerplate_detected': int(
            candidates['is_research_boilerplate'].sum()),
        'duplicates_detected': int(
            candidates['is_duplicate_for_topic_fit'].sum()),
        'filter_research_boilerplate': Config.FILTER_BOILERPLATE,
        'deduplicate_fit_documents': Config.DEDUPLICATE_FIT_DOCUMENTS,
    }
    return fit_frame, audit


def build_topic_model(embedding_model):
    umap_model = UMAP(
        n_neighbors=Config.UMAP_N_NEIGHBORS,
        n_components=Config.UMAP_N_COMPONENTS,
        min_dist=0.0,
        metric='cosine',
        random_state=Config.SEED,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=Config.MIN_TOPIC_SIZE,
        min_samples=Config.HDBSCAN_MIN_SAMPLES,
        metric='euclidean',
        cluster_selection_method=Config.HDBSCAN_SELECTION_METHOD,
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=Config.VECTORIZER_MIN_DF,
        max_df=0.90,
    )
    guided_terms = (
        [term for topic in Config.SEED_TOPIC_LIST for term in topic]
        if Config.USE_GUIDED_TERMS
        else None
    )
    ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=True,
        reduce_frequent_words=True,
        seed_words=guided_terms,
        seed_multiplier=1.2,
    )
    representation_model = [
        KeyBERTInspired(),
        MaximalMarginalRelevance(diversity=0.5),
    ]
    seed_topic_list = (
        [list(topic) for topic in Config.SEED_TOPIC_LIST]
        if Config.USE_SEED_TOPICS
        else None
    )
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        seed_topic_list=seed_topic_list,
        nr_topics=Config.NR_TOPICS,
        top_n_words=Config.TOP_N_WORDS,
        calculate_probabilities=True,
        verbose=True,
        low_memory=True,
    )


def encode_documents(embedding_model, documents):
    return embedding_model.encode(
        documents,
        batch_size=Config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def assign_topics_to_splits(
        topic_model, frames, embeddings_by_split, fit_frame,
        fit_topics, fit_probabilities, logger):
    """Restore fitted rows and transform excluded/held-out rows in place."""
    fit_topics = np.asarray(fit_topics, dtype=int)
    fit_probabilities = np.asarray(fit_probabilities, dtype=float)
    if fit_probabilities.ndim == 1:
        fit_probabilities = fit_probabilities.reshape(-1, 1)
    if fit_probabilities.ndim != 2:
        raise ValueError('Expected a BERTopic probability matrix')

    fit_lookup = {
        (split, int(source_row)): offset
        for offset, (split, source_row) in enumerate(
            fit_frame[['split', 'source_row']].itertuples(
                index=False, name=None))
    }
    topics_by_split = {}
    probabilities_by_split = {}

    for split, frame in frames.items():
        if split not in Config.FIT_SPLITS:
            logger.info('Assigning learned topics to %s documents', split)
            topics, probabilities = topic_model.transform(
                frame[Config.TOPIC_TEXT_COLUMN].tolist(),
                embeddings_by_split[split],
            )
            topics_by_split[split] = np.asarray(topics, dtype=int)
            probabilities_by_split[split] = probabilities
            continue

        split_topics = np.full(len(frame), -1, dtype=int)
        split_probabilities = np.zeros(
            (len(frame), fit_probabilities.shape[1]), dtype=float)
        transform_positions = []
        fit_offsets = []
        fit_positions = []
        for position, source_row in enumerate(frame['source_row']):
            offset = fit_lookup.get((split, int(source_row)))
            if offset is None:
                transform_positions.append(position)
            else:
                fit_positions.append(position)
                fit_offsets.append(offset)
        split_topics[fit_positions] = fit_topics[fit_offsets]
        split_probabilities[fit_positions] = fit_probabilities[fit_offsets]

        if transform_positions:
            logger.info(
                'Assigning learned topics to %s excluded %s documents',
                len(transform_positions), split)
            transformed_topics, transformed_probabilities = (
                topic_model.transform(
                    frame.iloc[transform_positions][
                        Config.TOPIC_TEXT_COLUMN].tolist(),
                    embeddings_by_split[split][transform_positions],
                )
            )
            transformed_probabilities = np.asarray(
                transformed_probabilities, dtype=float)
            if transformed_probabilities.ndim == 1:
                transformed_probabilities = transformed_probabilities.reshape(
                    -1, 1)
            if transformed_probabilities.shape[1] != (
                    split_probabilities.shape[1]):
                raise ValueError(
                    'Fitted and transformed probability columns differ')
            split_topics[transform_positions] = np.asarray(
                transformed_topics, dtype=int)
            split_probabilities[transform_positions] = (
                transformed_probabilities)

        topics_by_split[split] = split_topics
        probabilities_by_split[split] = split_probabilities

    return topics_by_split, probabilities_by_split


def _package_assignments(frame, topics, probabilities, topic_names):
    columns = [
        'split',
        'source_row',
        'Id_Number',
        Config.TEXT_COLUMN,
        Config.DISTORTED_TEXT_COLUMN,
        Config.TOPIC_TEXT_COLUMN,
        Config.TOPIC_TEXT_SOURCE_COLUMN,
        'has_distorted_excerpt',
        'label',
        'Dominant Distortion',
        'secondary_distortion',
    ]
    columns.extend(
        column
        for column in (
            'is_research_boilerplate',
            'is_duplicate_for_topic_fit',
            'excluded_from_topic_fit',
            'topic_fit_exclusion_reason',
        )
        if column in frame.columns
    )
    assigned = frame[columns].copy()
    assigned.insert(
        0,
        'document_id',
        [
            f'{split}_{identifier}'
            for split, identifier in zip(
                assigned['split'], assigned['Id_Number'])
        ],
    )
    assigned['distortion_type'] = assigned['label'].map(
        Config.DISTORTION_LABELS)
    assigned['distortion_status'] = np.where(
        assigned['label'] == 0,
        'No Distortion',
        'Has Distortion',
    )
    assigned['topic'] = np.asarray(topics, dtype=int)
    assigned['is_outlier'] = assigned['topic'] == -1
    assigned['topic_name'] = assigned['topic'].map(topic_names).fillna(
        'Outlier')
    assigned['topic_probability'] = assigned_probabilities(
        topics, probabilities)
    return assigned


def _save_probability_matrix(
        probabilities_by_split, output_dir, probability_topic_order):
    arrays = [
        np.asarray(probabilities_by_split[split])
        for split in Config.SPLIT_FILES
    ]
    if not arrays or any(array.ndim != 2 for array in arrays):
        return False
    if len({array.shape[1] for array in arrays}) != 1:
        return False
    if arrays[0].shape[1] != len(probability_topic_order):
        return False
    np.save(
        Path(output_dir) / 'document_topic_probabilities.npy',
        np.vstack(arrays),
    )
    save_json({
        'row_order': (
            'Same row order as document_topics.csv '
            '(train, validation, test)'),
        'column_topic_ids': probability_topic_order,
        'outlier_topic_has_dedicated_column': False,
    }, Path(output_dir) / 'probability_matrix_metadata.json')
    return True


def _topic_label_summary(document_topics, column):
    summary = (
        document_topics.groupby(
            ['topic', 'topic_name', column], as_index=False,
            dropna=False,
        )
        .size()
        .rename(columns={'size': 'document_count'}))
    summary['percentage_within_topic'] = (
        summary['document_count']
        / summary.groupby('topic')['document_count'].transform('sum'))
    return summary


def _dataset_summary(frames):
    split_summaries = {}
    for split, frame in frames.items():
        counts = frame['label'].value_counts().sort_index()
        split_summaries[split] = {
            'documents': int(len(frame)),
            'documents_with_distorted_excerpt': int(
                frame['has_distorted_excerpt'].sum()),
            'topic_text_source_counts': {
                str(source): int(count)
                for source, count in frame[
                    Config.TOPIC_TEXT_SOURCE_COLUMN].value_counts().items()
            },
            'label_counts': {
                str(label): int(counts.get(label, 0))
                for label in Config.DISTORTION_LABELS
            },
            'binary_counts': {
                'No Distortion': int((frame['label'] == 0).sum()),
                'Has Distortion': int((frame['label'] != 0).sum()),
            },
        }
    return {
        'question_text_column': Config.TEXT_COLUMN,
        'fit_text_column': Config.DISTORTED_TEXT_COLUMN,
        'topic_text_column': Config.TOPIC_TEXT_COLUMN,
        'fallback_text_column': Config.TEXT_COLUMN,
        'distortion_labels': {
            str(key): value
            for key, value in Config.DISTORTION_LABELS.items()
        },
        'splits': split_summaries,
    }


def _package_versions():
    packages = (
        'bertopic',
        'sentence-transformers',
        'umap-learn',
        'hdbscan',
        'scikit-learn',
        'pandas',
        'numpy',
        'torch',
        'transformers',
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = build_logger(Config.OUTPUT_DIR)
    set_reproducibility(Config.SEED)

    logger.info('CBT Header BERTopic training started')
    logger.info('Embedding model: %s', Config.EMBEDDING_MODEL)
    logger.info(
        'Topic-forming text: %s excerpts | fallback for reporting: %s',
        Config.DISTORTED_TEXT_COLUMN,
        Config.TEXT_COLUMN,
    )
    logger.info('Device: %s', Config.device())
    logger.info(
        'Fit splits: %s | transformed splits: %s',
        Config.FIT_SPLITS,
        Config.TRANSFORM_SPLITS,
    )

    frames = load_splits(Config.DATA_DIR)
    for split, frame in frames.items():
        logger.info('%s rows: %s', split, f'{len(frame):,}')
    save_json(_dataset_summary(frames),
              Config.OUTPUT_DIR / 'dataset_summary.json')

    embedding_model = SentenceTransformer(
        Config.EMBEDDING_MODEL,
        device=Config.device(),
    )
    embeddings_by_split = {}
    for split, frame in frames.items():
        logger.info('Encoding %s documents', split)
        embeddings_by_split[split] = encode_documents(
            embedding_model,
            frame[Config.TOPIC_TEXT_COLUMN].tolist(),
        )

    fit_frame, fit_filter_summary = prepare_fit_documents(frames)
    logger.info(
        'Topic-fit quality filter: %s included, %s excluded '
        '(%s without excerpts, %s recruitment boilerplate, '
        '%s duplicates detected)',
        fit_filter_summary['included_documents'],
        fit_filter_summary['excluded_documents'],
        fit_filter_summary['missing_distorted_excerpt'],
        fit_filter_summary['research_boilerplate_detected'],
        fit_filter_summary['duplicates_detected'],
    )
    save_json(
        fit_filter_summary,
        Config.OUTPUT_DIR / 'fit_filter_summary.json',
    )
    fit_documents = fit_frame[Config.TOPIC_TEXT_COLUMN].tolist()
    fit_embeddings = np.vstack([
        embeddings_by_split[split][
            fit_frame.loc[
                fit_frame['split'] == split, 'source_row'
            ].to_numpy(dtype=int)
        ]
        for split in Config.FIT_SPLITS
        if (fit_frame['split'] == split).any()
    ])
    topic_model = build_topic_model(embedding_model)
    logger.info('Fitting BERTopic on %s training documents', len(fit_frame))
    fit_topics, fit_probabilities = topic_model.fit_transform(
        fit_documents,
        fit_embeddings,
    )

    topics_by_split, probabilities_by_split = assign_topics_to_splits(
        topic_model,
        frames,
        embeddings_by_split,
        fit_frame,
        fit_topics,
        fit_probabilities,
        logger,
    )

    topic_info = topic_model.get_topic_info().copy()
    topic_names = dict(zip(topic_info['Topic'], topic_info['Name']))
    document_topics = pd.concat([
        _package_assignments(
            frames[split],
            topics_by_split[split],
            probabilities_by_split[split],
            topic_names,
        )
        for split in Config.SPLIT_FILES
    ], ignore_index=True)

    assigned_counts = (
        document_topics.groupby('topic').size()
        .rename('Assigned_Count_All_Splits'))
    topic_info = topic_info.merge(
        assigned_counts,
        how='left',
        left_on='Topic',
        right_index=True,
    )
    topic_info['Assigned_Count_All_Splits'] = (
        topic_info['Assigned_Count_All_Splits'].fillna(0).astype(int))

    topic_metrics = calculate_topic_metrics(
        topic_model,
        fit_topics,
        fit_embeddings,
        Config.SEED,
    )
    topic_metrics['fit_splits'] = list(Config.FIT_SPLITS)
    topic_metrics['transformed_splits'] = list(Config.TRANSFORM_SPLITS)

    document_topics.to_csv(
        Config.OUTPUT_DIR / 'document_topics.csv', index=False)
    topic_info.to_csv(
        Config.OUTPUT_DIR / 'topic_info.csv', index=False)
    topic_terms_frame(topic_model).to_csv(
        Config.OUTPUT_DIR / 'topic_terms.csv', index=False)
    _topic_label_summary(
        document_topics, 'distortion_status').to_csv(
            Config.OUTPUT_DIR / 'topic_by_distortion_status.csv',
            index=False,
        )
    _topic_label_summary(
        document_topics, 'distortion_type').to_csv(
            Config.OUTPUT_DIR / 'topic_by_distortion_type.csv',
            index=False,
        )

    probability_matrix_saved = _save_probability_matrix(
        probabilities_by_split,
        Config.OUTPUT_DIR,
        sorted(
            int(topic_id)
            for topic_id in topic_model.get_topics()
            if int(topic_id) != -1
        ),
    )
    save_static_plots(
        document_topics,
        topic_info,
        Config.OUTPUT_DIR,
        Config.MAX_PLOT_TOPICS,
    )
    interactive_plots = save_interactive_plots(
        topic_model,
        Config.OUTPUT_DIR,
        Config.MAX_PLOT_TOPICS,
    )

    model_dir = Config.OUTPUT_DIR / 'topic_model'
    topic_model.save(
        model_dir,
        serialization='safetensors',
        save_ctfidf=True,
        save_embedding_model=Config.EMBEDDING_MODEL,
    )
    save_json(topic_metrics, Config.OUTPUT_DIR / 'topic_metrics.json')
    save_json({
        'task': 'CBT Header unsupervised topic discovery',
        'embedding_model': Config.EMBEDDING_MODEL,
        'embedding_device': Config.device(),
        'embedding_batch_size': Config.EMBEDDING_BATCH_SIZE,
        'fit_splits': list(Config.FIT_SPLITS),
        'transformed_splits': list(Config.TRANSFORM_SPLITS),
        'topic_text_column': Config.TOPIC_TEXT_COLUMN,
        'fit_text_column': Config.DISTORTED_TEXT_COLUMN,
        'fallback_text_column': Config.TEXT_COLUMN,
        'labels_used_for_training': False,
        'distorted_excerpt_annotations_used_for_corpus_selection': True,
        'fit_filter': fit_filter_summary,
        'min_topic_size': Config.MIN_TOPIC_SIZE,
        'topic_reduction': Config.NR_TOPICS,
        'guided_representation_terms_enabled': Config.USE_GUIDED_TERMS,
        'guided_seed_topics_enabled': Config.USE_SEED_TOPICS,
        'seed_topic_list': (
            [list(topic) for topic in Config.SEED_TOPIC_LIST]
            if Config.USE_SEED_TOPICS else []
        ),
        'top_n_words': Config.TOP_N_WORDS,
        'representation': {
            'pipeline': ['KeyBERTInspired', 'MaximalMarginalRelevance'],
            'mmr_diversity': 0.5,
        },
        'ctfidf': {
            'bm25_weighting': True,
            'reduce_frequent_words': True,
            'guided_term_multiplier': (
                1.2 if Config.USE_GUIDED_TERMS else None),
        },
        'vectorizer': {
            'ngram_range': [1, 2],
            'min_df': Config.VECTORIZER_MIN_DF,
            'max_df': 0.90,
            'stop_words': 'english',
        },
        'umap': {
            'n_neighbors': Config.UMAP_N_NEIGHBORS,
            'n_components': Config.UMAP_N_COMPONENTS,
            'min_dist': 0.0,
            'metric': 'cosine',
        },
        'hdbscan': {
            'min_cluster_size': Config.MIN_TOPIC_SIZE,
            'min_samples': Config.HDBSCAN_MIN_SAMPLES,
            'metric': 'euclidean',
            'cluster_selection_method': (
                Config.HDBSCAN_SELECTION_METHOD),
            'prediction_data': True,
        },
        'probability_matrix_saved': probability_matrix_saved,
        'interactive_plots': interactive_plots,
        'package_versions': _package_versions(),
        'seed': Config.SEED,
    }, Config.OUTPUT_DIR / 'run_config.json')

    logger.info(
        'Complete: %s topics, %.2f%% training outliers',
        topic_metrics['topic_count_excluding_outliers'],
        topic_metrics['outlier_rate'] * 100,
    )
    logger.info('Outputs saved to %s', Config.OUTPUT_DIR)


if __name__ == '__main__':
    main()
