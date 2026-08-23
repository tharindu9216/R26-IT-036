"""Reporting helpers for the CBT Header BERTopic pipeline."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score


def assigned_probabilities(topics, probabilities):
    """Extract one confidence value for every assigned document topic."""
    topic_array = np.asarray(topics, dtype=int)
    if probabilities is None:
        return np.full(len(topic_array), np.nan, dtype=float)

    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.ndim == 1:
        return probability_array
    if probability_array.ndim != 2:
        raise ValueError(
            'BERTopic probabilities must be a vector or matrix')

    assigned = np.zeros(len(topic_array), dtype=float)
    for index, topic in enumerate(topic_array):
        if 0 <= topic < probability_array.shape[1]:
            assigned[index] = probability_array[index, topic]
        else:
            # Outlier topic -1 does not have its own probability column.
            assigned[index] = probability_array[index].max(initial=0.0)
    return assigned


def topic_terms_frame(topic_model):
    """Flatten BERTopic's topic-word representation into a CSV table."""
    rows = []
    for topic_id in sorted(topic_model.get_topics()):
        for rank, (term, weight) in enumerate(
                topic_model.get_topic(topic_id) or [], start=1):
            rows.append({
                'topic': int(topic_id),
                'rank': rank,
                'term': term,
                'c_tf_idf_weight': float(weight),
            })
    return pd.DataFrame(rows)


def topic_diversity(topic_model, top_n_words=10):
    """Fraction of unique terms among non-outlier topics' top words."""
    words = []
    for topic_id in sorted(topic_model.get_topics()):
        if topic_id == -1:
            continue
        words.extend(
            term
            for term, _ in (topic_model.get_topic(topic_id) or [])[
                :top_n_words]
        )
    return float(len(set(words)) / len(words)) if words else 0.0


def calculate_topic_metrics(topic_model, topics, embeddings, seed):
    """Calculate transparent, label-free topic-model diagnostics."""
    topic_array = np.asarray(topics, dtype=int)
    non_outlier = topic_array != -1
    topic_ids = sorted(set(topic_array[non_outlier].tolist()))
    metrics = {
        'documents': int(len(topic_array)),
        'topic_count_excluding_outliers': int(len(topic_ids)),
        'outlier_documents': int((~non_outlier).sum()),
        'outlier_rate': float((~non_outlier).mean()),
        'topic_diversity_top_10': topic_diversity(topic_model, 10),
        'silhouette_cosine': None,
    }
    if len(topic_ids) >= 2 and non_outlier.sum() > len(topic_ids):
        sample_size = min(2000, int(non_outlier.sum()))
        metrics['silhouette_cosine'] = float(silhouette_score(
            np.asarray(embeddings)[non_outlier],
            topic_array[non_outlier],
            metric='cosine',
            sample_size=sample_size,
            random_state=seed,
        ))
    return metrics


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f'Cannot serialize {type(value).__name__}')

    path.write_text(json.dumps(data, indent=2, default=convert))


def _save_distribution_heatmap(
        documents, column, title, filename, output_dir, width):
    counts = pd.crosstab(documents['topic'], documents[column])
    if counts.empty:
        return
    percentages = counts.div(counts.sum(axis=1), axis=0)
    plt.figure(figsize=(width, max(5, len(counts) * 0.38)))
    sns.heatmap(
        percentages,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        vmin=0,
        vmax=1,
    )
    plt.title(title)
    plt.xlabel(column.replace('_', ' ').title())
    plt.ylabel('Topic')
    plt.tight_layout()
    plt.savefig(
        Path(output_dir) / filename,
        dpi=200,
        bbox_inches='tight',
    )
    plt.close()


def save_static_plots(
        document_topics, topic_info, output_dir, max_topics=20):
    """Save topic-size and distortion-distribution PNG reports."""
    plot_dir = Path(output_dir) / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    non_outlier_info = topic_info[topic_info['Topic'] != -1].copy()
    non_outlier_info = non_outlier_info.sort_values(
        'Count', ascending=False).head(max_topics)
    if non_outlier_info.empty:
        return

    plt.figure(figsize=(12, max(6, len(non_outlier_info) * 0.4)))
    sns.barplot(
        data=non_outlier_info,
        x='Count',
        y='Name',
        color='#4472C4',
    )
    plt.title('CBT BERTopic — Largest Training Topics')
    plt.xlabel('Training documents')
    plt.ylabel('Topic')
    plt.tight_layout()
    plt.savefig(
        plot_dir / 'topic_sizes.png', dpi=200, bbox_inches='tight')
    plt.close()

    displayed_topic_ids = set(non_outlier_info['Topic'].astype(int))
    displayed_documents = document_topics[
        document_topics['topic'].isin(displayed_topic_ids)
    ]
    _save_distribution_heatmap(
        displayed_documents,
        'distortion_status',
        'Binary Distortion Status Within Each Topic',
        'topic_distortion_status_distribution.png',
        plot_dir,
        width=9,
    )
    _save_distribution_heatmap(
        displayed_documents,
        'distortion_type',
        'Original CBT Distortion Types Within Each Topic',
        'topic_distortion_type_distribution.png',
        plot_dir,
        width=17,
    )


def save_interactive_plots(topic_model, output_dir, max_topics=20):
    """Save BERTopic Plotly visualizations as standalone HTML files."""
    plot_dir = Path(output_dir) / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    factories = {
        'topics.html': lambda: topic_model.visualize_topics(),
        'topic_barchart.html': lambda: topic_model.visualize_barchart(
            top_n_topics=max_topics),
        'topic_heatmap.html': lambda: topic_model.visualize_heatmap(
            top_n_topics=max_topics),
        'topic_hierarchy.html': lambda: topic_model.visualize_hierarchy(
            top_n_topics=max_topics),
    }
    for filename, factory in factories.items():
        try:
            factory().write_html(plot_dir / filename)
            outcomes[filename] = 'saved'
        except (ValueError, TypeError, RuntimeError) as error:
            # Some plots require at least two non-outlier topics.
            outcomes[filename] = f'skipped: {error}'
    return outcomes
