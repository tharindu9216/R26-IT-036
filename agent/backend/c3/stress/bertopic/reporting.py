"""Reporting helpers for the Stress Header BERTopic pipeline."""

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
            # Outlier topic -1 has no dedicated probability column.
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
    """Fraction of unique terms among the top words of non-outlier topics."""
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


def save_static_plots(document_topics, topic_info, output_dir):
    """Save lightweight PNG summaries that do not require a browser."""
    output_dir = Path(output_dir)
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    non_outlier_info = topic_info[topic_info['Topic'] != -1].copy()
    non_outlier_info = non_outlier_info.sort_values(
        'Count', ascending=False).head(20)
    if not non_outlier_info.empty:
        plt.figure(figsize=(12, 7))
        sns.barplot(
            data=non_outlier_info,
            x='Count',
            y='Name',
            color='#4472C4',
        )
        plt.title('Stress BERTopic — Largest Training Topics')
        plt.xlabel('Documents')
        plt.ylabel('Topic')
        plt.tight_layout()
        plt.savefig(
            plot_dir / 'topic_sizes.png', dpi=200, bbox_inches='tight')
        plt.close()

    label_counts = pd.crosstab(
        document_topics['topic'],
        document_topics['stress_label'],
    )
    if not label_counts.empty:
        label_percentages = label_counts.div(
            label_counts.sum(axis=1), axis=0)
        plt.figure(figsize=(9, max(5, len(label_counts) * 0.35)))
        sns.heatmap(
            label_percentages,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            vmin=0,
            vmax=1,
        )
        plt.title('Stress-label Distribution Within Each Topic')
        plt.xlabel('Stress label')
        plt.ylabel('Topic')
        plt.tight_layout()
        plt.savefig(
            plot_dir / 'topic_stress_distribution.png',
            dpi=200,
            bbox_inches='tight',
        )
        plt.close()


def save_interactive_plots(topic_model, output_dir, max_topics=20):
    """Save BERTopic Plotly visualizations as standalone HTML files."""
    output_dir = Path(output_dir)
    plot_dir = output_dir / 'plots'
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
