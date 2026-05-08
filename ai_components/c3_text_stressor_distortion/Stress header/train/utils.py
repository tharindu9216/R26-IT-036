
import os, json, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
    roc_auc_score, matthews_corrcoef,
)


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(labels, preds, probs=None):
    m = {
        'accuracy'    : accuracy_score(labels, preds),
        'f1_macro'    : f1_score(labels, preds, average='macro',    zero_division=0),
        'f1_weighted' : f1_score(labels, preds, average='weighted', zero_division=0),
        'precision'   : precision_score(labels, preds, average='macro', zero_division=0),
        'recall'      : recall_score(labels, preds,    average='macro', zero_division=0),
        'mcc'         : matthews_corrcoef(labels, preds),
    }
    if probs is not None:
        try:
            m['roc_auc'] = roc_auc_score(labels, probs)
        except Exception:
            m['roc_auc'] = 0.0
    return m


def compute_head1b_metrics(labels, preds):
    return {
        'head1b_accuracy'    : accuracy_score(labels, preds),
        'head1b_f1_macro'    : f1_score(labels, preds, average='macro', zero_division=0),
    }


# =============================================================================
# Save / Load
# =============================================================================
def save_results(data, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    def _cvt(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return o

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=_cvt)


# =============================================================================
# Logger
# =============================================================================
class Logger:
    def __init__(self, log_path=None):
        self.t0 = time.time()
        self.log_path = log_path
        if log_path:
            os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
            open(log_path, 'w').close()

    def log(self, msg=''):
        e    = time.time() - self.t0
        line = f'[{int(e//3600):02d}:{int((e%3600)//60):02d}:{int(e%60):02d}] {msg}'
        print(line)
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(line + '\n')

    def section(self, title):
        self.log('=' * 65)
        self.log(f'  {title}')
        self.log('=' * 65)


# =============================================================================
# Visualizations
# =============================================================================
def plot_confusion_matrices(all_results, output_dir, class_names=None):
    """Save confusion matrix for each model."""
    if class_names is None:
        class_names = ['Not Stressed', 'Stressed']

    models = [(n, r) for n, r in all_results.items()
              if 'test_preds' in r and 'test_labels' in r]
    if not models:
        return

    ncols = min(3, len(models))
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for idx, (name, res) in enumerate(models):
        cm = confusion_matrix(res['test_labels'], res['test_preds'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx],
                    xticklabels=class_names, yticklabels=class_names,
                    cmap='Blues', cbar=False)
        f1  = res.get('test_metrics', {}).get('f1_macro', 0)
        axes[idx].set_title(f'{name}\nF1={f1:.4f}', fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')

    for ax in axes[len(models):]:
        ax.set_visible(False)

    plt.suptitle('Confusion Matrices — All Models', fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_training_curves(all_results, output_dir):
    """Save training curves (val loss + val F1) per transformer model."""
    transformer_results = {
        n: r for n, r in all_results.items()
        if 'fold_results' in r
    }
    if not transformer_results:
        return

    fig, axes = plt.subplots(
        len(transformer_results), 2,
        figsize=(14, 4 * len(transformer_results))
    )
    if len(transformer_results) == 1:
        axes = axes.reshape(1, -1)

    colors = ['#4472C4', '#70AD47', '#7030A0', '#FF0000', '#FFA500',
              '#00A6A6', '#6A4C93']

    for row, (name, res) in enumerate(transformer_results.items()):
        # Average history across folds
        all_losses = [f['history']['train_loss'] for f in res['fold_results']
                      if 'history' in f]
        all_f1s    = [f['history']['val_f1']    for f in res['fold_results']
                      if 'history' in f]

        if not all_losses:
            continue

        max_len  = max(len(x) for x in all_losses)
        avg_loss = np.mean([x + [x[-1]] * (max_len - len(x))
                            for x in all_losses], axis=0)
        avg_f1   = np.mean([x + [x[-1]] * (max_len - len(x))
                            for x in all_f1s],   axis=0)
        epochs   = range(1, len(avg_loss) + 1)

        axes[row, 0].plot(epochs, avg_loss, color=colors[row], marker='o')
        axes[row, 0].set_title(f'{name} — Avg Train Loss')
        axes[row, 0].set_xlabel('Epoch')
        axes[row, 0].set_ylabel('Loss')
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(epochs, avg_f1, color=colors[row], marker='o')
        axes[row, 1].set_title(f'{name} — Avg Val F1')
        axes[row, 1].set_xlabel('Epoch')
        axes[row, 1].set_ylabel('F1 Macro')
        axes[row, 1].grid(True, alpha=0.3)

    plt.suptitle('Training Curves (averaged across folds)', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_comparison_dashboard(all_results, output_dir):
    """Bar chart comparing all models on key metrics."""
    names    = list(all_results.keys())
    metrics  = ['accuracy', 'f1_macro', 'f1_weighted', 'mcc']
    labels   = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'MCC']
    colors   = ['#4472C4', '#ED7D31', '#70AD47', '#7030A0', '#FF0000',
                '#00A6A6', '#6A4C93']

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for ax, metric, label in zip(axes, metrics, labels):
        vals = [all_results[n].get('test_metrics', {}).get(metric, 0)
                for n in names]
        bars = ax.bar(names, vals,
                      color=colors[:len(names)], alpha=0.85,
                      edgecolor='white')
        ax.set_title(label, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=30)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Model Comparison Dashboard — Test Set', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'comparison_dashboard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# =============================================================================
# Final comparison table
# =============================================================================
def print_final_table(all_results):
    print('\n' + '=' * 100)
    print('  FINAL COMPARISON — TEST SET')
    print('=' * 100)
    print(f'{"Model":22s} {"Accuracy":>10s} {"F1 Macro":>10s} '
          f'{"F1 Wt":>8s} {"Precision":>10s} {"Recall":>8s} '
          f'{"MCC":>7s} {"ROC-AUC":>9s}')
    print('-' * 100)

    ranked = sorted(all_results.items(),
                    key=lambda x: x[1].get('test_metrics', {}).get('f1_macro', 0),
                    reverse=True)

    for rank, (name, res) in enumerate(ranked, 1):
        m      = res.get('test_metrics', {})
        marker = '  ←BEST' if rank == 1 else ''
        print(
            f'{name:22s} '
            f'{m.get("accuracy",    0):>10.4f} '
            f'{m.get("f1_macro",    0):>10.4f} '
            f'{m.get("f1_weighted", 0):>8.4f} '
            f'{m.get("precision",   0):>10.4f} '
            f'{m.get("recall",      0):>8.4f} '
            f'{m.get("mcc",         0):>7.4f} '
            f'{m.get("roc_auc",     0):>9.4f}'
            f'{marker}'
        )
    print('=' * 100)
