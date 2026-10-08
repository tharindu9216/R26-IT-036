"""Leakage-safe traditional ML baselines for binary CBT classification."""

import os
import pickle

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from config import Config
from training import select_binary_threshold
from utils import compute_metrics


BASELINE_NAMES = {
    'LR': 'TF-IDF + Logistic Regression',
    'SVM': 'TF-IDF + Calibrated Linear SVM',
}


def _build_classifier(name):
    """Construct one class-weighted binary baseline."""
    if name == 'LR':
        parameters = {
            **Config.LR_PARAMS,
            'random_state': Config.SEED,
        }
        return LogisticRegression(**parameters)
    if name == 'SVM':
        parameters = {
            **Config.SVM_PARAMS,
            'random_state': Config.SEED,
        }
        base = LinearSVC(**parameters)
        return CalibratedClassifierCV(base, cv=3)
    raise ValueError(
        f'Unknown binary baseline {name!r}; expected LR or SVM')


def _ordered_binary_probabilities(classifier, features):
    """Return predict_proba columns in fixed [No, Distortion] order."""
    raw = np.asarray(classifier.predict_proba(features), dtype=float)
    ordered = np.zeros((raw.shape[0], 2), dtype=float)
    for source_column, class_id in enumerate(classifier.classes_):
        ordered[:, int(class_id)] = raw[:, source_column]
    return ordered


def _binary_labels(dataframe):
    return (
        dataframe['label'].to_numpy(dtype=int) != 0).astype(int)


def train_binary_baseline(
        name, train_df, val_df, test_df, objective='f1_macro',
        logger=None):
    """Cross-validate, threshold-select, refit, and test one ML baseline.

    The TF-IDF vocabulary and classifier are fitted independently inside each
    outer fold. The decision threshold is selected from training-only OOF
    probabilities. The final artifact is then refitted on train+validation,
    while the held-out test set is used only for final evaluation.
    """
    if name not in BASELINE_NAMES:
        raise ValueError(
            f'Unknown binary baseline {name!r}; expected LR or SVM')

    display_name = BASELINE_NAMES[name]
    text_column = 'Patient Question'
    train_texts = train_df[text_column].fillna('').to_numpy(dtype=object)
    train_labels = _binary_labels(train_df)
    test_labels = _binary_labels(test_df)
    if set(np.unique(train_labels)) != {0, 1}:
        raise ValueError(
            'Binary baseline training data must contain both classes')

    if logger:
        logger.section(f'Binary Baseline: {display_name}')
        logger.log(
            f'  OOF selection: {Config.N_FOLDS} folds on real train rows')

    splitter = StratifiedKFold(
        n_splits=Config.N_FOLDS,
        shuffle=True,
        random_state=Config.SEED,
    )
    oof_probabilities = np.zeros((len(train_df), 2), dtype=float)
    fold_indices = []

    for fold, (fit_indices, holdout_indices) in enumerate(
            splitter.split(train_texts, train_labels), start=1):
        vectorizer = TfidfVectorizer(**Config.TFIDF_PARAMS)
        fit_vectors = vectorizer.fit_transform(
            train_texts[fit_indices].tolist())
        holdout_vectors = vectorizer.transform(
            train_texts[holdout_indices].tolist())
        classifier = _build_classifier(name)
        classifier.fit(fit_vectors, train_labels[fit_indices])
        fold_probabilities = _ordered_binary_probabilities(
            classifier, holdout_vectors)
        oof_probabilities[holdout_indices] = fold_probabilities
        fold_indices.append((fold, holdout_indices))

    threshold_result, threshold_curve = select_binary_threshold(
        train_labels,
        oof_probabilities[:, 1],
        objective=objective,
    )
    threshold = threshold_result['threshold']
    oof_predictions = (
        oof_probabilities[:, 1] >= threshold).astype(int)
    oof_metrics = compute_metrics(
        train_labels,
        oof_predictions,
        oof_probabilities,
        num_classes=2,
    )
    fold_results = []
    for fold, holdout_indices in fold_indices:
        fold_predictions = (
            oof_probabilities[holdout_indices, 1] >= threshold).astype(int)
        fold_metrics = compute_metrics(
            train_labels[holdout_indices],
            fold_predictions,
            oof_probabilities[holdout_indices],
            num_classes=2,
        )
        fold_results.append({
            'fold': fold,
            'f1_macro': fold_metrics['f1_macro'],
            'balanced_accuracy': fold_metrics['balanced_accuracy'],
            'mcc': fold_metrics['mcc'],
        })

    if logger:
        logger.log(
            f'  OOF threshold={threshold:.2f} | '
            f'Balanced Acc={oof_metrics["balanced_accuracy"]:.4f} | '
            f'F1={oof_metrics["f1_macro"]:.4f} | '
            f'MCC={oof_metrics["mcc"]:.4f}')

    full_texts = (
        train_df[text_column].fillna('').tolist()
        + val_df[text_column].fillna('').tolist()
    )
    full_labels = np.concatenate([
        _binary_labels(train_df),
        _binary_labels(val_df),
    ])
    test_texts = test_df[text_column].fillna('').tolist()
    final_vectorizer = TfidfVectorizer(**Config.TFIDF_PARAMS)
    full_vectors = final_vectorizer.fit_transform(full_texts)
    test_vectors = final_vectorizer.transform(test_texts)
    final_classifier = _build_classifier(name)
    final_classifier.fit(full_vectors, full_labels)
    test_probabilities = _ordered_binary_probabilities(
        final_classifier, test_vectors)
    test_predictions = (
        test_probabilities[:, 1] >= threshold).astype(int)
    test_metrics = compute_metrics(
        test_labels,
        test_predictions,
        test_probabilities,
        num_classes=2,
    )

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(
        Config.OUTPUT_DIR, f'binary_baseline_{name}.pkl')
    with open(model_path, 'wb') as handle:
        pickle.dump({
            'vectorizer': final_vectorizer,
            'model': final_classifier,
            'input_text_column': text_column,
            'class_order': [0, 1],
            'class_names': ['No Distortion', 'Distortion'],
            'binary_threshold': threshold,
        }, handle)

    if logger:
        logger.log(
            f'  Test Balanced Acc={test_metrics["balanced_accuracy"]:.4f} | '
            f'F1={test_metrics["f1_macro"]:.4f} | '
            f'MCC={test_metrics["mcc"]:.4f} | '
            f'PR-AUC={test_metrics.get("pr_auc", 0):.4f} | '
            f'ROC-AUC={test_metrics.get("roc_auc", 0):.4f}')
        logger.log(f'  Saved → {model_path}')

    return {
        'name': display_name,
        'family': 'traditional_ml_baseline',
        'text_col': text_column,
        'binary_threshold': threshold,
        'threshold_selection': threshold_result,
        'threshold_curve': threshold_curve,
        'selection_source': (
            'fixed-hyperparameter 5-fold OOF real training split'),
        'oof_metrics': oof_metrics,
        'cv_fold_results': fold_results,
        'model_path': model_path,
        'test_metrics': test_metrics,
        'test_preds': test_predictions.tolist(),
        'test_labels': test_labels.tolist(),
        'test_probs': test_probabilities.tolist(),
    }


def run_binary_baselines(
        train_df, val_df, test_df, objective='f1_macro', logger=None):
    """Train every configured binary ML baseline."""
    return {
        name: train_binary_baseline(
            name,
            train_df,
            val_df,
            test_df,
            objective=objective,
            logger=logger,
        )
        for name in Config.ML_BASELINES
    }
