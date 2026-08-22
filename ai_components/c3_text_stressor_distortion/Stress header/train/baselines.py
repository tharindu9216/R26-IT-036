
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from config import Config
from utils  import compute_metrics


def train_baseline(name, full_df, test_df, logger=None):
    """
    Train one ML baseline.
    Uses raw 'text' column — exactly as saved by preprocessing.
    Trained on train+val combined (full_df) per spec.
    """
    label = ('TF-IDF + Logistic Regression' if name == 'LR'
             else 'TF-IDF + LinearSVC')

    if logger:
        logger.section(f'Baseline: {label}')

    # Raw text (preprocessing saves this as 'text' column)
    X_train = full_df['text'].fillna('').tolist()
    X_test  = test_df['text'].fillna('').tolist()
    y_train = full_df['label'].values
    y_test  = test_df['label'].values

    # TF-IDF
    vectorizer = TfidfVectorizer(**Config.TFIDF_PARAMS)
    X_train_v  = vectorizer.fit_transform(X_train)
    X_test_v   = vectorizer.transform(X_test)

    if logger:
        logger.log(f'  TF-IDF vocab : {len(vectorizer.vocabulary_):,} features')
        logger.log(f'  Train size   : {X_train_v.shape}')

    # Model
    if name == 'LR':
        clf = LogisticRegression(**Config.LR_PARAMS)
    else:
        # LinearSVC — faster than SVC for large datasets
        # Wrap with CalibratedClassifierCV to get probabilities
        base = LinearSVC(**Config.SVM_PARAMS)
        clf  = CalibratedClassifierCV(base, cv=3)

    clf.fit(X_train_v, y_train)

    # Evaluate
    preds = clf.predict(X_test_v)
    probs = clf.predict_proba(X_test_v)[:, 1]
    metrics = compute_metrics(y_test.tolist(), preds.tolist(), probs.tolist())

    if logger:
        logger.log(f'  Test Accuracy : {metrics["accuracy"]:.4f}')
        logger.log(f'  Test F1 Macro : {metrics["f1_macro"]:.4f}')
        logger.log(f'  Test MCC      : {metrics["mcc"]:.4f}')
        logger.log(f'  Test ROC-AUC  : {metrics.get("roc_auc", 0):.4f}')

    # Save
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(Config.OUTPUT_DIR, f'baseline_{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'model': clf}, f)
    if logger:
        logger.log(f'  Saved → {path}')

    return {
        'name'        : label,
        'test_metrics': metrics,
        'test_preds'  : preds.tolist(),
        'test_labels' : y_test.tolist(),
        'test_probs'  : probs.tolist(),
        'model_path'  : path,
        'mean_val_f1' : metrics['f1_macro'],
        'std_val_f1'  : 0.0,
    }


def run_all_baselines(full_df, test_df, logger=None):
    results = {}
    for name in Config.ML_BASELINES:
        results[name] = train_baseline(name, full_df, test_df, logger)
    return results
