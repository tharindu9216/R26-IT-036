"""Train Binary V4: cognitive Distortion vs No Distortion.

BERT, MentalBERT, and DeBERTa-v3 are tuned and cross-validated separately.
Their out-of-fold probabilities select both a binary decision threshold and
positive soft-voting weights. TF-IDF Logistic Regression and Linear SVM are
trained as research baselines. The held-out test set is never used for model,
weight, or threshold selection.
"""

import json
import os
import random
import sys
from pathlib import Path

# Shared binary datasets, transformer architecture, metrics, and configuration
# live one level above the task entrypoint.
TASK_DIR = Path(__file__).resolve().parent
SHARED_TRAIN_DIR = TASK_DIR.parent
for module_dir in (SHARED_TRAIN_DIR, TASK_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

from binary_baselines import run_binary_baselines
from config import Config
from training import (
    run_binary_optuna,
    select_binary_threshold,
    select_binary_epoch_count,
    select_three_model_binary_ensemble,
    train_binary_final,
    train_binary_kfold,
)
from utils import (
    Logger,
    compute_metrics,
    plot_comparison_dashboard,
    plot_confusion_matrices,
    plot_training_curves,
    save_results,
)


def evaluate_binary(labels, probabilities, threshold):
    labels_arr = np.asarray(labels, dtype=int)
    probabilities_arr = np.asarray(probabilities, dtype=float)
    predictions = (
        probabilities_arr[:, 1] >= threshold).astype(int)
    metrics = compute_metrics(
        labels_arr,
        predictions,
        probabilities_arr,
        num_classes=2,
    )
    return metrics, predictions


def build_binary_criterion(labels):
    labels_arr = np.asarray(labels, dtype=int)
    weights = torch.tensor(
        compute_class_weight(
            class_weight='balanced',
            classes=np.arange(2),
            y=labels_arr,
        ),
        dtype=torch.float,
        device=Config.DEVICE,
    )
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=Config.BASE_HYPERPARAMS['label_smoothing'],
    )
    return criterion, weights


def print_binary_table(results):
    print('\n' + '=' * 126)
    print('  FINAL BINARY COMPARISON — HELD-OUT TEST SET')
    print('=' * 126)
    print(
        f'{"Model":32s} {"Bal Acc":>9s} {"Macro F1":>9s} '
        f'{"Dist Prec":>10s} {"Dist Rec":>9s} {"Specificity":>11s} '
        f'{"MCC":>8s} {"PR-AUC":>8s} {"ROC-AUC":>9s}')
    print('-' * 126)
    ranked = sorted(
        results.items(),
        key=lambda item: item[1]['test_metrics']['f1_macro'],
        reverse=True,
    )
    for name, result in ranked:
        metrics = result['test_metrics']
        marker = '  ← OOF-SELECTED DEPLOYMENT' if (
            name == 'Weighted_Ensemble') else ''
        print(
            f'{name:32s} '
            f'{metrics["balanced_accuracy"]:>9.4f} '
            f'{metrics["f1_macro"]:>9.4f} '
            f'{metrics["distortion_precision"]:>10.4f} '
            f'{metrics["distortion_recall"]:>9.4f} '
            f'{metrics["specificity"]:>11.4f} '
            f'{metrics["mcc"]:>8.4f} '
            f'{metrics.get("pr_auc", 0):>8.4f} '
            f'{metrics.get("roc_auc", 0):>9.4f}'
            f'{marker}'
        )
    print('=' * 126)


def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    logger = Logger(os.path.join(Config.OUTPUT_DIR, 'training.log'))
    logger.section('CBT BINARY V4 — DISTORTION VS NO DISTORTION')
    logger.log(
        f'Transformers: {", ".join(Config.TRANSFORMERS)} '
        '+ weighted ensemble')
    logger.log(
        f'ML baselines: {", ".join(Config.ML_BASELINES)}')
    logger.log(
        'Selection: macro-F1; accuracy and MCC tie-breakers')
    logger.log(f'Device: {Config.DEVICE}')
    if torch.cuda.is_available():
        logger.log(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.log(
            f'VRAM: '
            f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.section('Loading Real Data')
    with open(os.path.join(Config.DATA_DIR, 'metadata.json')) as handle:
        metadata = json.load(handle)
    train_df = pd.read_csv(
        os.path.join(Config.DATA_DIR, 'cbt_train.csv'))
    val_df = pd.read_csv(
        os.path.join(Config.DATA_DIR, 'cbt_val.csv'))
    test_df = pd.read_csv(
        os.path.join(Config.DATA_DIR, 'cbt_test.csv'))
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    train_binary_labels = (
        train_df['label'].to_numpy(dtype=int) != 0).astype(int)
    test_binary_labels = (
        test_df['label'].to_numpy(dtype=int) != 0).astype(int)
    full_binary_labels = (
        full_df['label'].to_numpy(dtype=int) != 0).astype(int)
    binary_criterion, binary_weights = build_binary_criterion(
        train_binary_labels,
    )
    final_binary_criterion, _ = build_binary_criterion(
        full_binary_labels,
    )
    logger.log(f'Real train rows: {len(train_df):,}')
    logger.log(f'Real validation rows: {len(val_df):,}')
    logger.log(f'Real test rows: {len(test_df):,}')
    logger.log(
        f'Train binary counts [No, Yes]: '
        f'{np.bincount(train_binary_labels, minlength=2).tolist()}')
    logger.log(
        f'Class weights: '
        f'{[round(float(value), 3) for value in binary_weights]}')
    logger.log(
        'Original labels 1-10 are merged into one Distortion class.')

    objective = Config.BASE_HYPERPARAMS['selection_metric']
    all_results = {}
    oof_probabilities = {}
    individual_configs = {}

    logger.section('TRADITIONAL ML BASELINES')
    baseline_results = run_binary_baselines(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        objective=objective,
        logger=logger,
    )
    all_results.update(baseline_results)
    baseline_configs = {
        name: {
            'name': result['name'],
            'checkpoint_file': os.path.basename(result['model_path']),
            'training_text_column': result['text_col'],
            'threshold': result['binary_threshold'],
            'selection_source': result['selection_source'],
        }
        for name, result in baseline_results.items()
    }

    for model_index, model_name in enumerate(
            Config.TRANSFORMERS, start=1):
        model_config = metadata['models'][model_name]
        hf_id = model_config['hf_id']
        text_col = model_config['text_col']
        max_len = model_config.get(
            'max_len', Config.BASE_HYPERPARAMS['MAX_LEN'])
        needs_global_attention = (
            model_name in Config.GLOBAL_ATTENTION_MODELS)

        logger.section(
            f'MODEL {model_index}/{len(Config.TRANSFORMERS)} — '
            f'{model_name}')
        logger.log(f'HF ID: {hf_id}')
        logger.log(f'Text column: {text_col}')
        logger.log(f'Max length: {max_len}')

        tokenizer = AutoTokenizer.from_pretrained(
            hf_id, token=os.getenv('HF_TOKEN'))

        best_params = run_binary_optuna(
            hf_id=hf_id,
            text_col=text_col,
            max_len=max_len,
            train_df=train_df,
            val_df=val_df,
            binary_criterion=binary_criterion,
            tokenizer=tokenizer,
            logger=logger,
            model_name=model_name,
            needs_global_attention=needs_global_attention,
        )
        hyperparameters = {
            **Config.BASE_HYPERPARAMS,
            **best_params,
        }
        logger.log(
            f'Binary hyperparameters for {model_name}: '
            f'{hyperparameters}')

        epoch_selection = select_binary_epoch_count(
            model_name=model_name,
            hf_id=hf_id,
            text_col=text_col,
            max_len=max_len,
            train_df=train_df,
            val_df=val_df,
            hp=hyperparameters,
            binary_criterion=binary_criterion,
            tokenizer=tokenizer,
            logger=logger,
            needs_global_attention=needs_global_attention,
        )
        final_epochs = epoch_selection['best_epoch']
        logger.log(
            f'{model_name} validation-selected fixed epoch count: '
            f'{final_epochs}')

        cross_validation = train_binary_kfold(
            model_name=model_name,
            hf_id=hf_id,
            text_col=text_col,
            max_len=max_len,
            full_df=train_df,
            hp=hyperparameters,
            tokenizer=tokenizer,
            num_epochs=final_epochs,
            logger=logger,
            needs_global_attention=needs_global_attention,
            keep_best_checkpoint=False,
        )
        model_oof_probs = np.asarray(
            cross_validation['oof_binary_probs'])
        oof_probabilities[model_name] = model_oof_probs
        threshold_result, threshold_curve = select_binary_threshold(
            cross_validation['oof_binary_labels'],
            model_oof_probs[:, 1],
            objective=objective,
        )
        threshold = threshold_result['threshold']
        logger.log(
            f'{model_name} OOF threshold={threshold:.2f} | '
            f'Accuracy={threshold_result["accuracy"]:.4f} | '
            f'F1={threshold_result["f1_macro"]:.4f} | '
            f'MCC={threshold_result["mcc"]:.4f}')

        final_result = train_binary_final(
            model_name=model_name,
            hf_id=hf_id,
            text_col=text_col,
            max_len=max_len,
            full_df=full_df,
            test_df=test_df,
            hp=hyperparameters,
            binary_criterion=final_binary_criterion,
            tokenizer=tokenizer,
            num_epochs=final_epochs,
            logger=logger,
            needs_global_attention=needs_global_attention,
        )
        test_probabilities = np.asarray(
            final_result['test_binary_probs'])
        test_metrics, test_predictions = evaluate_binary(
            test_binary_labels,
            test_probabilities,
            threshold,
        )
        logger.log(
            f'{model_name} final selected-threshold test | '
            f'Accuracy={test_metrics["accuracy"]:.4f} | '
            f'F1={test_metrics["f1_macro"]:.4f} | '
            f'MCC={test_metrics["mcc"]:.4f} | '
            f'PR-AUC={test_metrics.get("pr_auc", 0):.4f} | '
            f'AUC={test_metrics.get("roc_auc", 0):.4f}')

        all_results[model_name] = {
            'name': f'Binary {model_name}',
            'hf_id': hf_id,
            'text_col': text_col,
            'max_len': max_len,
            'hyperparameters': hyperparameters,
            'binary_threshold': threshold,
            'threshold_selection': threshold_result,
            'threshold_curve': threshold_curve,
            'epoch_selection': epoch_selection,
            'fold_results': cross_validation['fold_results'],
            'mean_val_binary_accuracy':
                cross_validation['mean_val_binary_accuracy'],
            'std_val_binary_accuracy':
                cross_validation['std_val_binary_accuracy'],
            'mean_val_binary_f1':
                cross_validation['mean_val_binary_f1'],
            'std_val_binary_f1':
                cross_validation['std_val_binary_f1'],
            'final_refit_epochs': final_epochs,
            'model_path': final_result['model_path'],
            'test_metrics': test_metrics,
            'test_preds': test_predictions.tolist(),
            'test_labels': test_binary_labels.tolist(),
            'test_probs': test_probabilities.tolist(),
        }
        individual_configs[model_name] = {
            'checkpoint_file': os.path.basename(
                final_result['model_path']),
            'hf_id': hf_id,
            'training_text_column': text_col,
            'max_length': max_len,
            'dropout': hyperparameters.get('dropout', 0.3),
            'head_dropout':
                hyperparameters.get('head_dropout', 0.1),
            'intermediate':
                hyperparameters.get('intermediate', 256),
            'threshold': threshold,
        }

        del tokenizer
        torch.cuda.empty_cache()

    logger.section('OOF-SELECTED THREE-MODEL WEIGHTED ENSEMBLE')
    ensemble_selection = select_three_model_binary_ensemble(
        labels=(
            train_df['label'].to_numpy(dtype=int) != 0).astype(int),
        probabilities_by_model=oof_probabilities,
        objective=objective,
    )
    ensemble_weights = ensemble_selection['weights']
    ensemble_threshold_result = (
        ensemble_selection['threshold_result'])
    ensemble_threshold = ensemble_threshold_result['threshold']
    logger.log(f'OOF-selected weights: {ensemble_weights}')
    logger.log(
        f'OOF-selected threshold={ensemble_threshold:.2f} | '
        f'Accuracy={ensemble_threshold_result["accuracy"]:.4f} | '
        f'F1={ensemble_threshold_result["f1_macro"]:.4f} | '
        f'MCC={ensemble_threshold_result["mcc"]:.4f}')

    ensemble_test_probs = sum(
        ensemble_weights[model_name]
        * np.asarray(all_results[model_name]['test_probs'])
        for model_name in Config.TRANSFORMERS
    )
    ensemble_metrics, ensemble_predictions = evaluate_binary(
        test_binary_labels,
        ensemble_test_probs,
        ensemble_threshold,
    )
    all_results['Weighted_Ensemble'] = {
        'name': 'OOF-weighted BERT + MentalBERT + DeBERTa-v3',
        'members': Config.TRANSFORMERS,
        'weights': ensemble_weights,
        'binary_threshold': ensemble_threshold,
        'threshold_selection': ensemble_threshold_result,
        'test_metrics': ensemble_metrics,
        'test_preds': ensemble_predictions.tolist(),
        'test_labels': test_binary_labels.tolist(),
        'test_probs': ensemble_test_probs.tolist(),
    }
    logger.log(
        f'Weighted Ensemble test | '
        f'Accuracy={ensemble_metrics["accuracy"]:.4f} | '
        f'F1={ensemble_metrics["f1_macro"]:.4f} | '
        f'MCC={ensemble_metrics["mcc"]:.4f} | '
        f'PR-AUC={ensemble_metrics.get("pr_auc", 0):.4f} | '
        f'AUC={ensemble_metrics.get("roc_auc", 0):.4f}')

    diagnostics = pd.DataFrame({
        'true_binary_label': test_binary_labels,
        **{
            f'{model_name}_distortion_probability':
                np.asarray(
                    all_results[model_name]['test_probs'])[:, 1]
            for model_name in (
                *Config.ML_BASELINES,
                *Config.TRANSFORMERS,
            )
        },
        **{
            f'{model_name}_prediction':
                np.asarray(
                    all_results[model_name]['test_preds'], dtype=int)
            for model_name in (
                *Config.ML_BASELINES,
                *Config.TRANSFORMERS,
            )
        },
        'ensemble_distortion_probability':
            ensemble_test_probs[:, 1],
        'ensemble_prediction': ensemble_predictions,
    })
    diagnostics.to_csv(
        os.path.join(
            Config.OUTPUT_DIR, 'binary_test_predictions.csv'),
        index=False,
    )

    save_results(
        {
            'version': 'binary_v4',
            'task': {
                '0': 'No Distortion',
                '1': 'Distortion',
            },
            'models': individual_configs,
            'research_baselines': baseline_configs,
            'deployment': {
                'method': 'weighted_probability_average',
                'members': Config.TRANSFORMERS,
                'weights': ensemble_weights,
                'threshold': ensemble_threshold,
                'selection_source': (
                    '5-fold fixed-epoch out-of-fold real training split'),
                'selection_objective': objective,
                'test_set_used_for_selection': False,
                'research_baselines_included': False,
            },
            'training': {
                'real_train_rows': len(train_df),
                'real_validation_rows': len(val_df),
                'real_test_rows': len(test_df),
                'class_counts': np.bincount(
                    train_binary_labels, minlength=2).tolist(),
                'class_weights':
                    binary_weights.detach().cpu().tolist(),
                'original_labels_merged': list(range(1, 11)),
                'synthetic_data_used': False,
            },
        },
        os.path.join(Config.OUTPUT_DIR, 'binary_config.json'),
    )
    save_results(
        {
            'objective': objective,
            'weights': ensemble_weights,
            'selected': ensemble_threshold_result,
            'curve': ensemble_selection['threshold_curve'],
        },
        os.path.join(
            Config.OUTPUT_DIR, 'ensemble_selection.json'),
    )

    logger.section('SAVING RESULTS AND VISUALIZATIONS')
    print_binary_table(all_results)
    plot_confusion_matrices(
        all_results,
        Config.OUTPUT_DIR,
        class_names=['No Distortion', 'Distortion'],
    )
    plot_training_curves(all_results, Config.OUTPUT_DIR)
    plot_comparison_dashboard(all_results, Config.OUTPUT_DIR)
    save_results(
        {
            name: {
                key: value
                for key, value in result.items()
                if key not in (
                    'test_preds', 'test_labels', 'test_probs',
                    'threshold_curve',
                )
            }
            for name, result in all_results.items()
        },
        os.path.join(Config.OUTPUT_DIR, 'all_results.json'),
    )

    logger.section('BINARY V4 TRAINING COMPLETE')
    for name, result in sorted(
        all_results.items(),
        key=lambda item: item[1]['test_metrics']['f1_macro'],
        reverse=True,
    ):
        metrics = result['test_metrics']
        logger.log(
            f'  {name:32s} '
            f'Acc={metrics["accuracy"]:.4f} '
            f'F1={metrics["f1_macro"]:.4f} '
            f'MCC={metrics["mcc"]:.4f} '
            f'PR-AUC={metrics.get("pr_auc", 0):.4f} '
            f'AUC={metrics.get("roc_auc", 0):.4f}')
    logger.log(f"""
BINARY V4 OUTPUT FILES:
  {Config.OUTPUT_DIR}/BERT_binary_final.pt
  {Config.OUTPUT_DIR}/MentalBERT_binary_final.pt
  {Config.OUTPUT_DIR}/DeBERTa-v3_binary_final.pt
  {Config.OUTPUT_DIR}/binary_baseline_LR.pkl
  {Config.OUTPUT_DIR}/binary_baseline_SVM.pkl
  {Config.OUTPUT_DIR}/binary_config.json
  {Config.OUTPUT_DIR}/ensemble_selection.json
  {Config.OUTPUT_DIR}/binary_test_predictions.csv
  {Config.OUTPUT_DIR}/all_results.json
  {Config.OUTPUT_DIR}/confusion_matrices.png
  {Config.OUTPUT_DIR}/training_curves.png
  {Config.OUTPUT_DIR}/comparison_dashboard.png
  {Config.OUTPUT_DIR}/training.log
""")


if __name__ == '__main__':
    main()
