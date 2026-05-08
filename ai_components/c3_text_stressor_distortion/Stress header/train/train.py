
import os, json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from config    import Config
from baselines import run_all_baselines
from tuner     import run_optuna
from kfold     import train_kfold
from utils     import (Logger, save_results, print_final_table,
                        compute_metrics,
                        plot_confusion_matrices,
                        plot_training_curves,
                        plot_comparison_dashboard)


def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    log = Logger(os.path.join(Config.OUTPUT_DIR, 'training.log'))

    log.section('STRESS DETECTION TRAINING PIPELINE')
    log.log(f'Author  : MUNASINGHE M.A.C.D | IT22252586')
    log.log(f'Device  : {Config.DEVICE}')
    if torch.cuda.is_available():
        log.log(f'GPU     : {torch.cuda.get_device_name(0)}')
        log.log(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    # Reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.section('Loading Data')

    with open(os.path.join(Config.DATA_DIR, 'metadata.json')) as f:
        meta = json.load(f)

    train_df = pd.read_csv(os.path.join(Config.DATA_DIR, 'dreaddit_train.csv'))
    val_df   = pd.read_csv(os.path.join(Config.DATA_DIR, 'dreaddit_val.csv'))
    test_df  = pd.read_csv(os.path.join(Config.DATA_DIR, 'dreaddit_test.csv'))
    full_df  = pd.concat([train_df, val_df], ignore_index=True)

    # Class weights — from metadata.json (computed by preprocessing)
    weights_1a     = torch.tensor(meta['class_weights']['head1a'],
                                  dtype=torch.float,
                                  device=Config.DEVICE)
    weights_1b     = torch.tensor(meta['class_weights']['head1b'],
                                  dtype=torch.float,
                                  device=Config.DEVICE)
    num_subreddits = meta['num_labels']['head1b']

    # Model configs — hf_id, text_col, max_len per model
    model_configs  = meta['models']

    log.log(f'Train+Val : {len(full_df):,} rows')
    log.log(f'Test      : {len(test_df):,} rows')
    log.log(f'Subreddits: {num_subreddits}')
    log.log(f'W_1A      : {meta["class_weights"]["head1a"]}')
    log.log(f'W_1B      : {[round(w,3) for w in meta["class_weights"]["head1b"]]}')

    # ── Loss functions (shared across all transformer models) ─────────────────
    # CrossEntropyLoss with class weights + label smoothing
    crit_1a = nn.CrossEntropyLoss(weight=weights_1a, label_smoothing=0.1)
    crit_1b = nn.CrossEntropyLoss(weight=weights_1b, label_smoothing=0.1)

    all_results = {}

    # =========================================================================
    # STEP 1 — ML Baselines
    # Trained on full_df (train+val combined), tested on test_df
    # < 2 minutes
    # =========================================================================
    log.section('STEP 1 — ML Baselines')
    baseline_results = run_all_baselines(full_df, test_df, logger=log)
    all_results.update(baseline_results)

    # =========================================================================
    # STEP 2 — Optuna per transformer model
    # Each transformer gets its own best params.
    # =========================================================================
    log.section('STEP 2 — Optuna Hyperparameter Search')

    hyperparams_by_model = {}

    for model_name in Config.TRANSFORMERS:
        log.section(f'Optuna — {model_name}')

        cfg      = model_configs[model_name]
        hf_id    = cfg['hf_id']
        text_col = cfg['text_col']
        max_len  = cfg.get('max_len', Config.BASE_HYPERPARAMS['MAX_LEN'])

        log.log(f'HF ID    : {hf_id}')
        log.log(f'Text col : {text_col}')
        log.log(f'Max len  : {max_len}')

        tokenizer = AutoTokenizer.from_pretrained(
            hf_id, token=os.getenv('HF_TOKEN'))

        best_params = run_optuna(
            hf_id          = hf_id,
            text_col       = text_col,
            max_len        = max_len,
            train_df       = train_df,
            val_df         = val_df,
            crit_1a        = crit_1a,
            crit_1b        = crit_1b,
            num_subreddits = num_subreddits,
            tokenizer      = tokenizer,
            logger         = log,
            model_name     = model_name,
        )

        hyperparams = {**Config.BASE_HYPERPARAMS, **best_params}
        hyperparams_by_model[model_name] = hyperparams
        log.log(f'Final hyperparams for {model_name}: {hyperparams}')

        del tokenizer
        torch.cuda.empty_cache()

    # =========================================================================
    # STEPS 3-5 — Train all 3 transformer models
    # Uses model-specific best_params from Optuna
    # =========================================================================
    for step_num, model_name in enumerate(Config.TRANSFORMERS, start=3):
        log.section(f'STEP {step_num} — {model_name}')

        cfg      = model_configs[model_name]
        hf_id    = cfg['hf_id']
        text_col = cfg['text_col']
        max_len  = cfg.get('max_len', Config.BASE_HYPERPARAMS['MAX_LEN'])

        log.log(f'HF ID    : {hf_id}')
        log.log(f'Text col : {text_col}')
        log.log(f'Max len  : {max_len}')

        log.log('Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(
            hf_id, token=os.getenv('HF_TOKEN'))

        results = train_kfold(
            model_name     = model_name,
            hf_id          = hf_id,
            text_col       = text_col,
            max_len        = max_len,
            full_df        = full_df,
            test_df        = test_df,
            hp             = hyperparams_by_model[model_name],
            crit_1a        = crit_1a,
            crit_1b        = crit_1b,
            num_subreddits = num_subreddits,
            tokenizer      = tokenizer,
            logger         = log,
        )

        all_results[model_name] = results

        #  Free GPU memory before next model
        del tokenizer
        torch.cuda.empty_cache()

    # =========================================================================
    # STEP 6 — Probability Ensemble
    # Average BERT + DeBERTa-v3 Head 1A probabilities.
    # =========================================================================
    log.section('STEP 6 — BERT + DeBERTa-v3 Ensemble')

    ensemble_members = ['BERT', 'DeBERTa-v3']
    if all(m in all_results and 'test_probs' in all_results[m]
           for m in ensemble_members):
        labels = all_results[ensemble_members[0]]['test_labels']
        probs = np.mean(
            [np.array(all_results[m]['test_probs']) for m in ensemble_members],
            axis=0,
        )
        preds = (probs >= 0.5).astype(int)
        metrics = compute_metrics(labels, preds.tolist(), probs.tolist())

        all_results['Ensemble_BERT_DeBERTa-v3'] = {
            'name'        : 'Average probabilities: BERT + DeBERTa-v3',
            'members'     : ensemble_members,
            'test_metrics': metrics,
            'test_preds'  : preds.tolist(),
            'test_labels' : labels,
            'test_probs'  : probs.tolist(),
            'mean_val_f1' : metrics['f1_macro'],
            'std_val_f1'  : 0.0,
        }

        log.log(f'  Members      : {", ".join(ensemble_members)}')
        log.log(f'  Test Accuracy: {metrics["accuracy"]:.4f}')
        log.log(f'  Test F1 Macro: {metrics["f1_macro"]:.4f}')
        log.log(f'  Test MCC     : {metrics["mcc"]:.4f}')
        log.log(f'  Test ROC-AUC : {metrics.get("roc_auc", 0):.4f}')
    else:
        log.log('  Skipped: missing test probabilities for BERT or DeBERTa-v3')

    # =========================================================================
    # STEP 7 — Results, Visualizations, Save
    # =========================================================================
    log.section('STEP 7 — Saving Results & Visualizations')

    # Print table
    print_final_table(all_results)

    # Plots
    log.log('Generating visualizations...')
    plot_confusion_matrices(all_results, Config.OUTPUT_DIR)
    plot_training_curves(all_results, Config.OUTPUT_DIR)
    plot_comparison_dashboard(all_results, Config.OUTPUT_DIR)

    # Save JSON
    save_results(
        {name: {k: v for k, v in res.items()
                if k not in ['test_preds', 'test_labels', 'test_probs']}
         for name, res in all_results.items()},
        os.path.join(Config.OUTPUT_DIR, 'all_results.json')
    )

    log.log(f'Saved → {Config.OUTPUT_DIR}all_results.json')
    log.log(f'Saved → {Config.OUTPUT_DIR}confusion_matrices.png')
    log.log(f'Saved → {Config.OUTPUT_DIR}training_curves.png')
    log.log(f'Saved → {Config.OUTPUT_DIR}comparison_dashboard.png')

    # =========================================================================
    # Final summary
    # =========================================================================
    log.section('TRAINING COMPLETE')
    medals = ['', '', '']
    ranked = sorted(all_results.items(),
                    key=lambda x: x[1].get('test_metrics', {}).get('f1_macro', 0),
                    reverse=True)

    for i, (name, res) in enumerate(ranked):
        m = res.get('test_metrics', {})
        medal = medals[i] if i < len(medals) else f'{i+1}th'
        log.log(f'  {medal} {name:25s} '
                f'F1={m.get("f1_macro",0):.4f}  '
                f'Acc={m.get("accuracy",0):.4f}  '
                f'MCC={m.get("mcc",0):.4f}')

    log.log(f"""
FEATURES:
   Two-layer ClassificationHead (768→256→GELU→Dropout→classes)
   ML Baselines        (TF-IDF + LR, TF-IDF + LinearSVC)
   Optuna per model    ({Config.N_OPTUNA_TRIALS} trials/model, MedianPruner)
   BERT+DeBERTa Ensemble (probability averaging)
   Layer-wise LR Decay (base_lr × 0.9^(12-i) per layer)
   Label Smoothing     (0.1)
   Cosine LR Schedule  (with warmup)
   Mixed Precision     (fp16/bf16 auto)
   Gradient Accumulation
   K-Fold CV           ({Config.N_FOLDS}-fold Stratified)
   Early Stopping      (patience={Config.BASE_HYPERPARAMS['patience']})
   GPU memory cleared  (del model + cuda.empty_cache after each model)

OUTPUT FILES:
  {Config.OUTPUT_DIR}BERT_best.pt
  {Config.OUTPUT_DIR}MentalBERT_best.pt
  {Config.OUTPUT_DIR}DeBERTa-v3_best.pt
  {Config.OUTPUT_DIR}all_results.json
  {Config.OUTPUT_DIR}comparison_dashboard.png
  {Config.OUTPUT_DIR}confusion_matrices.png
  {Config.OUTPUT_DIR}training_curves.png
  {Config.OUTPUT_DIR}training.log

NEXT: Preprocess CDT dataset → CDT head → Global multi-head
    """)


if __name__ == '__main__':
    main()
