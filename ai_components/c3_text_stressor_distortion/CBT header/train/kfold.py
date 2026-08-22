"""K-Fold training for CDT (cognitive distortion) transformer models.

This module trains a transformer model using 5 different train-validation
splits. `train_kfold` reports fold-level generalization estimates (mean/std
val F1) and the epoch budget each fold needed to peak — it also keeps the
best single fold's weights and test score around as a diagnostic, but that
is NOT what should be reported/deployed (see `train_final`, below, for why).
"""


import os
import shutil
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from config  import Config
from dataset import CDTDataset
from model   import CDTModel, get_layerwise_optimizer
from trainer import Trainer


def train_kfold(model_name, hf_id, text_col, max_len,
                full_df, test_df,
                hp, binary_criterion, type_criterion, num_types,
                tokenizer, logger=None,
                needs_global_attention=False):
    """
    5-fold Stratified K-Fold.
    Best fold model evaluated on test set (diagnostic only).
    Returns dict with fold_results, mean/std val F1, test metrics.
    """
    if logger:
        logger.section(f'K-Fold: {model_name} ({Config.N_FOLDS} folds)')

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    test_loader = DataLoader(
        CDTDataset(test_df, tokenizer, text_col, max_len,
                  needs_global_attention=needs_global_attention),
        batch_size=hp['batch_size'], shuffle=False,
        num_workers=2, pin_memory=True)

    skf    = StratifiedKFold(n_splits=Config.N_FOLDS,
                              shuffle=True, random_state=Config.SEED)
    labels = full_df['label'].values

    fold_results    = []
    best_global_f1  = float('-inf')
    best_model_path = os.path.join(
        Config.OUTPUT_DIR, f'{model_name}_hierarchical_best.pt')

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(full_df, labels)):
        if logger:
            logger.log(f'\n  ── Fold {fold+1}/{Config.N_FOLDS} '
                       f'(train={len(tr_idx)}, val={len(vl_idx)}) ──')

        train_ds = CDTDataset(
            full_df.iloc[tr_idx].reset_index(drop=True),
            tokenizer, text_col, max_len,
            augment=Config.AUGMENTATION.get('enabled', False),
            needs_global_attention=needs_global_attention,
        )
        val_ds = CDTDataset(
            full_df.iloc[vl_idx].reset_index(drop=True),
            tokenizer, text_col, max_len,
            needs_global_attention=needs_global_attention,
        )
        train_loader = DataLoader(
            train_ds, batch_size=hp['batch_size'], shuffle=True,
            num_workers=2, pin_memory=True)
        val_loader = DataLoader(
            val_ds, batch_size=hp['batch_size'], shuffle=False,
            num_workers=2, pin_memory=True)

        model = CDTModel(
            hf_id,
            dropout      = hp.get('dropout', 0.3),
            head_dropout = hp.get('head_dropout', 0.1),
            intermediate = hp.get('intermediate', 256),
            num_types    = num_types,
        ).to(Config.DEVICE)

        optimizer = get_layerwise_optimizer(
            model,
            base_lr      = hp.get('learning_rate', 2e-5),
            lr_decay     = hp.get('lr_decay',      0.9),
            head_lr_mult = hp.get('head_lr_mult',  10.0),
            weight_decay = hp.get('weight_decay',  0.01),
        )

        fold_path = os.path.join(
            Config.OUTPUT_DIR,
            f'{model_name}_hierarchical_fold{fold+1}.pt')
        trainer   = Trainer(model, optimizer, hp, logger=logger)

        best_f1, best_m, history = trainer.fit(
            train_loader, val_loader,
            binary_criterion, type_criterion, fold_path)

        # Epoch at which this fold hit its best val F1 (1-indexed epoch
        # count) — used later to pick an epoch budget for the full-data
        # refit. `history['val_f1']` only has entries up to early stopping,
        # so this is always in range.
        best_epoch = history['val_f1'].index(max(history['val_f1'])) + 1

        if best_f1 > best_global_f1:
            best_global_f1 = best_f1
            # `fold_path` contains the best epoch selected by Trainer.fit;
            # the in-memory model may already be on a later epoch.
            shutil.copyfile(fold_path, best_model_path)
            if logger:
                logger.log(f'   New global best F1={best_global_f1:.4f} '
                           f'(Fold {fold+1})')

        fold_results.append({
            'fold'      : fold + 1,
            'val_f1'    : best_f1,
            'best_epoch': best_epoch,
            'history'   : history,
        })

        #  Free GPU memory after each fold
        del model
        torch.cuda.empty_cache()

    #  Test evaluation with best fold model (diagnostic only)
    if logger:
        logger.log('\n  Loading best model → test evaluation...')

    best_model = CDTModel(
        hf_id,
        dropout      = hp.get('dropout', 0.3),
        head_dropout = hp.get('head_dropout', 0.1),
        intermediate = hp.get('intermediate', 256),
        num_types    = num_types,
    ).to(Config.DEVICE)
    best_model.load_state_dict(
        torch.load(best_model_path, map_location=Config.DEVICE))

    dummy_opt = get_layerwise_optimizer(
        best_model, base_lr=hp.get('learning_rate', 2e-5))
    test_m = Trainer(best_model, dummy_opt, hp).evaluate(
        test_loader, binary_criterion, type_criterion)

    fold_f1s = [r['val_f1'] for r in fold_results]
    mean_f1  = float(np.mean(fold_f1s))
    std_f1   = float(np.std(fold_f1s))

    if logger:
        logger.log(f'\n  {model_name} K-Fold Summary:')
        for r in fold_results:
            logger.log(f'    Fold {r["fold"]}: Val F1 = {r["val_f1"]:.4f}')
        logger.log(f'    Mean Val F1 : {mean_f1:.4f} ± {std_f1:.4f}')
        logger.log(f'    Test F1     : {test_m["f1_macro"]:.4f}')
        logger.log(f'    Binary F1   : {test_m["binary_f1_macro"]:.4f}')
        logger.log(f'    Type F1     : {test_m["type_f1_macro"]:.4f}')
        logger.log(f'    Test Acc    : {test_m["accuracy"]:.4f}')
        logger.log(f'    Test MCC    : {test_m["mcc"]:.4f}')
        logger.log(f'    Test ROC-AUC: {test_m.get("roc_auc", 0):.4f}')

    del best_model
    torch.cuda.empty_cache()

    return {
        'hf_id'        : hf_id,
        'hyperparams'  : hp,
        'fold_results' : fold_results,
        'mean_val_f1'  : mean_f1,
        'std_val_f1'   : std_f1,
        'test_metrics' : {k: v for k, v in test_m.items()
                          if k not in ['preds', 'labels', 'probs']},
        'test_preds'   : test_m['preds'],
        'test_labels'  : test_m['labels'],
        'test_probs'   : test_m['probs'],
        'model_path'   : best_model_path,
    }


def train_final(model_name, hf_id, text_col, max_len,
                full_df, test_df, hp,
                binary_criterion, type_criterion, num_types,
                tokenizer, num_epochs, logger=None,
                needs_global_attention=False):
    """
    Refit a single model on 100% of train+val (full_df) for a fixed
    `num_epochs`, then evaluate once on the held-out test set.

    This is the model that should actually be reported/deployed. K-Fold CV
    (train_kfold, above) is used to validate hyperparameters and to derive
    `num_epochs` (e.g. the average best-epoch across folds) — not to
    hand-pick whichever fold happened to score highest on its own
    validation split, which biases the reported result toward a lucky draw
    instead of the true expected performance.
    """
    if logger:
        logger.section(
            f'Final Refit: {model_name} '
            f'({num_epochs} epochs, 100% train+val)')

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    test_loader = DataLoader(
        CDTDataset(test_df, tokenizer, text_col, max_len,
                  needs_global_attention=needs_global_attention),
        batch_size=hp['batch_size'], shuffle=False,
        num_workers=2, pin_memory=True)

    full_ds = CDTDataset(
        full_df.reset_index(drop=True),
        tokenizer, text_col, max_len,
        augment=Config.AUGMENTATION.get('enabled', False),
        needs_global_attention=needs_global_attention,
    )
    full_loader = DataLoader(
        full_ds, batch_size=hp['batch_size'], shuffle=True,
        num_workers=2, pin_memory=True)

    model = CDTModel(
        hf_id,
        dropout      = hp.get('dropout', 0.3),
        head_dropout = hp.get('head_dropout', 0.1),
        intermediate = hp.get('intermediate', 256),
        num_types    = num_types,
    ).to(Config.DEVICE)

    optimizer = get_layerwise_optimizer(
        model,
        base_lr      = hp.get('learning_rate', 2e-5),
        lr_decay     = hp.get('lr_decay',      0.9),
        head_lr_mult = hp.get('head_lr_mult',  10.0),
        weight_decay = hp.get('weight_decay',  0.01),
    )

    final_path = os.path.join(
        Config.OUTPUT_DIR, f'{model_name}_hierarchical_final.pt')
    trainer    = Trainer(model, optimizer, hp, logger=logger)

    history = trainer.fit_fixed_epochs(
        full_loader, binary_criterion, type_criterion,
        final_path, num_epochs)

    if logger:
        logger.log('\n  Evaluating final refit model on test set...')
    test_m = trainer.evaluate(
        test_loader, binary_criterion, type_criterion)

    if logger:
        logger.log(f'  {model_name} Final Refit — Test F1  : {test_m["f1_macro"]:.4f}')
        logger.log(f'  {model_name} Final Refit — Binary F1: '
                   f'{test_m["binary_f1_macro"]:.4f}')
        logger.log(f'  {model_name} Final Refit — Type F1  : '
                   f'{test_m["type_f1_macro"]:.4f}')
        logger.log(f'  {model_name} Final Refit — Test Acc : {test_m["accuracy"]:.4f}')
        logger.log(f'  {model_name} Final Refit — Test MCC : {test_m["mcc"]:.4f}')

    del model
    torch.cuda.empty_cache()

    return {
        'num_epochs'   : num_epochs,
        'history'      : history,
        'test_metrics' : {k: v for k, v in test_m.items()
                          if k not in ['preds', 'labels', 'probs']},
        'test_preds'   : test_m['preds'],
        'test_labels'  : test_m['labels'],
        'test_probs'   : test_m['probs'],
        'model_path'   : final_path,
    }
