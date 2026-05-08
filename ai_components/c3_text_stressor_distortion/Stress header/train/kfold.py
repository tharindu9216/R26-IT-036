
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from config  import Config
from dataset import StressDataset
from model   import DualHeadStressModel, get_layerwise_optimizer
from trainer import Trainer


def train_kfold(model_name, hf_id, text_col, max_len,
                full_df, test_df,
                hp, crit_1a, crit_1b,
                num_subreddits, tokenizer, logger=None):
    """
    5-fold Stratified K-Fold.
    Best fold model evaluated on test set.
    Returns dict with fold_results, mean/std val F1, test metrics.
    """
    if logger:
        logger.section(f'K-Fold: {model_name} ({Config.N_FOLDS} folds)')

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    test_loader = DataLoader(
        StressDataset(test_df, tokenizer, text_col, max_len),
        batch_size=hp['batch_size'], shuffle=False,
        num_workers=2, pin_memory=True)

    skf    = StratifiedKFold(n_splits=Config.N_FOLDS,
                              shuffle=True, random_state=Config.SEED)
    labels = full_df['label'].values

    fold_results    = []
    best_global_f1  = 0.0
    best_model_path = os.path.join(Config.OUTPUT_DIR, f'{model_name}_best.pt')

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(full_df, labels)):
        if logger:
            logger.log(f'\n  ── Fold {fold+1}/{Config.N_FOLDS} '
                       f'(train={len(tr_idx)}, val={len(vl_idx)}) ──')

        train_ds = StressDataset(
            full_df.iloc[tr_idx].reset_index(drop=True),
            tokenizer,
            text_col,
            max_len,
            augment=Config.AUGMENTATION.get('enabled', False),
        )
        val_ds = StressDataset(
            full_df.iloc[vl_idx].reset_index(drop=True),
            tokenizer,
            text_col,
            max_len,
        )
        train_loader = DataLoader(
            train_ds, batch_size=hp['batch_size'], shuffle=True,
            num_workers=2, pin_memory=True)
        val_loader = DataLoader(
            val_ds, batch_size=hp['batch_size'], shuffle=False,
            num_workers=2, pin_memory=True)

        model = DualHeadStressModel(
            hf_id,
            dropout              = hp.get('dropout', 0.3),
            head_dropout         = hp.get('head_dropout', 0.1),
            intermediate         = hp.get('intermediate', 256),
            num_subreddit_labels = num_subreddits,
        ).to(Config.DEVICE)

        optimizer = get_layerwise_optimizer(
            model,
            base_lr      = hp.get('learning_rate', 2e-5),
            lr_decay     = hp.get('lr_decay',      0.9),
            head_lr_mult = hp.get('head_lr_mult',  10.0),
            weight_decay = hp.get('weight_decay',  0.01),
        )

        fold_path = os.path.join(Config.OUTPUT_DIR,
                                 f'{model_name}_fold{fold+1}.pt')
        trainer   = Trainer(model, optimizer, hp, logger=logger)

        best_f1, best_m, history = trainer.fit(
            train_loader, val_loader, crit_1a, crit_1b, fold_path)

        if best_f1 > best_global_f1:
            best_global_f1 = best_f1
            torch.save(model.state_dict(), best_model_path)
            if logger:
                logger.log(f'   New global best F1={best_global_f1:.4f} '
                           f'(Fold {fold+1})')

        fold_results.append({
            'fold'   : fold + 1,
            'val_f1' : best_f1,
            'history': history,
        })

        #  Free GPU memory after each fold
        del model
        torch.cuda.empty_cache()

    # ── Test evaluation with best fold model ──────────────────────────────────
    if logger:
        logger.log('\n  Loading best model → test evaluation...')

    best_model = DualHeadStressModel(
        hf_id,
        dropout              = hp.get('dropout', 0.3),
        head_dropout         = hp.get('head_dropout', 0.1),
        intermediate         = hp.get('intermediate', 256),
        num_subreddit_labels = num_subreddits,
    ).to(Config.DEVICE)
    best_model.load_state_dict(
        torch.load(best_model_path, map_location=Config.DEVICE))

    dummy_opt = get_layerwise_optimizer(
        best_model, base_lr=hp.get('learning_rate', 2e-5))
    test_m    = Trainer(best_model, dummy_opt, hp).evaluate(
        test_loader, crit_1a, crit_1b)

    fold_f1s = [r['val_f1'] for r in fold_results]
    mean_f1  = float(np.mean(fold_f1s))
    std_f1   = float(np.std(fold_f1s))

    if logger:
        logger.log(f'\n  {model_name} K-Fold Summary:')
        for r in fold_results:
            logger.log(f'    Fold {r["fold"]}: Val F1 = {r["val_f1"]:.4f}')
        logger.log(f'    Mean Val F1 : {mean_f1:.4f} ± {std_f1:.4f}')
        logger.log(f'    Test F1     : {test_m["f1_macro"]:.4f}')
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
