
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner

from config  import Config
from dataset import get_dataloaders
from model   import DualHeadStressModel, get_layerwise_optimizer
from trainer import Trainer

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna(hf_id, text_col, max_len,
               train_df, val_df,
               crit_1a, crit_1b,
               num_subreddits, tokenizer, logger=None, model_name=None):
    """
    Optuna search for a transformer model.

    Search space (from spec):
        lr          → suggest_float(5e-6, 3e-5, log=True)
        dropout     → suggest_float(0.1, 0.5)
        alpha       → suggest_float(0.4, 0.8)
        lr_decay    → suggest_float(0.8, 0.95)
        batch_size  → suggest_categorical([8, 16])
        accum_steps → suggest_categorical([2, 4])

    Uses MedianPruner to kill bad trials early.
    """
    display_name = model_name or hf_id
    if logger:
        logger.log(f'[Optuna] {Config.N_OPTUNA_TRIALS} trials × '
                   f'{Config.OPTUNA_EPOCHS} epochs on {display_name}')

    def objective(trial):
        lr          = trial.suggest_float('lr',         5e-6, 3e-5, log=True)
        dropout     = trial.suggest_float('dropout',    0.1,  0.5)
        alpha       = trial.suggest_float('alpha',      0.4,  0.8)
        lr_decay    = trial.suggest_float('lr_decay',   0.8,  0.95)
        batch_size  = trial.suggest_categorical('batch_size',  [8, 16])
        accum_steps = trial.suggest_categorical('accum_steps', [2, 4])

        hp = {
            **Config.BASE_HYPERPARAMS,
            'learning_rate' : lr,
            'dropout'       : dropout,
            'alpha'         : alpha,
            'lr_decay'      : lr_decay,
            'batch_size'    : batch_size,
            'accum_steps'   : accum_steps,
            'num_epochs'    : Config.OPTUNA_EPOCHS,
        }

        train_loader, val_loader, _ = get_dataloaders(
            train_df, val_df, val_df,
            tokenizer, text_col, max_len, batch_size)

        model = DualHeadStressModel(
            hf_id,
            dropout          = dropout,
            num_subreddit_labels = num_subreddits,
        ).to(Config.DEVICE)

        optimizer = get_layerwise_optimizer(
            model, lr, lr_decay,
            weight_decay=hp['weight_decay'])

        trainer = Trainer(model, optimizer, hp)
        trainer.build_scheduler(train_loader)

        for epoch in range(Config.OPTUNA_EPOCHS):
            trainer.train_epoch(train_loader, crit_1a, crit_1b)
            val_m = trainer.evaluate(val_loader, crit_1a, crit_1b)

            # MedianPruner — report intermediate value
            trial.report(val_m['f1_macro'], epoch)
            if trial.should_prune():
                del model
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

        del model
        torch.cuda.empty_cache()
        return val_m['f1_macro']

    study = optuna.create_study(
        direction = 'maximize',
        sampler   = TPESampler(seed=Config.SEED),
        pruner    = MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )
    study.optimize(objective,
                   n_trials=Config.N_OPTUNA_TRIALS,
                   show_progress_bar=True)

    best = study.best_params
    if logger:
        logger.log(f'[Optuna] Best params : {best}')
        logger.log(f'[Optuna] Best val F1 : {study.best_value:.4f}')

    # Remap param names to BASE_HYPERPARAMS keys
    return {
        'learning_rate': best['lr'],
        'dropout'      : best['dropout'],
        'alpha'        : best['alpha'],
        'lr_decay'     : best['lr_decay'],
        'batch_size'   : best['batch_size'],
        'accum_steps'  : best['accum_steps'],
    }
