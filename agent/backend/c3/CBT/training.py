"""Training utilities for binary CBT distortion detection.

The active pipeline trains BERT, MentalBERT, and DeBERTa-v3 to predict whether
a cognitive distortion is present. Hyperparameters and epoch count are chosen
on the dedicated validation split; ensemble weights and decision thresholds
are selected from fixed-epoch out-of-fold predictions on the training split.
The held-out test set is used only once for final evaluation.

Legacy conditional TF-IDF helpers remain below for older experiments.
"""

import math
import os
import pickle
import shutil

import numpy as np
import optuna
import torch
import torch.nn as nn
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from config import Config
from dataset import CDTDataset, get_dataloaders
from model import BinaryCDTModel, get_layerwise_optimizer
from utils import compute_metrics


@contextmanager
def _nullctx():
    yield


class BinaryTrainer:
    """Training loop for the dedicated Distortion/No-Distortion model."""

    def __init__(self, model, optimizer, hp, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.hp = hp
        self.logger = logger
        self.device = Config.DEVICE
        self.scheduler = None

        dtype = Config.get_amp_dtype()
        self.use_amp = dtype is not None
        self.dtype = dtype
        self.scaler = (
            GradScaler()
            if self.use_amp and dtype == torch.float16
            else None
        )

    def _amp(self):
        return autocast(dtype=self.dtype) if self.use_amp else _nullctx()

    def build_scheduler(self, train_loader, num_epochs=None):
        accum = self.hp.get('accum_steps', 4)
        epochs = (
            num_epochs
            if num_epochs is not None
            else self.hp.get('num_epochs', 5)
        )
        steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
        total_steps = max(1, steps_per_epoch * epochs)
        warmup_steps = int(
            total_steps * self.hp.get('warmup_ratio', 0.1))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps)

    @staticmethod
    def _forward(model, batch, device):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        if 'global_attention_mask' in batch:
            return model(
                input_ids,
                attention_mask,
                global_attention_mask=batch[
                    'global_attention_mask'].to(device),
            )
        return model(input_ids, attention_mask)

    def train_epoch(self, loader, criterion):
        self.model.train()
        accum = self.hp.get('accum_steps', 4)
        clip = self.hp.get('max_grad_norm', 1.0)
        total_loss = 0.0
        labels_all, probs_all = [], []
        self.optimizer.zero_grad()

        for step, batch in enumerate(loader):
            labels = batch['binary_label'].to(self.device)
            window_start = (step // accum) * accum
            window_size = min(accum, len(loader) - window_start)
            with self._amp():
                logits = self._forward(self.model, batch, self.device)
                raw_loss = criterion(logits, labels)
                loss = raw_loss / window_size

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += raw_loss.item()
            labels_all.extend(labels.detach().cpu().tolist())
            probs_all.extend(
                torch.softmax(logits.float(), dim=1)
                .detach().cpu().tolist())

        probs_arr = np.asarray(probs_all)
        preds = (probs_arr[:, 1] >= 0.5).astype(int)
        return (
            total_loss / max(1, len(loader)),
            compute_metrics(
                labels_all, preds, probs_arr, num_classes=2),
        )

    def evaluate(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        labels_all, probs_all = [], []

        with torch.no_grad():
            for batch in loader:
                labels = batch['binary_label'].to(self.device)
                with self._amp():
                    logits = self._forward(
                        self.model, batch, self.device)
                    loss = criterion(logits, labels)
                total_loss += loss.item()
                labels_all.extend(labels.cpu().tolist())
                probs_all.extend(
                    torch.softmax(logits.float(), dim=1)
                    .cpu().tolist())

        probs_arr = np.asarray(probs_all)
        preds = (probs_arr[:, 1] >= 0.5).astype(int)
        return {
            'loss': total_loss / max(1, len(loader)),
            'labels': labels_all,
            'preds': preds.tolist(),
            'probs': probs_arr.tolist(),
            **compute_metrics(
                labels_all, preds, probs_arr, num_classes=2),
        }

    def fit(self, train_loader, val_loader, criterion, save_path):
        patience = self.hp.get('patience', 3)
        min_epochs = self.hp.get('min_epochs', 5)
        num_epochs = self.hp.get('num_epochs', 10)
        patience_counter = 0
        selection_metric = self.hp.get(
            'selection_metric', 'f1_macro')
        if selection_metric not in ('accuracy', 'f1_macro'):
            raise ValueError(
                'selection_metric must be accuracy or f1_macro')
        best_score = float('-inf')
        best_epoch = 1
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_accuracy': [],
        }
        self.build_scheduler(train_loader, num_epochs)

        for epoch in range(num_epochs):
            train_loss, train_metrics = self.train_epoch(
                train_loader, criterion)
            val_metrics = self.evaluate(val_loader, criterion)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1_macro'])
            history['val_accuracy'].append(val_metrics['accuracy'])

            if self.logger:
                self.logger.log(
                    f'    Epoch {epoch + 1}/{num_epochs} | '
                    f'Loss: {train_loss:.4f} | '
                    f'Train Binary F1: '
                    f'{train_metrics["f1_macro"]:.4f} | '
                    f'Val Binary F1: '
                    f'{val_metrics["f1_macro"]:.4f} | '
                    f'Val Acc: {val_metrics["accuracy"]:.4f}'
                )

            current_score = val_metrics[selection_metric]
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                if self.logger:
                    self.logger.log(
                        f'     Best binary checkpoint saved '
                        f'{selection_metric}={best_score:.4f}')
            else:
                patience_counter += 1
                if self.logger:
                    self.logger.log(
                        f'    No improvement '
                        f'({patience_counter}/{patience})')
                if (
                    epoch + 1 >= min_epochs
                    and patience_counter >= patience
                ):
                    if self.logger:
                        self.logger.log(
                            f'    Early stopping at epoch {epoch + 1}')
                    break

        return best_score, best_epoch, history

    def fit_fixed_epochs(
            self, train_loader, criterion, save_path, num_epochs,
            val_loader=None):
        self.build_scheduler(train_loader, num_epochs)
        history = {
            'train_loss': [],
            'train_f1': [],
            'val_loss': [],
            'val_f1': [],
            'val_accuracy': [],
        }

        for epoch in range(num_epochs):
            loss, metrics = self.train_epoch(train_loader, criterion)
            history['train_loss'].append(loss)
            history['train_f1'].append(metrics['f1_macro'])
            val_metrics = (
                self.evaluate(val_loader, criterion)
                if val_loader is not None
                else None
            )
            if val_metrics is not None:
                history['val_loss'].append(val_metrics['loss'])
                history['val_f1'].append(val_metrics['f1_macro'])
                history['val_accuracy'].append(val_metrics['accuracy'])
            if self.logger:
                line = (
                    f'    Epoch {epoch + 1}/{num_epochs} | '
                    f'Loss: {loss:.4f} | '
                    f'Train Binary F1: {metrics["f1_macro"]:.4f}'
                )
                if val_metrics is not None:
                    line += (
                        f' | Val Binary F1: {val_metrics["f1_macro"]:.4f}'
                        f' | Val Acc: {val_metrics["accuracy"]:.4f}'
                    )
                self.logger.log(line)

        torch.save(self.model.state_dict(), save_path)
        if self.logger:
            self.logger.log(
                f'    Final binary model saved → {save_path}')
        return history


def run_binary_optuna(
        hf_id, text_col, max_len, train_df, val_df,
        binary_criterion, tokenizer, logger=None,
        model_name='DeBERTa-v3', needs_global_attention=False):
    """Tune the binary transformer using the configured validation metric."""
    if logger:
        logger.log(
            f'[Binary Optuna] {Config.N_OPTUNA_TRIALS} trials × '
            f'{Config.OPTUNA_EPOCHS} epochs on {model_name}')

    def objective(trial):
        hp = {
            **Config.BASE_HYPERPARAMS,
            'learning_rate': trial.suggest_float(
                'lr', 5e-6, 3e-5, log=True),
            'dropout': trial.suggest_float(
                'dropout', 0.1, 0.5),
            'lr_decay': trial.suggest_float(
                'lr_decay', 0.8, 0.95),
            'batch_size': trial.suggest_categorical(
                'batch_size', Config.batch_size_choices(max_len)),
            'accum_steps': trial.suggest_categorical(
                'accum_steps', [2, 4]),
            'num_epochs': Config.OPTUNA_EPOCHS,
        }

        train_loader, val_loader, _ = get_dataloaders(
            train_df, val_df, val_df,
            tokenizer, text_col, max_len, hp['batch_size'],
            needs_global_attention=needs_global_attention,
        )
        model = BinaryCDTModel(
            hf_id,
            dropout=hp['dropout'],
            head_dropout=hp.get('head_dropout', 0.1),
            intermediate=hp.get('intermediate', 256),
        ).to(Config.DEVICE)
        optimizer = get_layerwise_optimizer(
            model,
            hp['learning_rate'],
            hp['lr_decay'],
            hp.get('head_lr_mult', 10.0),
            hp.get('weight_decay', 0.01),
        )
        trainer = BinaryTrainer(model, optimizer, hp)
        trainer.build_scheduler(train_loader, Config.OPTUNA_EPOCHS)

        best_trial_score = float('-inf')
        for epoch in range(Config.OPTUNA_EPOCHS):
            trainer.train_epoch(train_loader, binary_criterion)
            metrics = trainer.evaluate(val_loader, binary_criterion)
            selection_metric = hp.get(
                'selection_metric', 'f1_macro')
            best_trial_score = max(
                best_trial_score, metrics[selection_metric])
            trial.report(best_trial_score, epoch)
            if trial.should_prune():
                del model
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

        del model
        torch.cuda.empty_cache()
        return best_trial_score

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=Config.SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=1),
    )
    study.optimize(
        objective,
        n_trials=Config.N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )
    best = study.best_params
    if logger:
        logger.log(f'[Binary Optuna] Best params: {best}')
        logger.log(
            f'[Binary Optuna] Best validation '
            f'{Config.BASE_HYPERPARAMS["selection_metric"]}: '
            f'{study.best_value:.4f}')
    return {
        'learning_rate': best['lr'],
        'dropout': best['dropout'],
        'lr_decay': best['lr_decay'],
        'batch_size': best['batch_size'],
        'accum_steps': best['accum_steps'],
    }


def select_binary_epoch_count(
        model_name, hf_id, text_col, max_len, train_df, val_df, hp,
        binary_criterion, tokenizer, logger=None,
        needs_global_attention=False):
    """Select the epoch count on the dedicated validation split.

    The selected count is later held fixed in outer-fold OOF training, so the
    outer fold labels cannot influence checkpoint selection.
    """
    if logger:
        logger.section(f'Epoch Selection: {model_name}')

    train_loader, val_loader, _ = get_dataloaders(
        train_df,
        val_df,
        val_df,
        tokenizer,
        text_col,
        max_len,
        hp['batch_size'],
        needs_global_attention=needs_global_attention,
    )
    model = BinaryCDTModel(
        hf_id,
        dropout=hp.get('dropout', 0.3),
        head_dropout=hp.get('head_dropout', 0.1),
        intermediate=hp.get('intermediate', 256),
    ).to(Config.DEVICE)
    optimizer = get_layerwise_optimizer(
        model,
        hp.get('learning_rate', 1.5e-5),
        hp.get('lr_decay', 0.9),
        hp.get('head_lr_mult', 10.0),
        hp.get('weight_decay', 0.01),
    )
    trainer = BinaryTrainer(model, optimizer, hp, logger)
    checkpoint_path = os.path.join(
        '/tmp',
        f'cbt_{os.getpid()}_{model_name}_epoch_selection.pt',
    )
    best_score, best_epoch, history = trainer.fit(
        train_loader,
        val_loader,
        binary_criterion,
        checkpoint_path,
    )
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    del model
    torch.cuda.empty_cache()
    return {
        'best_epoch': int(best_epoch),
        'best_score': float(best_score),
        'selection_metric': hp.get('selection_metric', 'f1_macro'),
        'history': history,
    }


def train_binary_kfold(
        model_name, hf_id, text_col, max_len, full_df, hp,
        tokenizer, num_epochs, logger=None,
        needs_global_attention=False, keep_best_checkpoint=True):
    """Train fixed-epoch outer folds and return leakage-safe OOF probabilities."""
    if logger:
        logger.section(
            f'Binary K-Fold: {model_name} ({Config.N_FOLDS} folds)')

    binary_labels = (
        full_df['label'].to_numpy(dtype=int) != 0).astype(int)
    oof_probs = np.zeros((len(full_df), 2), dtype=float)
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS,
        shuffle=True,
        random_state=Config.SEED,
    )
    fold_results = []
    best_global_score = float('-inf')
    best_path = (
        os.path.join(
            Config.OUTPUT_DIR, f'{model_name}_binary_best.pt')
        if keep_best_checkpoint
        else None
    )

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(full_df, binary_labels), start=1):
        if logger:
            logger.log(
                f'\n  ── Binary Fold {fold}/{Config.N_FOLDS} '
                f'(train={len(train_idx)}, val={len(val_idx)}) ──')

        train_loader = DataLoader(
            CDTDataset(
                full_df.iloc[train_idx].reset_index(drop=True),
                tokenizer, text_col, max_len,
                augment=Config.AUGMENTATION.get('enabled', False),
                needs_global_attention=needs_global_attention,
            ),
            batch_size=hp['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            CDTDataset(
                full_df.iloc[val_idx].reset_index(drop=True),
                tokenizer, text_col, max_len,
                needs_global_attention=needs_global_attention,
            ),
            batch_size=hp['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        model = BinaryCDTModel(
            hf_id,
            dropout=hp.get('dropout', 0.3),
            head_dropout=hp.get('head_dropout', 0.1),
            intermediate=hp.get('intermediate', 256),
        ).to(Config.DEVICE)
        optimizer = get_layerwise_optimizer(
            model,
            hp.get('learning_rate', 1.5e-5),
            hp.get('lr_decay', 0.9),
            hp.get('head_lr_mult', 10.0),
            hp.get('weight_decay', 0.01),
        )
        trainer = BinaryTrainer(model, optimizer, hp, logger)
        fold_weights = torch.tensor(
            compute_class_weight(
                class_weight='balanced',
                classes=np.arange(2),
                y=binary_labels[train_idx],
            ),
            dtype=torch.float,
            device=Config.DEVICE,
        )
        fold_criterion = nn.CrossEntropyLoss(
            weight=fold_weights,
            label_smoothing=hp.get('label_smoothing', 0.0),
        )
        fold_path = os.path.join(
            '/tmp',
            f'cbt_{os.getpid()}_{model_name}_binary_fold{fold}.pt',
        )
        history = trainer.fit_fixed_epochs(
            train_loader,
            fold_criterion,
            fold_path,
            num_epochs,
            val_loader=val_loader,
        )

        model.load_state_dict(
            torch.load(fold_path, map_location=Config.DEVICE))
        val_metrics = trainer.evaluate(val_loader, fold_criterion)
        selection_metric = hp.get('selection_metric', 'f1_macro')
        best_score = val_metrics[selection_metric]
        best_epoch = num_epochs
        oof_probs[val_idx] = np.asarray(val_metrics['probs'])

        if keep_best_checkpoint and best_score > best_global_score:
            best_global_score = best_score
            shutil.copyfile(fold_path, best_path)
        if os.path.exists(fold_path):
            os.remove(fold_path)

        fold_results.append({
            'fold': fold,
            'val_binary_f1': val_metrics['f1_macro'],
            'val_binary_accuracy': val_metrics['accuracy'],
            'selection_score': best_score,
            'best_epoch': best_epoch,
            'history': history,
        })
        del model
        torch.cuda.empty_cache()

    fold_f1s = [row['val_binary_f1'] for row in fold_results]
    fold_accuracies = [
        row['val_binary_accuracy'] for row in fold_results]
    if logger:
        logger.log('\n  Binary K-Fold Summary:')
        for row in fold_results:
            logger.log(
                f'    Fold {row["fold"]}: '
                f'Binary Val Acc={row["val_binary_accuracy"]:.4f} | '
                f'F1={row["val_binary_f1"]:.4f}')
        logger.log(
            f'    Mean Binary Val F1: '
            f'{np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}')
        logger.log(
            f'    Mean Binary Val Accuracy: '
            f'{np.mean(fold_accuracies):.4f} ± '
            f'{np.std(fold_accuracies):.4f}')

    return {
        'fold_results': fold_results,
        'mean_val_binary_f1': float(np.mean(fold_f1s)),
        'std_val_binary_f1': float(np.std(fold_f1s)),
        'mean_val_binary_accuracy':
            float(np.mean(fold_accuracies)),
        'std_val_binary_accuracy':
            float(np.std(fold_accuracies)),
        'oof_binary_labels': binary_labels.tolist(),
        'oof_binary_probs': oof_probs.tolist(),
        'best_model_path': best_path,
    }


def select_binary_threshold(
        labels, distortion_probs, objective='f1_macro'):
    """Select a threshold using validation/OOF accuracy or macro-F1."""
    if objective not in ('accuracy', 'f1_macro'):
        raise ValueError('objective must be accuracy or f1_macro')
    labels_arr = np.asarray(labels, dtype=int)
    probs_arr = np.asarray(distortion_probs, dtype=float)
    rows = []
    best = None

    for threshold in np.round(np.arange(0.20, 0.801, 0.01), 2):
        preds = (probs_arr >= threshold).astype(int)
        score = f1_score(
            labels_arr, preds, average='macro', zero_division=0)
        accuracy = accuracy_score(labels_arr, preds)
        mcc = matthews_corrcoef(labels_arr, preds)
        row = {
            'threshold': float(threshold),
            'f1_macro': float(score),
            'accuracy': float(accuracy),
            'mcc': float(mcc),
        }
        rows.append(row)
        primary = accuracy if objective == 'accuracy' else score
        secondary = score if objective == 'accuracy' else accuracy
        rank = (
            primary,
            secondary,
            mcc,
            -abs(float(threshold) - 0.5),
        )
        if best is None or rank > best[0]:
            best = (rank, row)

    return best[1], rows


def select_three_model_binary_ensemble(
        labels, probabilities_by_model, objective='accuracy'):
    """Select positive ensemble weights and threshold using OOF predictions.

    A coarse 0.1 weight grid is intentionally used to limit overfitting.
    Every model receives at least 0.1 weight, so BERT, MentalBERT, and
    DeBERTa-v3 all participate in the final prediction.
    """
    model_names = list(probabilities_by_model)
    if len(model_names) != 3:
        raise ValueError(
            'The weighted ensemble requires exactly three models')

    arrays = {
        name: np.asarray(probabilities_by_model[name], dtype=float)
        for name in model_names
    }
    row_counts = {len(value) for value in arrays.values()}
    if len(row_counts) != 1:
        raise ValueError('All OOF probability arrays must have equal rows')

    best = None
    for first_units in range(1, 9):
        for second_units in range(1, 10 - first_units):
            third_units = 10 - first_units - second_units
            if third_units < 1:
                continue
            weights = {
                model_names[0]: first_units / 10,
                model_names[1]: second_units / 10,
                model_names[2]: third_units / 10,
            }
            combined = sum(
                weights[name] * arrays[name]
                for name in model_names
            )
            threshold_result, threshold_curve = (
                select_binary_threshold(
                    labels,
                    combined[:, 1],
                    objective=objective,
                )
            )
            primary = threshold_result[objective]
            secondary_key = (
                'f1_macro'
                if objective == 'accuracy'
                else 'accuracy'
            )
            equal_distance = sum(
                (weight - 1 / 3) ** 2
                for weight in weights.values()
            )
            rank = (
                primary,
                threshold_result[secondary_key],
                threshold_result['mcc'],
                -equal_distance,
            )
            if best is None or rank > best[0]:
                best = (
                    rank,
                    weights,
                    threshold_result,
                    threshold_curve,
                    combined,
                )

    return {
        'weights': best[1],
        'threshold_result': best[2],
        'threshold_curve': best[3],
        'oof_probs': best[4].tolist(),
        'objective': objective,
    }


def train_binary_final(
        model_name, hf_id, text_col, max_len, full_df, test_df,
        hp, binary_criterion, tokenizer, num_epochs, logger=None,
        needs_global_attention=False):
    """Refit one binary transformer on all real train+validation rows."""
    if logger:
        logger.section(
            f'Final Binary Refit: {model_name} '
            f'({num_epochs} epochs, 100% real train+val)')

    train_loader = DataLoader(
        CDTDataset(
            full_df.reset_index(drop=True),
            tokenizer, text_col, max_len,
            augment=Config.AUGMENTATION.get('enabled', False),
            needs_global_attention=needs_global_attention,
        ),
        batch_size=hp['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        CDTDataset(
            test_df.reset_index(drop=True),
            tokenizer, text_col, max_len,
            needs_global_attention=needs_global_attention,
        ),
        batch_size=hp['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    model = BinaryCDTModel(
        hf_id,
        dropout=hp.get('dropout', 0.3),
        head_dropout=hp.get('head_dropout', 0.1),
        intermediate=hp.get('intermediate', 256),
    ).to(Config.DEVICE)
    optimizer = get_layerwise_optimizer(
        model,
        hp.get('learning_rate', 1.5e-5),
        hp.get('lr_decay', 0.9),
        hp.get('head_lr_mult', 10.0),
        hp.get('weight_decay', 0.01),
    )
    trainer = BinaryTrainer(model, optimizer, hp, logger)
    final_path = os.path.join(
        Config.OUTPUT_DIR, f'{model_name}_binary_final.pt')
    history = trainer.fit_fixed_epochs(
        train_loader, binary_criterion, final_path, num_epochs)
    test_metrics = trainer.evaluate(test_loader, binary_criterion)

    if logger:
        logger.log(
            f'  Binary final — Test F1: '
            f'{test_metrics["f1_macro"]:.4f}')
        logger.log(
            f'  Binary final — Test Acc: '
            f'{test_metrics["accuracy"]:.4f}')

    del model
    torch.cuda.empty_cache()
    return {
        'model_path': final_path,
        'num_epochs': num_epochs,
        'history': history,
        'test_metrics': {
            key: value
            for key, value in test_metrics.items()
            if key not in ('labels', 'preds', 'probs')
        },
        'test_binary_labels': test_metrics['labels'],
        'test_binary_probs': test_metrics['probs'],
    }


def _ordered_probabilities(classifier, probabilities, num_classes):
    """Return predict_proba columns in the fixed 0..num_classes-1 order."""
    ordered = np.zeros((len(probabilities), num_classes), dtype=float)
    for source_col, class_id in enumerate(classifier.classes_):
        ordered[:, int(class_id)] = probabilities[:, source_col]
    return ordered


def train_type_logistic_regression(
        train_df, val_df, full_df, test_df, output_dir, logger=None):
    """Tune and refit the conditional 10-class classifier.

    Every input dataframe passed here must contain only real or reviewed
    training additions as appropriate. Validation and test remain unchanged.
    """
    text_col = 'Patient Question'
    train_distorted = train_df[train_df['label'] != 0].copy()
    val_distorted = val_df[val_df['label'] != 0].copy()
    full_distorted = full_df[full_df['label'] != 0].copy()
    test_distorted = test_df[test_df['label'] != 0].copy()

    if logger:
        logger.section('Conditional Type Model: weighted TF-IDF + LR')
        logger.log(
            f'  Type train rows: {len(train_distorted):,}')
        logger.log(
            f'  Type validation rows: {len(val_distorted):,}')

    vectorizer = TfidfVectorizer(**Config.TYPE_TFIDF_PARAMS)
    train_vectors = vectorizer.fit_transform(
        train_distorted[text_col].fillna(''))
    val_vectors = vectorizer.transform(
        val_distorted[text_col].fillna(''))
    train_labels = train_distorted['label'].to_numpy(dtype=int) - 1
    val_labels = val_distorted['label'].to_numpy(dtype=int) - 1

    candidates = []
    for c_value in Config.TYPE_LR_C_VALUES:
        classifier = LogisticRegression(
            **{**Config.TYPE_LR_PARAMS, 'C': c_value})
        classifier.fit(train_vectors, train_labels)
        probabilities = _ordered_probabilities(
            classifier,
            classifier.predict_proba(val_vectors),
            10,
        )
        predictions = probabilities.argmax(axis=1)
        score = f1_score(
            val_labels, predictions, average='macro',
            zero_division=0)
        candidates.append({
            'C': float(c_value),
            'val_f1_macro': float(score),
        })
        if logger:
            logger.log(
                f'    C={c_value:g} | Type Val F1={score:.4f}')

    best_row = max(
        candidates,
        key=lambda row: (
            row['val_f1_macro'],
            -abs(math.log2(row['C'])),
        ),
    )
    best_c = best_row['C']
    if logger:
        logger.log(
            f'  Selected C={best_c:g} using validation Type F1 '
            f'{best_row["val_f1_macro"]:.4f}')

    final_vectorizer = TfidfVectorizer(**Config.TYPE_TFIDF_PARAMS)
    full_vectors = final_vectorizer.fit_transform(
        full_distorted[text_col].fillna(''))
    test_vectors_all = final_vectorizer.transform(
        test_df[text_col].fillna(''))
    full_labels = full_distorted['label'].to_numpy(dtype=int) - 1
    final_classifier = LogisticRegression(
        **{**Config.TYPE_LR_PARAMS, 'C': best_c})
    final_classifier.fit(full_vectors, full_labels)
    test_type_probs = _ordered_probabilities(
        final_classifier,
        final_classifier.predict_proba(test_vectors_all),
        10,
    )

    distorted_mask = test_df['label'].to_numpy(dtype=int) != 0
    test_type_labels = (
        test_df.loc[distorted_mask, 'label'].to_numpy(dtype=int) - 1)
    conditional_probs = test_type_probs[distorted_mask]
    conditional_preds = conditional_probs.argmax(axis=1)
    conditional_metrics = compute_metrics(
        test_type_labels,
        conditional_preds,
        conditional_probs,
        num_classes=10,
    )

    model_path = os.path.join(
        output_dir, 'conditional_type_tfidf_lr.pkl')
    with open(model_path, 'wb') as handle:
        pickle.dump({
            'vectorizer': final_vectorizer,
            'model': final_classifier,
            'input_text_column': text_col,
            'output_label_offset': 1,
            'class_order': list(range(10)),
        }, handle)

    if logger:
        logger.log(
            f'  Conditional Test Type F1: '
            f'{conditional_metrics["f1_macro"]:.4f}')
        logger.log(f'  Type model saved → {model_path}')

    return {
        'model_path': model_path,
        'selected_c': best_c,
        'validation_candidates': candidates,
        'validation_f1': best_row['val_f1_macro'],
        'test_metrics': conditional_metrics,
        'test_type_probs': test_type_probs.tolist(),
        'training_rows': len(full_distorted),
        'real_test_distorted_rows': len(test_distorted),
    }


def combine_hybrid_probabilities(binary_probs, type_probs):
    """Build a normalized 11-class probability matrix for metrics/ROC-AUC."""
    binary_arr = np.asarray(binary_probs, dtype=float)
    type_arr = np.asarray(type_probs, dtype=float)
    if binary_arr.ndim != 2 or binary_arr.shape[1] != 2:
        raise ValueError('binary_probs must have shape (n_samples, 2)')
    if type_arr.ndim != 2 or type_arr.shape[1] != 10:
        raise ValueError('type_probs must have shape (n_samples, 10)')
    if len(binary_arr) != len(type_arr):
        raise ValueError('binary_probs and type_probs row counts must match')
    return np.column_stack([
        binary_arr[:, 0],
        binary_arr[:, 1:2] * type_arr,
    ])
