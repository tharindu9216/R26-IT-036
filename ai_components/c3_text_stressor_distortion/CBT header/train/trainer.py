"""Training and evaluation utilities for CDT (cognitive distortion) transformer training.

This module trains the hierarchical CBT model. It combines the binary and
conditional type probabilities into one calibrated 11-class distribution for
end-to-end evaluation and deployment.
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup

from config import Config
from utils  import compute_hierarchical_metrics, hard_route_predictions


@contextmanager
def _nullctx():
    yield


class Trainer:

    def __init__(self, model, optimizer, hp, logger=None):
        self.model     = model
        self.optimizer = optimizer
        self.hp        = hp
        self.logger    = logger
        self.device    = Config.DEVICE
        self.scheduler = None

        # Mixed precision
        dtype        = Config.get_amp_dtype()
        self.use_amp = dtype is not None
        self.dtype   = dtype
        self.scaler  = (GradScaler()
                        if self.use_amp and dtype == torch.float16
                        else None)

    def _amp(self):
        return autocast(dtype=self.dtype) if self.use_amp else _nullctx()

    def build_scheduler(self, train_loader, num_epochs=None):
        """
        Cosine schedule with linear warmup.
        Better than linear — smooth LR decrease, better convergence.
        """
        accum        = self.hp.get('accum_steps', 4)
        epochs       = (num_epochs if num_epochs is not None
                        else self.hp.get('num_epochs', 5))
        warmup_ratio = self.hp.get('warmup_ratio', 0.1)
        total        = (len(train_loader) // accum) * epochs
        warmup       = int(total * warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup, total)

    @staticmethod
    def _forward(model, batch, device):
        """Run the shared encoder and return binary/type logits."""
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        if 'global_attention_mask' in batch:
            global_mask = batch['global_attention_mask'].to(device)
            return model(ids, mask, global_attention_mask=global_mask)
        return model(ids, mask)

    @staticmethod
    def combine_probabilities(binary_logits, type_logits):
        """
        Convert both heads into an end-to-end 11-class distribution.

        Column 0 is P(No Distortion). Columns 1-10 are
        P(Distortion) * P(type | Distortion).
        """
        binary_probs = torch.softmax(binary_logits.float(), dim=1)
        type_probs = torch.softmax(type_logits.float(), dim=1)
        return torch.cat(
            [binary_probs[:, :1], binary_probs[:, 1:2] * type_probs],
            dim=1,
        )

    def _loss_and_probabilities(
            self, batch, binary_criterion, type_criterion):
        labels = batch['label'].to(self.device)
        binary_labels = batch['binary_label'].to(self.device)
        type_labels = batch['type_label'].to(self.device)

        binary_logits, type_logits = self._forward(
            self.model, batch, self.device)
        binary_loss = binary_criterion(binary_logits, binary_labels)

        distorted = type_labels != -100
        if distorted.any():
            type_loss = type_criterion(
                type_logits[distorted], type_labels[distorted])
        else:
            # Preserve a differentiable zero if a batch contains only
            # No-Distortion examples.
            type_loss = type_logits.sum() * 0.0

        type_weight = self.hp.get('type_loss_weight', 1.0)
        loss = binary_loss + type_weight * type_loss
        joint_probs = self.combine_probabilities(binary_logits, type_logits)
        return loss, labels, joint_probs

    # Train one epoch
    def train_epoch(self, loader, binary_criterion, type_criterion):
        """Train both weighted heads with fp16/bf16 and accumulation."""
        self.model.train()
        accum = self.hp.get('accum_steps', 4)
        clip  = self.hp.get('max_grad_norm', 1.0)

        total_loss = 0.0
        preds_all, labels_all, probs_all = [], [], []
        self.optimizer.zero_grad()

        for step, batch in enumerate(loader):
            with self._amp():
                raw_loss, labels, probs = self._loss_and_probabilities(
                    batch, binary_criterion, type_criterion)
                loss = raw_loss / accum

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

            total_loss += loss.item() * accum

            probs = probs.detach().cpu().numpy()
            threshold = self.hp.get('binary_threshold', 0.5)
            preds = hard_route_predictions(probs, threshold)
            probs_all.extend(probs.tolist())
            preds_all.extend(preds.tolist())
            labels_all.extend(labels.cpu().numpy())

        return (total_loss / len(loader),
                compute_hierarchical_metrics(
                    labels_all, preds_all, probs_all,
                    binary_threshold=self.hp.get(
                        'binary_threshold', 0.5)))

    # Evaluate
    def evaluate(self, loader, binary_criterion, type_criterion):
        """Evaluate binary, conditional type, and end-to-end performance."""
        self.model.eval()

        total_loss = 0.0
        preds_all, labels_all, probs_all = [], [], []

        with torch.no_grad():
            for batch in loader:
                with self._amp():
                    loss, labels, probs = self._loss_and_probabilities(
                        batch, binary_criterion, type_criterion)
                total_loss += loss.item()

                probs = probs.cpu().numpy()
                preds = hard_route_predictions(
                    probs, self.hp.get('binary_threshold', 0.5))
                probs_all.extend(probs.tolist())
                preds_all.extend(preds.tolist())
                labels_all.extend(labels.cpu().numpy())

        return {
            'loss'  : total_loss / len(loader),
            'preds' : preds_all,
            'labels': labels_all,
            'probs' : probs_all,
            **compute_hierarchical_metrics(
                labels_all, preds_all, probs_all,
                binary_threshold=self.hp.get(
                    'binary_threshold', 0.5)),
        }

    # Full training loop with early stopping
    def fit(self, train_loader, val_loader,
            binary_criterion, type_criterion, save_path):
        """
        Early stopping (patience from hp).
        Saves best model weights to save_path.
        Returns best_val_f1, best_val_metrics, history.
        """
        patience         = self.hp.get('patience', 3)
        min_epochs       = self.hp.get('min_epochs', 5)
        num_epochs       = self.hp.get('num_epochs', 5)
        patience_counter = 0
        best_f1          = float('-inf')
        best_metrics     = {}
        history          = {'train_loss': [], 'val_f1': [], 'val_loss': []}

        self.build_scheduler(train_loader, num_epochs=num_epochs)

        for epoch in range(num_epochs):
            tr_loss, tr_m = self.train_epoch(
                train_loader, binary_criterion, type_criterion)
            val_m = self.evaluate(
                val_loader, binary_criterion, type_criterion)

            history['train_loss'].append(tr_loss)
            history['val_f1'].append(val_m['f1_macro'])
            history['val_loss'].append(val_m['loss'])

            if self.logger:
                self.logger.log(
                    f'    Epoch {epoch+1}/{num_epochs} | '
                    f'Loss: {tr_loss:.4f} | '
                    f'Val F1: {val_m["f1_macro"]:.4f} | '
                    f'Binary F1: {val_m["binary_f1_macro"]:.4f} | '
                    f'Type F1: {val_m["type_f1_macro"]:.4f} | '
                    f'Val Acc: {val_m["accuracy"]:.4f} | '
                    f'Val MCC: {val_m["mcc"]:.4f}'
                )

            if val_m['f1_macro'] > best_f1:
                best_f1          = val_m['f1_macro']
                best_metrics     = val_m
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                if self.logger:
                    self.logger.log(f'     Best saved F1={best_f1:.4f}')
            else:
                patience_counter += 1
                if self.logger:
                    self.logger.log(
                        f'    No improvement ({patience_counter}/{patience})')
                # Always give the conditional type head enough time to learn
                # before allowing early stopping.
                if (epoch + 1 >= min_epochs
                        and patience_counter >= patience):
                    if self.logger:
                        self.logger.log(
                            f'    Early stopping at epoch {epoch+1}')
                    break

        return best_f1, best_metrics, history

    # Fixed-epoch refit — no validation split, no early stopping
    def fit_fixed_epochs(self, train_loader, binary_criterion, type_criterion,
                         save_path, num_epochs):
        """
        Trains for exactly `num_epochs` with no early stopping.

        Used for the final refit on 100% of train+val data, once K-Fold CV
        has already validated the hyperparameters and supplied an epoch
        budget (e.g. the average best-epoch across folds). There is no
        validation split left to monitor at this stage, so the model is
        simply saved after the last epoch.
        """
        history = {'train_loss': []}
        self.build_scheduler(train_loader, num_epochs=num_epochs)

        for epoch in range(num_epochs):
            tr_loss, tr_m = self.train_epoch(
                train_loader, binary_criterion, type_criterion)
            history['train_loss'].append(tr_loss)

            if self.logger:
                self.logger.log(
                    f'    Epoch {epoch+1}/{num_epochs} | '
                    f'Loss: {tr_loss:.4f} | '
                    f'Train F1: {tr_m["f1_macro"]:.4f} | '
                    f'Binary F1: {tr_m["binary_f1_macro"]:.4f} | '
                    f'Type F1: {tr_m["type_f1_macro"]:.4f}'
                )

        torch.save(self.model.state_dict(), save_path)
        if self.logger:
            self.logger.log(f'    Final refit model saved → {save_path}')

        return history
