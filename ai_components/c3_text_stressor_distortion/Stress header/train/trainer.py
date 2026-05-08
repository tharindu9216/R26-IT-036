
import torch
import torch.nn as nn
from contextlib import contextmanager
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup

from config import Config
from utils  import compute_metrics, compute_head1b_metrics


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

    def build_scheduler(self, train_loader):
        """
        Cosine schedule with linear warmup.
        Better than linear — smooth LR decrease, better convergence.
        """
        accum        = self.hp.get('accum_steps', 4)
        epochs       = self.hp.get('num_epochs',  5)
        warmup_ratio = self.hp.get('warmup_ratio', 0.1)
        total        = (len(train_loader) // accum) * epochs
        warmup       = int(total * warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup, total)

    # =========================================================================
    # Train one epoch
    # =========================================================================
    def train_epoch(self, loader, crit_1a, crit_1b):
        """
        fp16 autocast + gradient accumulation.
        loss = (0.6 × loss_1a + 0.4 × loss_1b) / accum_steps
        """
        self.model.train()
        alpha  = self.hp.get('alpha', 0.6)
        beta   = 1.0 - alpha
        accum  = self.hp.get('accum_steps', 4)
        clip   = self.hp.get('max_grad_norm', 1.0)

        total_loss = 0.0
        preds_all, labels_all, probs_all = [], [], []
        self.optimizer.zero_grad()

        for step, batch in enumerate(loader):
            ids   = batch['input_ids'].to(self.device)
            mask  = batch['attention_mask'].to(self.device)
            lab1a = batch['label_1a'].to(self.device)
            lab1b = batch['label_1b'].to(self.device)

            with self._amp():
                lg1a, lg1b = self.model(ids, mask)
                loss = (alpha * crit_1a(lg1a, lab1a) +
                        beta  * crit_1b(lg1b, lab1b)) / accum

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

            probs = torch.softmax(lg1a.float(), dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(lg1a.float(), dim=1).detach().cpu().numpy()
            probs_all.extend(probs)
            preds_all.extend(preds)
            labels_all.extend(lab1a.cpu().numpy())

        return (total_loss / len(loader),
                compute_metrics(labels_all, preds_all, probs_all))

    # =========================================================================
    # Evaluate
    # =========================================================================
    def evaluate(self, loader, crit_1a, crit_1b):
        """
        Returns metrics for Head 1A (primary) + Head 1B accuracy.
        probs = softmax(logits_1a)[:, 1]  ← stressed probability
        """
        self.model.eval()
        alpha = self.hp.get('alpha', 0.6)
        beta  = 1.0 - alpha

        total_loss = 0.0
        p1a, l1a, pr1a = [], [], []
        p1b, l1b       = [], []

        with torch.no_grad():
            for batch in loader:
                ids   = batch['input_ids'].to(self.device)
                mask  = batch['attention_mask'].to(self.device)
                lab1a = batch['label_1a'].to(self.device)
                lab1b = batch['label_1b'].to(self.device)

                with self._amp():
                    lg1a, lg1b = self.model(ids, mask)
                    loss = (alpha * crit_1a(lg1a, lab1a) +
                            beta  * crit_1b(lg1b, lab1b))
                total_loss += loss.item()

                pr1a.extend(torch.softmax(lg1a.float(), dim=1)[:, 1].cpu().numpy())
                p1a.extend(torch.argmax(lg1a.float(), dim=1).cpu().numpy())
                l1a.extend(lab1a.cpu().numpy())
                p1b.extend(torch.argmax(lg1b.float(), dim=1).cpu().numpy())
                l1b.extend(lab1b.cpu().numpy())

        return {
            'loss'  : total_loss / len(loader),
            'preds' : p1a,
            'labels': l1a,
            'probs' : pr1a,
            **compute_metrics(l1a, p1a, pr1a),
            **compute_head1b_metrics(l1b, p1b),
        }

    # =========================================================================
    # Full training loop with early stopping
    # =========================================================================
    def fit(self, train_loader, val_loader, crit_1a, crit_1b, save_path):
        """
        Early stopping (patience from hp).
        Saves best model weights to save_path.
        Returns best_val_f1, best_val_metrics, history.
        """
        patience         = self.hp.get('patience', 2)
        num_epochs       = self.hp.get('num_epochs', 5)
        patience_counter = 0
        best_f1          = 0.0
        best_metrics     = {}
        history          = {'train_loss': [], 'val_f1': [], 'val_loss': []}

        self.build_scheduler(train_loader)

        for epoch in range(num_epochs):
            tr_loss, tr_m = self.train_epoch(train_loader, crit_1a, crit_1b)
            val_m         = self.evaluate(val_loader,   crit_1a, crit_1b)

            history['train_loss'].append(tr_loss)
            history['val_f1'].append(val_m['f1_macro'])
            history['val_loss'].append(val_m['loss'])

            if self.logger:
                self.logger.log(
                    f'    Epoch {epoch+1}/{num_epochs} | '
                    f'Loss: {tr_loss:.4f} | '
                    f'Val F1: {val_m["f1_macro"]:.4f} | '
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
                if patience_counter >= patience:
                    if self.logger:
                        self.logger.log(
                            f'    Early stopping at epoch {epoch+1}')
                    break

        return best_f1, best_metrics, history
