"""Training callbacks for early stopping and loss tracking."""

import logging
from transformers import TrainerCallback, TrainerState, TrainerControl

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping based on validation loss.
    Patience=2: stop if val loss doesn't improve for 2 evaluations.
    """

    def __init__(self, early_stopping_patience: int = 2):
        self.patience = early_stopping_patience
        self.best_val_loss = float("inf")
        self.no_improve_count = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        val_loss = metrics.get("eval_loss", float("inf"))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.no_improve_count = 0
            logger.info(f"  ✓ New best val_loss: {val_loss:.4f}")
        else:
            self.no_improve_count += 1
            logger.info(
                f"  ✗ No improvement ({self.no_improve_count}/{self.patience}), "
                f"val_loss: {val_loss:.4f}"
            )
            if self.no_improve_count >= self.patience:
                logger.info("Early stopping triggered.")
                control.should_training_stop = True
        return control


class LossLoggerCallback(TrainerCallback):
    """Logs train/val loss per epoch for the paper's Table IV."""

    def __init__(self):
        self.epoch_losses = []

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        train_loss = state.log_history[-1].get("loss", None)
        self.epoch_losses.append({"epoch": int(state.epoch), "train_loss": train_loss})

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if self.epoch_losses:
            self.epoch_losses[-1]["val_loss"] = metrics.get("eval_loss")
            epoch = self.epoch_losses[-1]
            if "train_loss" in epoch and "val_loss" in epoch:
                gap = epoch["val_loss"] - epoch["train_loss"]
                logger.info(
                    f"Epoch {epoch['epoch']} — "
                    f"train: {epoch['train_loss']:.4f}, "
                    f"val: {epoch['val_loss']:.4f}, "
                    f"Δgap: {gap:.4f}"
                )
