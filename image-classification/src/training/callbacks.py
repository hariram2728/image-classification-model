"""
Training callbacks for monitoring and controlling training.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """
    Early stopping callback to halt training when validation loss stops improving.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value (loss or accuracy)

        Returns:
            True if early stopping is triggered
        """
        if self.best_value is None:
            self.best_value = value
            return False

        # Check for improvement
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class ModelCheckpoint:
    """
    Save model checkpoints during training.
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_best_only: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        save_frequency: int = 1,
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Only save when monitored metric improves
            monitor: Metric to monitor ('val_loss', 'val_acc', etc.)
            mode: 'min' for loss, 'max' for accuracy
            save_frequency: Save every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.save_frequency = save_frequency
        self.best_value = None
        self.current_epoch = 0

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        model: torch.nn.Module,
        epoch: int,
        val_loss: float,
        val_acc: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Save checkpoint if conditions are met.

        Args:
            model: Model to save
            epoch: Current epoch number
            val_loss: Current validation loss
            val_acc: Current validation accuracy
            optimizer: Optimizer state to save
        """
        self.current_epoch = epoch

        # Get monitored value
        if self.monitor == "val_loss":
            current_value = val_loss
        elif self.monitor == "val_acc":
            current_value = val_acc
        else:
            current_value = val_loss

        # Check if we should save
        should_save = False

        if not self.save_best_only:
            if epoch % self.save_frequency == 0:
                should_save = True
        else:
            if self.best_value is None:
                should_save = True
            elif self.mode == "min" and current_value < self.best_value:
                should_save = True
            elif self.mode == "max" and current_value > self.best_value:
                should_save = True

        if should_save:
            self.best_value = current_value
            self._save_checkpoint(model, epoch, val_loss, val_acc, optimizer)

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        val_loss: float,
        val_acc: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """Save model checkpoint to disk."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # Save best model
        best_path = self.checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")

        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, epoch_path)
        logger.info(f"Saved checkpoint to {epoch_path}")


class LearningRateScheduler:
    """
    Learning rate scheduler callback.
    """

    def __init__(
        self,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.scheduler = scheduler
        self.monitor = monitor
        self.mode = mode

    def __call__(self, metric: float):
        """Step the scheduler based on metric."""
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()


class MetricsLogger:
    """
    Log training metrics to file.
    """

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    def log(self, metrics: Dict[str, float], epoch: int):
        """Log metrics for an epoch."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

        # Save to file
        log_path = self.log_dir / "training_history.json"
        import json
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_history(self) -> Dict[str, list]:
        """Get full training history."""
        return self.history


class ProgressBar:
    """
    Custom progress bar callback using tqdm.
    """

    def __init__(self, total: int, desc: str = "Training"):
        from tqdm import tqdm
        self.pbar = tqdm(total=total, desc=desc)
        self.current = 0

    def update(self, n: int = 1, **kwargs):
        """Update progress bar."""
        self.pbar.update(n)
        if kwargs:
            self.pbar.set_postfix(**kwargs)

    def close(self):
        """Close progress bar."""
        self.pbar.close()


class CallbackList:
    """
    Combine multiple callbacks into one.
    """

    def __init__(self, callbacks: list = None):
        self.callbacks = callbacks or []

    def add(self, callback):
        """Add a callback."""
        self.callbacks.append(callback)

    def on_epoch_begin(self, epoch: int):
        """Called at the beginning of each epoch."""
        for callback in self.callbacks:
            if hasattr(callback, "on_epoch_begin"):
                callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        """Called at the end of each epoch."""
        for callback in self.callbacks:
            if hasattr(callback, "on_epoch_end"):
                callback.on_epoch_end(epoch, logs)

    def on_train_begin(self):
        """Called at the beginning of training."""
        for callback in self.callbacks:
            if hasattr(callback, "on_train_begin"):
                callback.on_train_begin()

    def on_train_end(self):
        """Called at the end of training."""
        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end()
