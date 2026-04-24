"""
Main training loop for image classification models.
"""

import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logger import setup_logger, get_logger
from src.utils.config import TrainingConfig, DataConfig, load_config
from src.utils.seed import set_seed
from src.data.data_loader import create_dataloaders
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.models.model_factory import create_model
from src.evaluation.metrics import AccuracyMeter
from src.training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

logger = get_logger(__name__)


class Trainer:
    """
    Main training class for image classification.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or TrainingConfig()

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=0.001,
        )
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=self.config.checkpoint_dir,
            save_best_only=self.config.save_best_only,
        )
        self.lr_scheduler_callback = LearningRateScheduler(scheduler)

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            if self.config.use_amp:
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
                scaler.scale(loss).backward()
                if self.config.gradient_clip_value:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_value,
                    )
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                if self.config.gradient_clip_value:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_value,
                    )
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % self.config.log_frequency == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%',
                })

        avg_loss = total_loss / len(self.train_loader.dataset)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%',
            })

        avg_loss = total_loss / len(self.val_loader.dataset)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, num_epochs: int) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            epoch_time = time.time() - epoch_start

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )

            # Checkpoint
            self.checkpoint(
                model=self.model,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                optimizer=self.optimizer,
            )

            # Early stopping
            if self.early_stopping(val_loss):
                logger.info("Early stopping triggered")
                break

            # LR scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        return self.history


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup logging
    setup_logger(level="INFO")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load config
    config = load_config(args.config) if os.path.exists(args.config) else {}

    # Create data loaders
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    dataloaders = create_dataloaders(
        train_dir=os.path.join(args.data_dir, "raw", "train"),
        val_dir=os.path.join(args.data_dir, "raw", "val"),
        train_transform=train_transforms,
        val_transform=val_transforms,
    )

    # Create model
    model = create_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained=True,
    )
    model.print_summary()

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
    )

    # Train
    history = trainer.train(num_epochs=args.epochs)

    # Save final model
    model.save_checkpoint("models/final_model.pth", optimizer=optimizer)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
