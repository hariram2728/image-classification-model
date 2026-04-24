"""
Dataset loading utilities for image classification.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from src.utils.logger import get_logger
from src.utils.config import DataConfig

logger = get_logger(__name__)


class ImageClassificationDataset(Dataset):
    """
    Custom PyTorch Dataset for image classification.

    Args:
        root_dir: Root directory containing images
        transform: Transformations to apply to images
        extensions: Valid image file extensions
        recursive: Search subdirectories recursively
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        recursive: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions
        self.recursive = recursive

        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        self._load_samples()

    def _load_samples(self) -> None:
        """Load image paths and labels from directory structure."""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        # Scan directories for classes
        class_dirs = sorted(
            [d for d in self.root_dir.iterdir() if d.is_dir()]
        )

        if not class_dirs:
            raise ValueError(f"No class directories found in {self.root_dir}")

        # Build class index mapping
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx
            self.idx_to_class[idx] = class_dir.name

        logger.info(f"Found {len(self.class_to_idx)} classes")

        # Load samples
        pattern = "**/*" if self.recursive else "*"
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]

            for ext in self.extensions:
                if self.recursive:
                    img_paths = class_dir.rglob(f"*{ext}")
                else:
                    img_paths = class_dir.glob(f"*{ext}")

                for img_path in img_paths:
                    self.samples.append((img_path, class_idx))

        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image on error
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.

        Returns:
            Tensor of class weights
        """
        class_counts = torch.zeros(len(self.class_to_idx))
        for _, label in self.samples:
            class_counts[label] += 1

        # Avoid division by zero
        class_counts = torch.clamp(class_counts, min=1)

        # Calculate weights as inverse frequency
        weights = class_counts.sum() / (len(self.class_to_idx) * class_counts)
        return weights

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of samples per class.

        Returns:
            Dictionary mapping class names to sample counts
        """
        distribution = {name: 0 for name in self.class_to_idx.keys()}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            distribution[class_name] += 1
        return distribution


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None,
    train_transform=None,
    val_transform=None,
    config: Optional[DataConfig] = None,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        test_dir: Path to test data directory (optional)
        train_transform: Transformations for training data
        val_transform: Transformations for validation/test data
        config: Data configuration object

    Returns:
        Dictionary containing DataLoaders
    """
    if config is None:
        config = DataConfig()

    dataloaders = {}

    # Training DataLoader
    if os.path.exists(train_dir):
        train_dataset = ImageClassificationDataset(
            root_dir=train_dir,
            transform=train_transform,
            recursive=True,
        )

        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=config.shuffle_train,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True,
        )
        logger.info(f"Created training DataLoader with {len(train_dataset)} samples")

    # Validation DataLoader
    if os.path.exists(val_dir):
        val_dataset = ImageClassificationDataset(
            root_dir=val_dir,
            transform=val_transform,
            recursive=True,
        )

        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        logger.info(f"Created validation DataLoader with {len(val_dataset)} samples")

    # Test DataLoader
    if test_dir and os.path.exists(test_dir):
        test_dataset = ImageClassificationDataset(
            root_dir=test_dir,
            transform=val_transform,
            recursive=True,
        )

        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=config.test_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        logger.info(f"Created test DataLoader with {len(test_dataset)} samples")

    return dataloaders


def load_image(image_path: Union[str, Path], transform=None) -> torch.Tensor:
    """
    Load a single image for inference.

    Args:
        image_path: Path to the image file
        transform: Optional transformations to apply

    Returns:
        Transformed image tensor
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    if transform:
        image = transform(image)

    return image
