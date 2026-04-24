"""
Data augmentation pipelines for training.
"""

import random
from typing import Dict, List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

from src.utils.config import DataConfig


def get_train_transforms(
    image_size: int = 224,
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    augmentation_level: str = "medium",
) -> A.Compose:
    """
    Get training data augmentation transforms.

    Args:
        image_size: Target image size
        normalize_mean: Mean values for normalization
        normalize_std: Standard deviation values for normalization
        augmentation_level: Level of augmentation ('none', 'light', 'medium', 'heavy')

    Returns:
        Albumentations compose object with transforms
    """
    # Base transforms applied to all levels
    base_transforms = [
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
    ]

    # Level-specific augmentations
    if augmentation_level == "none":
        aug_transforms = []
    elif augmentation_level == "light":
        aug_transforms = [
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ]
    elif augmentation_level == "medium":
        aug_transforms = [
            A.Rotate(limit=30, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        ]
    elif augmentation_level == "heavy":
        aug_transforms = [
            A.Rotate(limit=45, p=0.8),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.7),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        ]
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")

    # Final transforms (always applied)
    final_transforms = [
        A.Normalize(mean=normalize_mean, std=normalize_std),
        ToTensorV2(),
    ]

    return A.Compose(base_transforms + aug_transforms + final_transforms)


def get_val_transforms(
    image_size: int = 224,
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Get validation/test data transforms (no augmentation).

    Args:
        image_size: Target image size
        normalize_mean: Mean values for normalization
        normalize_std: Standard deviation values for normalization

    Returns:
        Albumentations compose object with transforms
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=normalize_mean, std=normalize_std),
        ToTensorV2(),
    ])


class AdvancedAugmentation:
    """
    Advanced augmentation strategies for improved generalization.
    """

    @staticmethod
    def mixup(
        images: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.4,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp augmentation.

        Args:
            images: Batch of images
            targets: Batch of labels
            alpha: MixUp concentration parameter

        Returns:
            Mixed images, mixed targets, and lambda value
        """
        import torch

        lamb = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        mixed_images = lamb * images + (1 - lamb) * images[index]
        mixed_targets = lamb * targets + (1 - lamb) * targets[index]

        return mixed_images, mixed_targets, lamb

    @staticmethod
    def cutmix(
        images: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation.

        Args:
            images: Batch of images
            targets: Batch of labels
            alpha: CutMix concentration parameter

        Returns:
            Mixed images, mixed targets, and lambda value
        """
        import torch

        lamb = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        # Generate random bounding box
        bbx1, bby1, bbx2, bby2 = _rand_bbox(
            img_size=images.shape[-1],
            lam=lamb,
        )

        # Apply cutmix
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.shape[-1] * images.shape[-2]))

        mixed_targets = lam * targets + (1 - lam) * targets[index]

        return images, mixed_targets, lam

    @staticmethod
    def randaugment(
        image: np.ndarray,
        n: int = 2,
        m: int = 9,
    ) -> np.ndarray:
        """
        Apply RandAugment strategy.

        Args:
            image: Input image
            n: Number of transformations to apply
            m: Magnitude of transformations (0-9)

        Returns:
            Augmented image
        """
        # Available transformations
        transforms = [
            A.Equalize,
            A.InvertImg,
            A.Posterize,
            A.Solarize,
            A.Superpixels,
            A.ChannelShuffle,
            A.Downscale,
        ]

        # Randomly select n transforms
        selected = random.sample(transforms, min(n, len(transforms)))

        for transform in selected:
            # Map magnitude to appropriate range
            if transform == A.Posterize:
                aug = transform(bits=max(1, int(4 * m / 9)), p=0.5)
            elif transform == A.Solarize:
                aug = transform(threshold=int(255 * m / 9), p=0.5)
            else:
                aug = transform(p=0.5)

            image = aug(image=image)["image"]

        return image


def _rand_bbox(img_size: int, lam: float) -> Tuple[int, int, int, int]:
    """Generate random bounding box for CutMix."""
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(img_size * cut_rat)
    cut_h = int(img_size * cut_rat)

    cx = np.random.randint(img_size)
    cy = np.random.randint(img_size)

    bbx1 = np.clip(cx - cut_w // 2, 0, img_size)
    bby1 = np.clip(cy - cut_h // 2, 0, img_size)
    bbx2 = np.clip(cx + cut_w // 2, 0, img_size)
    bby2 = np.clip(cy + cut_h // 2, 0, img_size)

    return bbx1, bby1, bbx2, bby2


def create_augmentation_pipeline(config: Optional[DataConfig] = None) -> Dict[str, A.Compose]:
    """
    Create complete augmentation pipeline from config.

    Args:
        config: Data configuration object

    Returns:
        Dictionary containing train and validation transforms
    """
    if config is None:
        config = DataConfig()

    return {
        "train": get_train_transforms(
            image_size=config.image_size,
            normalize_mean=config.normalize_mean,
            normalize_std=config.normalize_std,
        ),
        "val": get_val_transforms(
            image_size=config.image_size,
            normalize_mean=config.normalize_mean,
            normalize_std=config.normalize_std,
        ),
    }
