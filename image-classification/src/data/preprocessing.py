"""
Preprocessing transforms for image data.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image


class Normalize(nn.Module):
    """Normalize tensor with mean and standard deviation."""

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class Denormalize(nn.Module):
    """Denormalize tensor to original range."""

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


class ResizePad:
    """Resize image with padding to maintain aspect ratio."""

    def __init__(self, size: int = 224, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        scale = self.size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create padded image
        new_img = Image.new("RGB", (self.size, self.size), color=self.fill)
        paste_x = (self.size - new_width) // 2
        paste_y = (self.size - new_height) // 2
        new_img.paste(img, (paste_x, paste_y))

        return new_img


class HistogramEqualization:
    """Apply histogram equalization to images."""

    def __init__(self, clahe: bool = True, clip_limit: float = 2.0):
        self.clahe = clahe
        self.clip_limit = clip_limit

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)

        if self.clahe:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
            if len(img_np.shape) == 3:
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                img_np = clahe.apply(img_np)
        else:
            if len(img_np.shape) == 3:
                ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                img_np = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            else:
                img_np = cv2.equalizeHist(img_np)

        return Image.fromarray(img_np)


class GaussianNoise:
    """Add Gaussian noise to images."""

    def __init__(self, mean: float = 0.0, std: float = 0.1, p: float = 0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() > self.p:
            return img

        img_np = np.array(img).astype(np.float32)
        noise = np.random.normal(self.mean, self.std, img_np.shape)
        img_np = np.clip(img_np + noise * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)


class RandomErasing:
    """Randomly erase rectangular regions in images."""

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: int = 0,
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() > self.p:
            return img

        img_np = np.array(img)
        height, width = img_np.shape[:2]
        area = height * width

        for _ in range(10):
            target_area = np.random.uniform(*self.scale) * area
            log_ratio = np.log(np.array(self.ratio))
            aspect_ratio = np.exp(np.random.uniform(*log_ratio))

            w = int(np.sqrt(target_area * aspect_ratio))
            h = int(np.sqrt(target_area / aspect_ratio))

            if w <= width and h <= height:
                i = np.random.randint(0, height - h + 1)
                j = np.random.randint(0, width - w + 1)

                if len(img_np.shape) == 3:
                    img_np[i:i+h, j:j+w, :] = self.value
                else:
                    img_np[i:i+h, j:j+w] = self.value

                break

        return Image.fromarray(img_np)


def create_preprocessing_pipeline(
    image_size: int = 224,
    normalize: bool = True,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> nn.Sequential:
    """
    Create a preprocessing pipeline.

    Args:
        image_size: Target image size
        normalize: Whether to apply normalization
        mean: Normalization mean values
        std: Normalization std values

    Returns:
        Sequential preprocessing module
    """
    transforms = [
        ResizePad(size=image_size),
    ]

    if normalize:
        transforms.append(Normalize(mean=mean, std=std))

    return nn.Sequential(*transforms)
