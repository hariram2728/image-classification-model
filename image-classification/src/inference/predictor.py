"""
Inference class for image classification predictions.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import numpy as np

from src.utils.logger import get_logger
from src.data.preprocessing import Normalize, ResizePad
from src.data.augmentation import get_val_transforms

logger = get_logger(__name__)


class Predictor:
    """
    High-level inference class for image classification.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        class_names: Optional[List[str]] = None,
        transform=None,
    ):
        """
        Initialize predictor.

        Args:
            model: Trained model for inference
            device: Device to run inference on
            class_names: List of class names (optional)
            transform: Image transformation pipeline
        """
        self.model = model.to(device).eval()
        self.device = device
        self.class_names = class_names
        self.transform = transform or get_val_transforms()

        if class_names:
            self.num_classes = len(class_names)
        else:
            self.num_classes = model.num_classes if hasattr(model, 'num_classes') else model.classifier.out_features

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, any]:
        """
        Make prediction for a single image.

        Args:
            image: Input image (path, PIL Image, or numpy array)

        Returns:
            Dictionary containing prediction results
        """
        # Load and preprocess image
        img_tensor = self._preprocess_image(image)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Get prediction
        logits = self.model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, 1)

        result = {
            "predicted_class": int(predicted.item()),
            "confidence": float(confidence.item()),
            "probabilities": probs.cpu().numpy()[0].tolist(),
        }

        if self.class_names:
            result["predicted_label"] = self.class_names[result["predicted_class"]]
            result["all_predictions"] = [
                {"class": i, "label": self.class_names[i], "probability": float(p)}
                for i, p in enumerate(result["probabilities"])
            ]

        return result

    @torch.no_grad()
    def predict_top_k(
        self,
        image: Union[str, Path, Image.Image],
        k: int = 5,
    ) -> List[Dict[str, any]]:
        """
        Get top-k predictions for an image.

        Args:
            image: Input image
            k: Number of top predictions to return

        Returns:
            List of top-k predictions with class and probability
        """
        img_tensor = self._preprocess_image(image)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        logits = self.model(img_tensor)
        probs = torch.softmax(logits, dim=1)

        # Get top-k
        top_probs, top_indices = torch.topk(probs, k, dim=1)

        results = []
        for i in range(k):
            pred = {
                "rank": i + 1,
                "class": int(top_indices[0, i].item()),
                "probability": float(top_probs[0, i].item()),
            }
            if self.class_names:
                pred["label"] = self.class_names[pred["class"]]
            results.append(pred)

        return results

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
    ) -> List[Dict[str, any]]:
        """
        Make predictions for a batch of images.

        Args:
            images: List of input images
            batch_size: Batch size for inference

        Returns:
            List of prediction results
        """
        all_results = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_tensors = [self._preprocess_image(img) for img in batch_images]
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, 1)

            for j in range(len(batch_images)):
                result = {
                    "predicted_class": int(predicted[j].item()),
                    "confidence": float(confidence[j].item()),
                    "probabilities": probs[j].cpu().numpy().tolist(),
                }
                if self.class_names:
                    result["predicted_label"] = self.class_names[result["predicted_class"]]
                all_results.append(result)

        return all_results

    def _preprocess_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: Input image

        Returns:
            Preprocessed tensor
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self.transform:
            if callable(self.transform):
                # Albumentations transform returns dict with 'image' key
                if hasattr(self.transform, '__call__'):
                    try:
                        transformed = self.transform(image=np.array(image))
                        image = transformed['image']
                    except:
                        image = self.transform(image)
            else:
                image = self.transform(image)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        return image

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = "cuda",
        class_names: Optional[List[str]] = None,
    ) -> "Predictor":
        """
        Create predictor from model checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference
            class_names: List of class names

        Returns:
            Initialized Predictor instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Try to get model config from checkpoint
        model_config = checkpoint.get("model_config", {})

        # Import model factory
        from src.models.model_factory import create_model

        model = create_model(
            model_name=model_config.get("model_name", "resnet50"),
            num_classes=model_config.get("num_classes", 10),
            pretrained=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model=model, device=device, class_names=class_names)
