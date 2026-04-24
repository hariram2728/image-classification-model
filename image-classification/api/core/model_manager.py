"""
Model manager for loading and caching models.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
import threading
import os

from src.utils.logger import get_logger
from src.models.model_factory import create_model
from src.inference.predictor import Predictor

logger = get_logger(__name__)


class ModelManager:
    """
    Manages model loading, caching, and inference.
    
    SECURITY FIXES:
    - Uses weights_only=True to prevent pickle deserialization attacks
    - Validates model path to prevent path traversal
    - Sanitizes checkpoint data before use
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self.model: Optional[nn.Module] = None
        self.predictor: Optional[Predictor] = None
        self.model_info: Dict[str, Any] = {}
        self._initialized = True

    def _sanitize_path(self, model_path: str) -> Path:
        """
        Sanitize and validate model path to prevent path traversal attacks.
        
        Args:
            model_path: Raw model path
            
        Returns:
            Validated and resolved Path object
            
        Raises:
            ValueError: If path is invalid or attempts path traversal
        """
        # Resolve to absolute path
        resolved_path = Path(model_path).resolve()
        
        # Get the project root (parent of api directory)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        
        # Ensure the path is within the project directory
        try:
            resolved_path.relative_to(project_root)
        except ValueError:
            # Allow paths in /models directory at project root
            models_dir = project_root / "models"
            if not str(resolved_path).startswith(str(models_dir)):
                raise ValueError(
                    f"Model path must be within project directory or models folder. "
                    f"Got: {resolved_path}"
                )
        
        return resolved_path

    def load_model(
        self,
        model_path: str,
        device: str = "cuda",
        class_names: Optional[list] = None,
    ) -> None:
        """
        Load model from checkpoint.
        
        SECURITY: Uses weights_only=True to prevent arbitrary code execution
        via malicious pickle data in model files.

        Args:
            model_path: Path to model checkpoint
            device: Device to load model on
            class_names: List of class names
            
        Raises:
            FileNotFoundError: If model checkpoint not found
            ValueError: If path is invalid or security validation fails
            RuntimeError: If model loading fails
        """
        logger.info(f"Loading model from {model_path}")

        # Sanitize and validate path
        safe_path = self._sanitize_path(model_path)
        
        if not safe_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {safe_path}")

        try:
            # SECURITY FIX: Use weights_only=True to prevent pickle deserialization attacks
            # This only works with PyTorch >= 1.13. For older versions, consider using
            # torch.load with pickle=False or switch to safe serialization formats like ONNX
            checkpoint = torch.load(
                str(safe_path),
                map_location=device,
                weights_only=True,  # CRITICAL: Prevents arbitrary code execution
            )
            
            # Validate checkpoint structure
            if not isinstance(checkpoint, dict):
                raise ValueError("Invalid checkpoint format: expected a dictionary")
            
            # Safely extract model config with defaults
            model_config = checkpoint.get("model_config", {})
            if not isinstance(model_config, dict):
                model_config = {}
                
            model_name = model_config.get("model_name", "resnet50")
            num_classes = model_config.get("num_classes", 10)
            
            # Validate extracted values
            if not isinstance(model_name, str):
                model_name = "resnet50"
            if not isinstance(num_classes, int) or num_classes <= 0:
                num_classes = 10

            # Create model
            self.model = create_model(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=False,
            )

            # Load weights
            state_dict = checkpoint.get("model_state_dict")
            if state_dict is None:
                # Try alternative key names
                state_dict = checkpoint.get("state_dict") or checkpoint
            
            if state_dict is None:
                raise ValueError("No model weights found in checkpoint")
            
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()

            # Create predictor
            self.predictor = Predictor(
                model=self.model,
                device=device,
                class_names=class_names,
            )

            # Safely store model info with type validation
            epoch = checkpoint.get("epoch", -1)
            val_acc = checkpoint.get("val_acc", 0.0)
            
            self.model_info = {
                "model_name": str(model_name),
                "num_classes": int(num_classes),
                "checkpoint_path": str(safe_path),
                "device": device,
                "epoch": int(epoch) if isinstance(epoch, (int, float)) else -1,
                "val_acc": float(val_acc) if isinstance(val_acc, (int, float)) else 0.0,
            }

            logger.info(f"Model loaded successfully: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.predictor is not None:
            del self.predictor
            self.predictor = None

        logger.info("Model unloaded")

    def predict(self, image) -> Dict[str, Any]:
        """
        Make prediction for an image.

        Args:
            image: Input image

        Returns:
            Prediction results
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        return self.predictor.predict(image)

    def predict_batch(self, images: list, batch_size: int = 32) -> list:
        """
        Make predictions for a batch of images.

        Args:
            images: List of input images
            batch_size: Batch size

        Returns:
            List of prediction results
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        return self.predictor.predict_batch(images, batch_size=batch_size)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self.model_info.copy()  # Return copy to prevent external modification

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
