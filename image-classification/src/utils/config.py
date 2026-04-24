"""
Configuration management utilities.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Base configuration class."""

    environment: str = "development"
    debug: bool = False
    seed: int = 42

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        with open(output_path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


@dataclass
class DataConfig(Config):
    """Data pipeline configuration."""

    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    augmented_data_dir: str = "./data/augmented"

    # Image settings
    image_size: int = 224
    num_channels: int = 3
    normalize_mean: tuple = field(default_factory=lambda: (0.485, 0.456, 0.406))
    normalize_std: tuple = field(default_factory=lambda: (0.229, 0.224, 0.225))

    # Batch sizes
    train_batch_size: int = 32
    val_batch_size: int = 32
    test_batch_size: int = 32

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True

    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5


@dataclass
class ModelConfig(Config):
    """Model architecture configuration."""

    model_name: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 10
    dropout_rate: float = 0.5

    # Architecture-specific
    efficientnet_variant: str = "b0"
    vit_patch_size: int = 16
    vit_hidden_dim: int = 768
    vit_num_layers: int = 12
    vit_num_heads: int = 12


@dataclass
class TrainingConfig(Config):
    """Training hyperparameters configuration."""

    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "adamw"

    # Learning rate scheduler
    scheduler: str = "cosine"
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.1
    warmup_epochs: int = 5

    # Training loop
    num_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_value: float = 1.0

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True
    save_frequency: int = 5

    # Logging
    log_frequency: int = 10
    tensorboard_log_dir: str = "./logs/tensorboard"
    mlflow_tracking_uri: str = "http://localhost:5000"


@dataclass
class InferenceConfig(Config):
    """Inference settings configuration."""

    model_path: str = "./models/best_model.pth"
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 4

    # Post-processing
    apply_softmax: bool = True
    top_k: int = 5
    confidence_threshold: float = 0.5

    # Export formats
    export_onnx: bool = True
    export_torchscript: bool = True


def load_config(config_path: str, config_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file
        config_type: Type of config ('data', 'model', 'training', 'inference')

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config_type:
        return config.get(config_type, config)
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def get_env_config() -> Dict[str, Any]:
    """
    Get configuration from environment variables.

    Returns:
        Configuration dictionary from environment
    """
    return {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "model_path": os.getenv("MODEL_PATH", "./models"),
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }
