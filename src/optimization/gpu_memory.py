"""
GPU Memory Optimization Module.
Implements mixed precision training, gradient checkpointing, and memory-efficient inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from contextlib import contextmanager
import logging
import gc

logger = logging.getLogger(__name__)


class MixedPrecisionTrainer:
    """
    Mixed precision training for reduced GPU memory usage.
    
    Benefits:
    - 2-4x memory reduction
    - Faster training on Tensor Cores
    - Maintains accuracy with loss scaling
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        amp_enabled: bool = True,
        amp_dtype: str = "float16",  # or "bfloat16"
        grad_scaler_init_scale: float = 65536.0,
    ):
        """
        Initialize mixed precision trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            amp_enabled: Enable automatic mixed precision
            amp_dtype: Precision type ("float16" or "bfloat16")
            grad_scaler_init_scale: Initial loss scale factor
        """
        self.model = model
        self.optimizer = optimizer
        self.amp_enabled = amp_enabled and torch.cuda.is_available()
        
        if self.amp_enabled:
            if amp_dtype == "bfloat16":
                if not torch.cuda.is_bf16_supported():
                    logger.warning("bfloat16 not supported, falling back to float16")
                    amp_dtype = "float16"
                self.scaler = torch.cuda.amp.GradScaler(
                    init_scale=grad_scaler_init_scale,
                    growth_interval=2000,
                )
            else:
                self.scaler = torch.cuda.amp.GradScaler(
                    init_scale=grad_scaler_init_scale,
                    growth_interval=2000,
                )
            
            self.dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
            logger.info(f"Mixed precision enabled with dtype={amp_dtype}")
        else:
            self.scaler = None
            self.dtype = torch.float32
            logger.info("Mixed precision disabled")
    
    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        if self.amp_enabled:
            with torch.cuda.amp.autocast(dtype=self.dtype):
                yield
        else:
            yield
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with gradient scaling."""
        if self.amp_enabled and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self):
        """Optimizer step with gradient unscaling."""
        if self.amp_enabled and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get state dict including scaler."""
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Load state dict including scaler."""
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])


class GradientCheckpointingModel(nn.Module):
    """
    Wrapper for gradient checkpointing to reduce memory usage.
    
    Trade-off: Recomputes activations during backward pass to save memory.
    Memory reduction: 3-5x for deep networks.
    Speed impact: ~20-30% slower training.
    """
    
    def __init__(self, module: nn.Module, checkpoint_segments: int = 1):
        """
        Initialize gradient checkpointing wrapper.
        
        Args:
            module: Module to wrap
            checkpoint_segments: Number of segments for checkpointing
        """
        super().__init__()
        self.module = module
        self.checkpoint_segments = checkpoint_segments
        
        # Enable gradient checkpointing
        if hasattr(module, 'gradient_checkpointing_enable'):
            module.gradient_checkpointing_enable()
        else:
            # Manual checkpointing for custom models
            self._apply_checkpointing()
    
    def _apply_checkpointing(self):
        """Apply checkpointing to submodules."""
        if hasattr(self.module, 'layers'):
            for i, layer in enumerate(self.module.layers):
                if i % self.checkpoint_segments == 0 and i > 0:
                    layer._apply_gradient_checkpointing = True
    
    def forward(self, *args, **kwargs):
        """Forward pass with checkpointing."""
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self.module,
                *args,
                use_reentrant=False,
                **kwargs
            )
        else:
            return self.module(*args, **kwargs)


class MemoryEfficientInference:
    """
    Memory-efficient inference with batch splitting and cleanup.
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 32,
        target_memory_gb: float = 10.0,
    ):
        """
        Initialize memory-efficient inference.
        
        Args:
            model: Model for inference
            max_batch_size: Maximum batch size before splitting
            target_memory_gb: Target GPU memory usage in GB
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.target_memory_bytes = target_memory_gb * 1024**3
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run inference with automatic batch splitting.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output predictions
        """
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        
        total_samples = len(inputs)
        
        if total_samples <= self.max_batch_size:
            return self._predict_batch(inputs)
        
        # Split into smaller batches
        outputs = []
        batch_size = min(
            self.max_batch_size,
            self._estimate_optimal_batch_size(inputs[0:1])
        )
        
        for i in range(0, total_samples, batch_size):
            batch = inputs[i:i+batch_size]
            output = self._predict_batch(batch)
            outputs.append(output)
            
            # Cleanup after each batch
            del batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.cat(outputs, dim=0)
    
    def _predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Predict on a single batch."""
        with torch.no_grad():
            return self.model(batch)
    
    def _estimate_optimal_batch_size(self, sample: torch.Tensor) -> int:
        """Estimate optimal batch size based on available memory."""
        if not torch.cuda.is_available():
            return self.max_batch_size
        
        try:
            mem_info = torch.cuda.mem_get_info()
            free_memory = mem_info[0]
            
            # Estimate memory per sample
            sample_mem = sample.element_size() * sample.nelement()
            
            # Reserve 20% for overhead
            available = free_memory * 0.8
            optimal_batch = int(available / (sample_mem * 10))  # Safety factor
            
            return min(optimal_batch, self.max_batch_size)
        except Exception:
            return self.max_batch_size
    
    def cleanup(self):
        """Force GPU memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


def enable_memory_efficient_mode(
    model: nn.Module,
    training: bool = True,
) -> nn.Module:
    """
    Apply all memory optimizations to a model.
    
    Args:
        model: PyTorch model
        training: Whether in training mode
        
    Returns:
        Optimized model
    """
    # Enable channels_last memory format for better performance
    if training:
        model = model.to(memory_format=torch.channels_last)
    
    # Compile model for faster execution (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode="reduce-overhead")
    
    logger.info("Memory efficient mode enabled")
    
    return model


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "cached": 0, "free": 0}
    
    allocated = torch.cuda.memory_allocated() / 1024**2
    cached = torch.cuda.memory_reserved() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    
    return {
        "allocated_mb": allocated,
        "cached_mb": cached,
        "free_mb": total - cached,
        "total_mb": total,
        "utilization_pct": (allocated / total) * 100,
    }


def profile_memory_usage(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Profile memory usage for a model.
    
    Args:
        model: Model to profile
        input_shape: Shape of input tensor (without batch dimension)
        device: Device to run profiling
        
    Returns:
        Memory usage statistics
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    model = model.to(device)
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Warmup
    with torch.no_grad():
        _ = model(dummy_input)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Profile
    with torch.no_grad():
        output = model(dummy_input)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    current_memory = torch.cuda.memory_allocated() / 1024**2
    
    return {
        "peak_memory_mb": peak_memory,
        "current_memory_mb": current_memory,
        "output_shape": list(output.shape),
        "device": device,
    }


# Training loop integration example
def create_optimized_training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    use_mixed_precision: bool = True,
    use_gradient_checkpointing: bool = True,
) -> Tuple[MixedPrecisionTrainer, nn.Module]:
    """Create optimized training components."""
    
    # Apply gradient checkpointing
    if use_gradient_checkpointing:
        model = GradientCheckpointingModel(model)
    
    # Setup mixed precision
    mp_trainer = MixedPrecisionTrainer(
        model=model,
        optimizer=optimizer,
        amp_enabled=use_mixed_precision,
    )
    
    return mp_trainer, model
