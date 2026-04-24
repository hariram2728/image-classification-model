"""
Model Export Utility
Exports trained models to deployment-ready formats (PTH, ONNX, TorchScript)
"""
import torch
import torch.nn as nn
import onnx
from pathlib import Path
from typing import Optional, Dict, Any
import json

from src.models.model_factory import create_model
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelExporter:
    """Handles model export to various formats for deployment"""
    
    def __init__(self, model_name: str, num_classes: int, config_path: str = "configs/model_config.yaml"):
        self.model_name = model_name
        self.num_classes = num_classes
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load model from factory or checkpoint"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Create model architecture
        self.model = create_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            pretrained=False
        )
        
        # Load weights if checkpoint provided
        if checkpoint_path:
            logger.info(f"Loading weights from: {checkpoint_path}")
            # Secure loading with weights_only=True
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")
        return self.model
    
    def export_to_pth(self, output_path: str, metadata: Optional[Dict[str, Any]] = None):
        """Export to PyTorch .pth format"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint with metadata
        checkpoint = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'state_dict': self.model.state_dict(),
            'config': self.config,
            'metadata': metadata or {}
        }
        
        # Save with weights_only compatible format
        torch.save(checkpoint, output_path)
        logger.info(f"Model exported to PTH: {output_path}")
        return output_path
    
    def export_to_onnx(self, output_path: str, input_size: tuple = (1, 3, 224, 224), opset_version: int = 11):
        """Export to ONNX format for cross-platform inference"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(input_size, device=self.device)
        
        # Set model to eval mode
        self.model.eval()
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"Model exported to ONNX: {output_path}")
        return output_path
    
    def export_to_torchscript(self, output_path: str, method: str = 'trace'):
        """Export to TorchScript for optimized Python-free inference"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        
        if method == 'trace':
            # Tracing method - good for static control flow
            dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
            scripted_model = torch.jit.trace(self.model, dummy_input)
        elif method == 'script':
            # Scripting method - good for dynamic control flow
            scripted_model = torch.jit.script(self.model)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")
        
        scripted_model.save(str(output_path))
        logger.info(f"Model exported to TorchScript: {output_path}")
        return output_path
    
    def export_all_formats(self, output_dir: str, metadata: Optional[Dict[str, Any]] = None):
        """Export model to all supported formats"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exports = {}
        
        # Export to PTH
        pth_path = output_dir / f"{self.model_name}.pth"
        exports['pth'] = str(self.export_to_pth(pth_path, metadata))
        
        # Export to ONNX
        onnx_path = output_dir / f"{self.model_name}.onnx"
        exports['onnx'] = str(self.export_to_onnx(onnx_path))
        
        # Export to TorchScript
        ts_path = output_dir / f"{self.model_name}.pt"
        exports['torchscript'] = str(self.export_to_torchscript(ts_path))
        
        # Save class labels if available
        if metadata and 'class_labels' in metadata:
            labels_path = output_dir / "class_labels.json"
            with open(labels_path, 'w') as f:
                json.dump(metadata['class_labels'], f, indent=2)
            exports['class_labels'] = str(labels_path)
        
        # Save manifest
        manifest_path = output_dir / "export_manifest.json"
        manifest = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'export_timestamp': str(torch.__version__),
            'exports': exports
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        exports['manifest'] = str(manifest_path)
        
        logger.info(f"All formats exported to: {output_dir}")
        return exports


def export_model_from_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    model_name: str,
    num_classes: int,
    metadata: Optional[Dict[str, Any]] = None
):
    """Convenience function to export a model from checkpoint"""
    exporter = ModelExporter(model_name, num_classes)
    exporter.load_model(checkpoint_path)
    return exporter.export_all_formats(output_dir, metadata)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export model for deployment")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="artifacts/models", help="Output directory")
    parser.add_argument("--model-name", type=str, default="resnet50", help="Model architecture name")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--metadata", type=str, help="Path to metadata JSON file")
    
    args = parser.parse_args()
    
    # Load metadata if provided
    metadata = None
    if args.metadata:
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
    
    # Export model
    exports = export_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_classes=args.num_classes,
        metadata=metadata
    )
    
    print("\n✅ Export Complete!")
    print(f"Generated files:")
    for fmt, path in exports.items():
        print(f"  - {fmt}: {path}")
