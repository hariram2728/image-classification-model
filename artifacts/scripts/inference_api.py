"""
Lightweight Inference API for Deployment
Optimized for low-resource environments and cloud deployment
"""
import torch
import io
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image
import base64
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import logging

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Classification API",
    description="Lightweight image classification service",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    """Request model for base64 encoded images"""
    image_base64: str = Field(..., description="Base64 encoded image")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of top predictions")
    
    @validator('image_base64')
    def validate_image(cls, v):
        """Validate base64 image data"""
        if not v:
            raise ValueError("Image data cannot be empty")
        if len(v) > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError("Image too large (max 10MB)")
        return v


class Prediction(BaseModel):
    """Single prediction result"""
    label: str
    confidence: float
    index: int


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    success: bool
    predictions: List[Prediction]
    processing_time_ms: float
    model_name: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str


# Global model cache
_model_cache = {
    'model': None,
    'class_labels': None,
    'device': None,
    'model_name': None
}


def load_model(model_path: str, labels_path: Optional[str] = None):
    """Load model and class labels from files"""
    logger.info(f"Loading model from: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load checkpoint securely
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Extract model state
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            model_name = checkpoint.get('model_name', 'unknown')
            num_classes = checkpoint.get('num_classes', 1000)
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            model_name = checkpoint.get('model_name', 'unknown')
            num_classes = checkpoint.get('num_classes', 1000)
        else:
            state_dict = checkpoint
            model_name = 'unknown'
            num_classes = 1000
    else:
        state_dict = checkpoint
        model_name = 'unknown'
        num_classes = 1000
    
    # Import here to avoid circular dependencies
    from src.models.model_factory import create_model
    
    # Create and load model
    model = create_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load class labels if available
    class_labels = None
    if labels_path and Path(labels_path).exists():
        with open(labels_path, 'r') as f:
            class_labels = json.load(f)
        logger.info(f"Loaded {len(class_labels)} class labels")
    elif num_classes <= 1000:
        # Default ImageNet labels
        class_labels = [f"class_{i}" for i in range(num_classes)]
    
    # Cache model
    _model_cache['model'] = model
    _model_cache['class_labels'] = class_labels
    _model_cache['device'] = device
    _model_cache['model_name'] = model_name
    
    logger.info("Model loaded successfully")
    return True


def preprocess_image(image_data: bytes) -> torch.Tensor:
    """Preprocess image for model input"""
    from torchvision import transforms
    
    # Load image from bytes
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    tensor = transform(image).unsqueeze(0)
    return tensor


def predict_image(tensor: torch.Tensor, top_k: int = 5) -> List[Dict[str, Any]]:
    """Run inference on preprocessed image"""
    import time
    start_time = time.time()
    
    model = _model_cache['model']
    device = _model_cache['device']
    class_labels = _model_cache['class_labels']
    
    # Move to device
    tensor = tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    # Format results
    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        label_idx = idx.item()
        label = class_labels[label_idx] if class_labels else f"class_{label_idx}"
        predictions.append({
            'label': label,
            'confidence': round(prob.item(), 4),
            'index': label_idx
        })
    
    processing_time = (time.time() - start_time) * 1000  # ms
    logger.info(f"Inference completed in {processing_time:.2f}ms")
    
    return predictions


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    model_path = Path("artifacts/models/model.pth")
    labels_path = Path("artifacts/models/class_labels.json")
    
    if model_path.exists():
        load_model(str(model_path), str(labels_path) if labels_path.exists() else None)
    else:
        logger.warning(f"Model not found at {model_path}. API will return errors until model is loaded.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if _model_cache['model'] else "unhealthy",
        model_loaded=_model_cache['model'] is not None,
        device=str(_model_cache['device']) if _model_cache['device'] else "none"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...), top_k: int = Form(default=5)):
    """Predict class for uploaded image"""
    import time
    start_time = time.time()
    
    if not _model_cache['model']:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read file
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    # Preprocess
    try:
        tensor = preprocess_image(contents)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Predict
    predictions = predict_image(tensor, top_k)
    
    processing_time = (time.time() - start_time) * 1000
    
    return PredictionResponse(
        success=True,
        predictions=predictions,
        processing_time_ms=round(processing_time, 2),
        model_name=_model_cache['model_name'] or "unknown"
    )


@app.post("/predict-base64", response_model=PredictionResponse)
async def predict_base64(request: PredictionRequest):
    """Predict class for base64 encoded image"""
    import time
    import base64
    
    if not _model_cache['model']:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Decode base64
    try:
        image_bytes = base64.b64decode(request.image_base64)
    except Exception as e:
        logger.error(f"Base64 decode error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid base64: {str(e)}")
    
    # Preprocess
    try:
        tensor = preprocess_image(image_bytes)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Predict
    predictions = predict_image(tensor, request.top_k)
    
    return PredictionResponse(
        success=True,
        predictions=predictions,
        processing_time_ms=round((time.time() - start_time) * 1000, 2),
        model_name=_model_cache['model_name'] or "unknown"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
