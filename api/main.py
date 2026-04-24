"""
FastAPI Backend for Image Classification
Deploy this folder separately on Render/Railway/Hugging Face
"""
import os
import io
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="Professional image classification service",
    version="1.0.0"
)

# CORS middleware - Update with your Streamlit Cloud URL in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific URL in production: ["https://your-app.streamlit.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
transform = None
class_labels = None
device = None

def load_model():
    """Load model once at startup"""
    global model, transform, class_labels, device
    
    model_path = os.getenv("MODEL_PATH", "models/model.pth")
    labels_path = os.getenv("LABELS_PATH", "models/class_labels.json")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load class labels
    import json
    try:
        with open(labels_path, 'r') as f:
            class_labels = json.load(f)
        logger.info(f"Loaded {len(class_labels)} class labels")
    except FileNotFoundError:
        # Default labels for testing
        class_labels = ["cat", "dog", "bird"]
        logger.warning("Using default class labels")
    
    # Load model
    try:
        # Adjust this based on your actual model architecture
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_labels))
        
        # Load trained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}, using pretrained weights only")
        
        model.to(device)
        model.eval()
        
        # Define transforms (match training preprocessing)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict image class from uploaded file
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
        
        # Prepare response
        result = {
            "prediction": class_labels[predicted_idx.item()],
            "confidence": round(confidence.item() * 100, 2),
            "all_probabilities": {
                class_labels[i]: round(prob.item() * 100, 2) 
                for i, prob in enumerate(probabilities)
            }
        }
        
        logger.info(f"Prediction: {result['prediction']} ({result['confidence']}%)")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Image Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
