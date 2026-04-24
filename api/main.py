import os
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging

# --- Configuration ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/demo_model.pth")
LABELS_PATH = os.getenv("LABELS_PATH", "models/class_labels.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Variables ---
model = None
class_labels = []
transform = None

app = FastAPI(title="Image Classification API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    global model, class_labels, transform
    
    logger.info(f"Loading model from {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Did setup_demo.py run?")
    
    # 1. Load Labels
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels not found at {LABELS_PATH}")
        
    with open(LABELS_PATH, "r") as f:
        class_labels = json.load(f)
    logger.info(f"Loaded {len(class_labels)} class labels.")
    
    # 2. Initialize Model Architecture (MUST MATCH setup_demo.py)
    # We use ResNet18 here because setup_demo.py downloads ResNet18
    logger.info("Initializing ResNet18 architecture...")
    model = models.resnet18(weights=None)
    
    # Ensure final layer matches number of classes (1000 for ImageNet)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(class_labels))
    
    # 3. Load Weights
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint)
        logger.info("Weights loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        raise e

    model.to(DEVICE)
    model.eval()
    
    # 4. Define Transforms (Must match ImageNet training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    logger.info(f"Model ready on {DEVICE}")

@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

class PredictionResult(BaseModel):
    label: str
    confidence: float
    top_5: List[dict]

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Security: Check file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be an image.")
    
    try:
        contents = await file.read()
        
        # Security: Check size
        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Max {MAX_UPLOAD_SIZE // 1024 // 1024}MB.")
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 5
            top5_prob, top5_idx = torch.topk(probabilities, 5)
            
            results = [
                {"label": class_labels[idx.item()], "confidence": round(prob.item(), 4)}
                for idx, prob in zip(top5_idx, top5_prob)
            ]
        
        return PredictionResult(
            label=results[0]["label"],
            confidence=results[0]["confidence"],
            top_5=results
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@app.get("/health")
async def health():
    # Return 200 OK as long as the server is running, 
    # regardless of model loading status
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "device": DEVICE,
        "message": "Server running" if model else "Model still loading..."
    }
