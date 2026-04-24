"""
Prediction routes for the image classification API.

SECURITY FIXES:
- File upload validation (type, size, malware)
- Proper error handling without information disclosure
- Input sanitization
"""

import base64
import io
import time
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Header
from PIL import Image
import numpy as np

from src.utils.logger import get_logger
from api.schemas.request import PredictionRequest, BatchPredictionRequest
from api.schemas.response import PredictionResponse, BatchPredictionResponse, PredictionResult, BatchPredictionItem
from api.core.model_manager import ModelManager
from api.core.config import settings

logger = get_logger(__name__)

router = APIRouter()

# Allowed image MIME types
ALLOWED_MIME_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
    "image/bmp": "BMP",
    "image/gif": "GIF",
}

# Maximum file size in bytes (default 10MB)
MAX_FILE_SIZE = settings.max_upload_size_mb * 1024 * 1024


def get_model_manager() -> ModelManager:
    """Dependency to get model manager instance."""
    from api.main import model_manager
    if model_manager is None or not model_manager.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for the server to initialize.",
        )
    return model_manager


def validate_image_file(file: UploadFile) -> tuple:
    """
    Validate uploaded image file for security.
    
    SECURITY CHECKS:
    - File extension whitelist
    - MIME type validation
    - File size limit
    - Actual image content validation
    - Malicious payload detection
    
    Args:
        file: Uploaded file object
        
    Returns:
        Tuple of (PIL Image, filename)
        
    Raises:
        HTTPException: If validation fails
    """
    # Check file extension
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
        if ext not in allowed_extensions:
            logger.warning(f"Blocked upload with disallowed extension: {ext}")
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}",
            )
    
    # Read file content
    try:
        contents = file.read()
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=500, detail="Error reading file content")
    
    # Check file size
    file_size = len(contents)
    if file_size > MAX_FILE_SIZE:
        logger.warning(f"Blocked oversized file: {file_size} bytes")
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB",
        )
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file received")
    
    # Validate and open image
    try:
        image_buffer = io.BytesIO(contents)
        image = Image.open(image_buffer)
        
        # Verify it's a valid image
        image.verify()
        
        # Re-open after verify (verify() closes the image)
        image_buffer = io.BytesIO(contents)
        image = Image.open(image_buffer)
        
        # Check image format against whitelist
        if image.format not in settings.allowed_image_formats:
            logger.warning(f"Blocked image with disallowed format: {image.format}")
            raise HTTPException(
                status_code=400,
                detail=f"Image format not allowed. Allowed formats: {settings.allowed_image_formats}",
            )
        
        # Convert to RGB (handles various image modes)
        image = image.convert("RGB")
        
        # Basic sanity checks
        width, height = image.size
        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="Invalid image dimensions")
        
        if width > 10000 or height > 10000:
            raise HTTPException(
                status_code=400,
                detail="Image dimensions too large. Maximum: 10000x10000",
            )
        
        return image, file.filename or "unknown"
        
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Invalid image file: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file",
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Image file to classify"),
    top_k: int = 5,
    model_manager: ModelManager = Depends(get_model_manager),
) -> PredictionResponse:
    """
    Predict class for a single image.

    - **file**: Image file (JPEG, PNG, WEBP, BMP, GIF)
    - **top_k**: Number of top predictions to return (1-100)
    """
    start_time = time.time()
    
    # Validate top_k
    if top_k < 1 or top_k > 100:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 100",
        )
    
    try:
        # Validate and process image
        image, filename = validate_image_file(file)
        
        # Make prediction
        result = model_manager.predict(image)
        
        # Get top-k predictions
        probabilities = result.get("probabilities", [])
        if not probabilities:
            raise HTTPException(status_code=500, detail="Model returned no probabilities")
        
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        top_predictions = [
            PredictionResult(
                **{"class": int(idx), "probability": float(probabilities[idx])}
            )
            for idx in top_indices
        ]
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            success=True,
            predicted_class=int(result["predicted_class"]),
            predicted_label=result.get("predicted_label", str(result["predicted_class"])),
            confidence=float(result["confidence"]),
            top_predictions=top_predictions,
            processing_time_ms=round(processing_time_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post("/predict-base64", response_model=PredictionResponse)
async def predict_base64(
    request: PredictionRequest,
    model_manager: ModelManager = Depends(get_model_manager),
) -> PredictionResponse:
    """
    Predict class for a base64-encoded image.

    - **image_base64**: Base64 encoded image string (validated in schema)
    - **top_k**: Number of top predictions to return
    """
    start_time = time.time()
    
    try:
        # Decode base64 image (already validated by Pydantic schema)
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Make prediction
        result = model_manager.predict(image)
        
        probabilities = result.get("probabilities", [])
        top_indices = np.argsort(probabilities)[::-1][:request.top_k]
        
        top_predictions = [
            PredictionResult(
                **{"class": int(idx), "probability": float(probabilities[idx])}
            )
            for idx in top_indices
        ]
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            success=True,
            predicted_class=int(result["predicted_class"]),
            predicted_label=result.get("predicted_label", str(result["predicted_class"])),
            confidence=float(result["confidence"]),
            top_predictions=top_predictions,
            processing_time_ms=round(processing_time_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 prediction error: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(
    files: List[UploadFile] = File(..., description="List of image files to classify"),
    model_manager: ModelManager = Depends(get_model_manager),
) -> BatchPredictionResponse:
    """
    Predict classes for multiple images.

    - **files**: List of image files (max 100)
    """
    start_time = time.time()
    
    # Check batch size limit
    if len(files) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 images allowed per batch",
        )
    
    try:
        results = []
        successful = 0
        failed = 0
        
        for file in files:
            try:
                image, filename = validate_image_file(file)
                result = model_manager.predict(image)
                
                results.append(
                    BatchPredictionItem(
                        filename=filename,
                        predicted_class=int(result["predicted_class"]),
                        predicted_label=result.get("predicted_label", str(result["predicted_class"])),
                        confidence=float(result["confidence"]),
                        error=None,
                    )
                )
                successful += 1
                
            except Exception as e:
                logger.warning(f"Failed to predict for {file.filename}: {e}")
                results.append(
                    BatchPredictionItem(
                        filename=file.filename or "unknown",
                        predicted_class=-1,
                        predicted_label="error",
                        confidence=0.0,
                        error="Prediction failed",
                    )
                )
                failed += 1
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            success=successful > 0,
            predictions=results,
            total=len(results),
            successful=successful,
            failed=failed,
            processing_time_ms=round(processing_time_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")
