"""
Pydantic response models for API endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    """Single prediction result."""
    
    class_: int = Field(..., alias="class", description="Class index")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    
    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    """Response model for single image prediction."""
    
    success: bool = Field(..., description="Whether the prediction was successful")
    predicted_class: int = Field(..., description="Predicted class index")
    predicted_label: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    top_predictions: List[PredictionResult] = Field(
        default_factory=list, 
        description="Top-k predictions"
    )
    processing_time_ms: Optional[float] = Field(
        default=None, 
        description="Processing time in milliseconds"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BatchPredictionItem(BaseModel):
    """Single item in batch prediction response."""
    
    filename: str = Field(..., description="Original filename")
    predicted_class: int = Field(..., description="Predicted class index")
    predicted_label: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BatchPredictionResponse(BaseModel):
    """Response model for batch image prediction."""
    
    success: bool = Field(..., description="Whether the batch prediction was successful")
    predictions: List[BatchPredictionItem] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of images processed")
    successful: int = Field(default=0, description="Number of successful predictions")
    failed: int = Field(default=0, description="Number of failed predictions")
    processing_time_ms: Optional[float] = Field(
        default=None, 
        description="Total processing time in milliseconds"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    
    detail: str = Field(..., description="Error description")
    error_type: Optional[str] = Field(default=None, description="Type of error")
    status_code: int = Field(..., description="HTTP status code")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    environment: str = Field(..., description="Current environment")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_name: str = Field(..., description="Name of the model architecture")
    num_classes: int = Field(..., description="Number of output classes")
    device: str = Field(..., description="Device model is running on")
    checkpoint_path: str = Field(..., description="Path to model checkpoint")
    epoch: int = Field(..., description="Training epoch when checkpoint was saved")
    validation_accuracy: float = Field(..., description="Validation accuracy at checkpoint")
    input_size: tuple = Field(default=(224, 224), description="Expected input image size")


class MetricsResponse(BaseModel):
    """Prometheus metrics response."""
    
    metrics: str = Field(..., description="Prometheus-formatted metrics")
    content_type: str = Field(default="text/plain; version=0.0.4; charset=utf-8")
