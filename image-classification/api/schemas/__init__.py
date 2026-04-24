"""
Pydantic schemas for API request/response models.
"""

from api.schemas.request import PredictionRequest, BatchPredictionRequest
from api.schemas.response import (
    PredictionResponse,
    BatchPredictionResponse,
    PredictionResult,
    BatchPredictionItem,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    MetricsResponse,
)

__all__ = [
    # Requests
    "PredictionRequest",
    "BatchPredictionRequest",
    # Responses
    "PredictionResponse",
    "BatchPredictionResponse",
    "PredictionResult",
    "BatchPredictionItem",
    "ErrorResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "MetricsResponse",
]