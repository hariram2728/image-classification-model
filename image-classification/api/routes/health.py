"""
Health check endpoints for the API.
"""

from fastapi import APIRouter, Depends
from api.schemas.response import HealthResponse
from api.core.model_manager import ModelManager

router = APIRouter()


def get_model_manager() -> ModelManager:
    """Dependency to get model manager instance."""
    from api.main import model_manager
    return model_manager


@router.get("/health", response_model=HealthResponse)
async def health_check(
    model_manager: ModelManager = Depends(get_model_manager),
) -> HealthResponse:
    """
    Basic health check endpoint.
    
    Returns the current status of the API service.
    """
    from api.core.config import settings
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=model_manager.is_loaded() if model_manager else False,
        environment=settings.environment,
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_check(
    model_manager: ModelManager = Depends(get_model_manager),
) -> HealthResponse:
    """
    Readiness check endpoint.
    
    Returns ready status only if the model is loaded and service is ready to handle requests.
    """
    from api.core.config import settings
    
    if not model_manager or not model_manager.is_loaded():
        return HealthResponse(
            status="not_ready",
            version="1.0.0",
            model_loaded=False,
            environment=settings.environment,
        )
    
    return HealthResponse(
        status="ready",
        version="1.0.0",
        model_loaded=True,
        environment=settings.environment,
    )
