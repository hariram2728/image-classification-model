"""
Model information endpoint for the API.
"""

from fastapi import APIRouter, HTTPException, Depends
from api.schemas.response import ModelInfoResponse
from api.core.model_manager import ModelManager

router = APIRouter()


def get_model_manager() -> ModelManager:
    """Dependency to get model manager instance."""
    from api.main import model_manager
    if model_manager is None or not model_manager.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for the server to initialize.",
        )
    return model_manager


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(
    model_manager: ModelManager = Depends(get_model_manager),
) -> ModelInfoResponse:
    """
    Get information about the loaded model.
    
    Returns details about the model architecture, training, and configuration.
    """
    info = model_manager.get_model_info()
    
    if not info:
        raise HTTPException(
            status_code=503,
            detail="Model information not available",
        )
    
    return ModelInfoResponse(
        model_name=info.get("model_name", "unknown"),
        num_classes=info.get("num_classes", 0),
        device=info.get("device", "cpu"),
        checkpoint_path=info.get("checkpoint_path", ""),
        epoch=info.get("epoch", -1),
        validation_accuracy=info.get("val_acc", 0.0),
        input_size=(224, 224),  # Default, could be stored in model_info
    )
