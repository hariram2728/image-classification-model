"""
Pydantic request models for API endpoints.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
import base64
import io
from PIL import Image


class PredictionRequest(BaseModel):
    """Request model for single image prediction."""
    
    image_base64: str = Field(
        ..., 
        description="Base64 encoded image string",
        min_length=1,
        max_length=10 * 1024 * 1024  # 10MB limit for base64 string
    )
    top_k: int = Field(
        default=5, 
        ge=1, 
        le=100,
        description="Number of top predictions to return"
    )
    
    @field_validator('image_base64')
    @classmethod
    def validate_image_base64(cls, v: str) -> str:
        """Validate that the base64 string is a valid image."""
        try:
            # Decode base64
            image_data = base64.b64decode(v)
            
            # Check size limit (10MB)
            if len(image_data) > 10 * 1024 * 1024:
                raise ValueError("Image size exceeds 10MB limit")
            
            # Validate it's a valid image
            image = Image.open(io.BytesIO(image_data))
            image.verify()  # Verify it's a valid image
            
            # Re-open to check format (verify() closes the image)
            image = Image.open(io.BytesIO(image_data))
            
            # Check allowed formats
            allowed_formats = {'JPEG', 'PNG', 'WEBP', 'BMP', 'GIF'}
            if image.format not in allowed_formats:
                raise ValueError(f"Image format {image.format} not allowed. Allowed: {allowed_formats}")
            
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {str(e)}")


class BatchPredictionRequest(BaseModel):
    """Request model for batch image prediction."""
    
    images_base64: List[str] = Field(
        ..., 
        description="List of base64 encoded image strings",
        min_length=1,
        max_length=100  # Max 100 images per batch
    )
    top_k: int = Field(
        default=5, 
        ge=1, 
        le=100,
        description="Number of top predictions to return per image"
    )
    
    @field_validator('images_base64')
    @classmethod
    def validate_images_base64(cls, v: List[str]) -> List[str]:
        """Validate each base64 string in the list."""
        if len(v) > 100:
            raise ValueError("Maximum 100 images allowed per batch")
        
        for i, img_data in enumerate(v):
            try:
                image_bytes = base64.b64decode(img_data)
                
                # Check individual image size (10MB)
                if len(image_bytes) > 10 * 1024 * 1024:
                    raise ValueError(f"Image {i+1} exceeds 10MB limit")
                
                # Validate image
                image = Image.open(io.BytesIO(image_bytes))
                image.verify()
                
                image = Image.open(io.BytesIO(image_bytes))
                allowed_formats = {'JPEG', 'PNG', 'WEBP', 'BMP', 'GIF'}
                if image.format not in allowed_formats:
                    raise ValueError(f"Image {i+1}: format {image.format} not allowed")
                    
            except Exception as e:
                raise ValueError(f"Invalid image at index {i}: {str(e)}")
        
        return v
