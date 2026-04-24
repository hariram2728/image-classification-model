"""
API configuration settings.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_workers: int = int(os.getenv("API_WORKERS", "4"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")

    # Model Settings
    model_path: str = os.getenv("MODEL_PATH", "./models/best_model.pth")
    model_name: str = os.getenv("MODEL_NAME", "image_classifier")
    device: str = "cuda" if os.getenv("DEVICE", "cuda") == "cuda" else "cpu"

    # Authentication - CRITICAL: Must be set in production
    api_key: Optional[str] = os.getenv("API_KEY")
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")

    # Rate Limiting
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

    # CORS - SECURITY: Explicit origins only in production
    allowed_origins: List[str] = []

    # Security
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    allowed_image_formats: List[str] = ["JPEG", "PNG", "WEBP", "BMP", "GIF"]
    enable_https_redirect: bool = os.getenv("ENABLE_HTTPS_REDIRECT", "true").lower() == "true"

    # Monitoring
    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"

    @field_validator('jwt_secret_key')
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        """Ensure JWT secret is set and sufficiently long."""
        if not v:
            raise ValueError("JWT_SECRET_KEY must be set in production")
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters long")
        return v

    @field_validator('allowed_origins')
    @classmethod
    def validate_cors_origins(cls, v: List[str]) -> List[str]:
        """Validate CORS origins - no wildcards with credentials."""
        if "*" in v:
            raise ValueError("Wildcard '*' origin not allowed when credentials are enabled")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
