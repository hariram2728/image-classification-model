"""
FastAPI application for image classification inference.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer

from src.utils.logger import setup_logger, get_logger
from api.core.config import settings
from api.core.model_manager import ModelManager
from api.routes import predict, health, metrics, model_info
from api.middleware import RateLimitMiddleware, AuthMiddleware, LoggingMiddleware

logger = get_logger(__name__)

# Global model manager
model_manager: ModelManager = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan events."""
    global model_manager

    # Startup
    logger.info("Starting up Image Classification API...")
    setup_logger(level=settings.log_level)

    # Initialize model manager
    model_manager = ModelManager(
        model_path=settings.model_path,
        device=settings.device,
    )
    app.state.model_manager = model_manager

    logger.info(f"Model loaded successfully from {settings.model_path}")

    yield

    # Shutdown
    logger.info("Shutting down Image Classification API...")
    if model_manager:
        model_manager.unload_model()


# Create FastAPI application with security headers
app = FastAPI(
    title="Image Classification API",
    description="Production-ready image classification inference service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    # XSS protection
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # Referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Content Security Policy (adjust as needed)
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
    # Permissions Policy
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    # HSTS (only in production with HTTPS)
    if settings.environment == "production" and settings.enable_https_redirect:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response


# CORS middleware - SECURE CONFIGURATION
# Only allow explicit origins, never wildcards with credentials
if settings.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Correlation-ID"],
        expose_headers=["X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-Correlation-ID"],
        max_age=600,  # Cache preflight for 10 minutes
    )
    logger.info(f"CORS configured for origins: {settings.allowed_origins}")
else:
    logger.warning("No CORS origins configured - cross-origin requests will be blocked")


# Rate limiting middleware
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)

# Authentication middleware - NOW PROPERLY REGISTERED
app.add_middleware(AuthMiddleware, api_key=settings.api_key)

# Logging middleware
app.add_middleware(LoggingMiddleware)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Image Classification API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "environment": settings.environment,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler.
    
    SECURITY FIX: Does not expose stack traces or internal details in production.
    """
    logger.error(f"Unhandled exception: {type(exc).__name__}", exc_info=settings.debug)
    
    # In production, return generic error message
    if settings.environment == "production":
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error_type": type(exc).__name__,
            },
        )
    else:
        # In development, include more details for debugging
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc),
                "error_type": type(exc).__name__,
                "debug_info": repr(exc),
            },
        )


# Include routers
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(metrics.router, prefix="/api/v1", tags=["Metrics"])
app.include_router(model_info.router, prefix="/api/v1", tags=["Model Info"])


def main():
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=settings.api_workers if not settings.debug else 1,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
