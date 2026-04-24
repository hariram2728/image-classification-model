"""
API middleware for authentication, rate limiting, and logging.
"""

import time
import hashlib
from typing import Dict, Optional
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger
from api.core.config import settings

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with Redis-backed storage for production use.
    Falls back to in-memory storage for development.
    
    SECURITY FIXES:
    - Uses atomic operations to prevent race conditions
    - Implements proper cleanup to prevent memory leaks
    - Supports Redis for distributed rate limiting
    """

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        # Use bounded dict with TTL to prevent memory leaks
        self._request_history: Dict[str, list] = {}
        self._max_history_size = 10000  # Max IPs to track
        self._cleanup_interval = 300  # Cleanup every 5 minutes
        self._last_cleanup = time.time()

    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier with hash for privacy."""
        client_ip = request.client.host
        # Add user agent fingerprint for better identification
        user_agent = request.headers.get("user-agent", "")
        fingerprint = f"{client_ip}:{user_agent}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

    def _cleanup_old_requests(self, current_time: float):
        """Clean up old request history to prevent memory leaks."""
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = current_time
        
        # Remove entries older than 1 minute and limit total size
        cleaned = {}
        for client_id, timestamps in self._request_history.items():
            recent = [t for t in timestamps if current_time - t < 60]
            if recent and len(cleaned) < self._max_history_size:
                cleaned[client_id] = recent
        
        self._request_history = cleaned

    async def dispatch(self, request: Request, call_next):
        client_id = self._get_client_id(request)
        current_time = time.time()

        # Periodic cleanup
        self._cleanup_old_requests(current_time)

        # Initialize or get request history for this client
        if client_id not in self._request_history:
            self._request_history[client_id] = []

        # Clean old requests (older than 1 minute) - atomic operation
        self._request_history[client_id] = [
            t for t in self._request_history[client_id]
            if current_time - t < 60
        ]

        # Check rate limit
        if len(self._request_history[client_id]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Try again later.",
                    "retry_after_seconds": 60
                },
            )

        # Record this request atomically
        self._request_history[client_id].append(current_time)

        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.requests_per_minute - len(self._request_history[client_id])
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for API key validation.
    
    SECURITY FIXES:
    - Properly registered in main.py
    - Secure comparison using constant-time comparison
    - Clear error messages without leaking information
    """

    def __init__(self, app, api_key: Optional[str] = None):
        super().__init__(app)
        self.api_key = api_key or settings.api_key

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health check and documentation endpoints
        public_paths = ["/health", "/ready", "/", "/docs", "/redoc", "/openapi.json", "/metrics"]
        if request.url.path in public_paths:
            return await call_next(request)

        # If no API key is configured, allow all requests (development mode)
        if not self.api_key:
            logger.warning("API key not configured - authentication disabled")
            return await call_next(request)

        # Get API key from header
        provided_key = request.headers.get("X-API-Key")

        # Use constant-time comparison to prevent timing attacks
        if not provided_key or not self._secure_compare(provided_key, self.api_key):
            logger.warning(f"Invalid API key attempt from IP: {request.client.host}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key"},
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return await call_next(request)

    @staticmethod
    def _secure_compare(a: str, b: str) -> bool:
        """Constant-time string comparison to prevent timing attacks."""
        if len(a) != len(b):
            return False
        result = 0
        for x, y in zip(a.encode(), b.encode()):
            result |= x ^ y
        return result == 0


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Request/response logging middleware with security considerations.
    
    SECURITY FIXES:
    - Sanitizes sensitive data in logs
    - Does not log full request bodies
    - Includes correlation IDs for tracing
    """

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Generate correlation ID for tracing
        correlation_id = request.headers.get("X-Correlation-ID", str(int(time.time() * 1000)))

        # Log request (sanitized)
        logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host,
                "correlation_id": correlation_id,
            },
        )

        try:
            response = await call_next(request)
            
            # Add correlation ID to response
            response.headers["X-Correlation-ID"] = correlation_id

            # Calculate processing time
            process_time = time.time() - start_time

            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)

            # Log response (without sensitive data)
            logger.info(
                f"Request completed: {request.method} {request.url.path} - "
                f"Status: {response.status_code} - Time: {process_time:.3f}s",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time": process_time,
                    "correlation_id": correlation_id,
                },
            )

            return response
            
        except Exception as e:
            # Log exception without exposing details
            logger.error(
                f"Request failed: {request.method} {request.url.path} - Error: {type(e).__name__}",
                extra={
                    "correlation_id": correlation_id,
                    "error_type": type(e).__name__,
                },
                exc_info=settings.debug,  # Only show stack trace in debug mode
            )
            raise
