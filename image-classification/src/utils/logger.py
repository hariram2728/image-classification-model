"""
Utility functions for structured logging.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from loguru import logger


def setup_logger(
    name: str = "image_classification",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "json",
) -> logging.Logger:
    """
    Setup structured logging for the application.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Format type ('json' or 'console')

    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()

    # Console handler with colored output
    if log_format == "console":
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=level,
            colorize=True,
        )
    else:
        # JSON format for production
        logger.add(
            sys.stderr,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            serialize=True,
        )

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            serialize=(log_format == "json"),
        )

    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


class LoggerContext:
    """Context manager for adding contextual information to logs."""

    def __init__(self, **kwargs):
        self.context = kwargs
        self.token = None

    def __enter__(self):
        self.token = logger.contextualize(**self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            logger.contextualize(reset=True)
        return False
