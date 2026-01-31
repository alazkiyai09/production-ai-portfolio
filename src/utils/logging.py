"""
Logging configuration for AgenticFlow.

This module provides centralized logging setup using Loguru for
structured logging throughout the application.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.config import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_json: bool = False,
) -> None:
    """
    Configure application logging with Loguru.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional, logs to stdout if None)
        log_json: Whether to use JSON format for logs

    Example:
        >>> setup_logging(log_level="INFO", log_file="app.log")
        >>> logger.info("Application started")
    """
    # Remove default handler
    logger.remove()

    # Get settings
    level = log_level or settings.log_level
    file_path = log_file or settings.log_file
    use_json = log_json or settings.log_json

    # Console handler
    format_string = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    if use_json:
        # JSON format for structured logging
        logger.add(
            sys.stdout,
            format="{message}",
            level=level,
            serialize=True,
            colorize=False,
        )
    else:
        # Pretty console format
        logger.add(
            sys.stdout,
            format=format_string,
            level=level,
            colorize=True,
        )

    # File handler (if specified)
    if file_path:
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )

    # Set logger options
    logger.configure(
        extra={
            "service": "agenticflow",
            "version": "0.1.0",
        }
    )


def get_logger(name: str):
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> from src.utils.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Message")
    """
    return logger.bind(name=name)


# Initial setup
setup_logging()
