# ============================================================
# Security Utilities for All Projects
# ============================================================
"""
Security utilities for API key redaction and log sanitization.

This module provides functions to redact sensitive information from logs and error messages
to prevent API key exposure in logs.

Example:
    >>> from shared.security import redact_sensitive_data
    >>> error_msg = "Failed with api_key=sk-abc123 for user@domain.com"
    >>> redact_sensitive_data(error_msg)
    "Failed with api_key=REDACTED for user@domain.com"
"""

import logging
import re
from typing import Any


# ============================================================
# Sensitive Data Patterns
# ============================================================

SENSITIVE_PATTERNS = [
    # OpenAI API Keys
    (r'sk-[a-zA-Z0-9]{20,}', 'sk-REDACTED'),
    (r'Bearer sk-[a-zA-Z0-9]{20,}', 'Bearer sk-REDACTED'),
    (r'openai["\']?\s*[:=]\s*["\']sk-[a-zA-Z0-9]{20,}["\']', 'openai="REDACTED"'),

    # Anthropic API Keys
    (r'sk-ant-[a-zA-Z0-9]{20,}', 'sk-ant-REDACTED'),
    (r'ANTHROPIC_API_KEY["\']?\s*[:=]\s*["\'][^"\']+["\']', 'ANTHROPIC_API_KEY="REDACTED"'),

    # Generic API key patterns
    (r'api[_-]?key["\']?\s*[:=]\s*["\'][^"\']+["\']', 'api_key="REDACTED"'),
    (r'apikey["\']?\s*[:=]\s*["\'][^"\']+["\']', 'apikey="REDACTED"'),
    (r'API[_-]?KEY["\']?\s*[:=]\s*["\'][^"\']+["\']', 'API_KEY="REDACTED"'),

    # Cohere API Keys
    (r'Bearer [a-zA-Z0-9]{40,}', 'Bearer REDACTED'),
    (r'cohere-[-a-zA-Z0-9]{40,}', 'cohere-REDACTED'),

    # ZhipuAI/GLM API Keys
    (r'[\w-]{32,}\.glm', 'REDACTED.glm'),
    (r'["\']?token["\']?\s*[:=]\s*["\'][\w-]+["\']', 'token="REDACTED"'),

    # Email addresses (partially redacted for privacy)
    (r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', r'\1@***'),
    (r'"email"\s*[:=]\s*["\']([^"\']+)["\']', 'email="***@***"'),

    # IP addresses
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '***.***.***.***'),
    (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'x.x.x.x'),

    # Passwords in JSON
    (r'["\']?password["\']?\s*[:=]\s*["\'][^"\']+["\']', 'password="REDACTED"'),
    (r'["\']?passwd["\']?\s*[:=]\s*["\'][^"\']+["\']', 'passwd="REDACTED"'),
]


def redact_sensitive_data(text: str, additional_patterns: list[tuple[str, str]] = None) -> str:
    """
    Redact sensitive information from text for logging.

    This function protects against accidental logging of:
    - API keys
    - Auth tokens
    - Email addresses
    - IP addresses
    - Passwords
    - Secrets

    Args:
        text: Text to sanitize
        additional_patterns: Optional list of (regex, replacement) tuples

    Returns:
        Sanitized text with sensitive data redacted

    Example:
        >>> error = "API call failed with api_key=sk-abc123xyz and email@test.com"
        >>> redact_sensitive_data(error)
        "API call failed with api_key=REDACTED and ***@***"

        >>> # With custom pattern
        >>> custom = [(r'SECRET=["\'][^"\']+["\']', 'SECRET="REDACTED"')]
        >>> redact_sensitive_data("SECRET=abc123", additional_patterns=custom)
        "SECRET=REDACTED"
    """
    if not text or not isinstance(text, str):
        return text

    result = text

    # Apply standard patterns
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Apply additional custom patterns if provided
    if additional_patterns:
        for pattern, replacement in additional_patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def redact_dict(data: dict[str, Any], keys_to_redact: list[str] = None) -> dict[str, Any]:
    """
    Redact sensitive values in a dictionary.

    Args:
        data: Dictionary to sanitize
        keys_to_redact: List of keys to redact (default: all sensitive keys)

    Returns:
        Dictionary with sensitive values redacted

    Example:
        >>> data = {"api_key": "sk-abc123", "user": "john@example.com"}
        >>> redact_dict(data)
        {"api_key": "sk-REDACTED", "user": "john@example.com"}

        >>> # Redact specific keys
        >>> redact_dict(data, keys_to_redact=["api_key"])
        {"api_key": "sk-REDACTED", "user": "john@example.com"}
    """
    if not keys_to_redact:
        keys_to_redact = [
            "api_key", "apikey", "api_key", "APIKEY",
            "secret", "password", "passwd", "token",
            "auth_token", "access_token", "refresh_token",
            "session_token", "jwt", "bearer_token",
        ]

    result = {}
    for key, value in data.items():
        if any(sensitive in key.lower() for sensitive in keys_to_redact):
            if isinstance(value, str):
                result[key] = "REDACTED"
            else:
                result[key] = value
        else:
            result[key] = value

    return result


def redact_exception(e: Exception) -> str:
    """
    Redact sensitive information from exception before logging.

    Args:
        e: Exception to sanitize

    Returns:
        Sanitized error message

    Example:
        >>> try:
        ...     raise ValueError("Invalid api_key=sk-abc123")
        >>> except Exception as e:
        ...     print(redact_exception(e))
        "Invalid api_key=REDACTED"
    """
    error_str = str(e)
    return redact_sensitive_data(error_str)


# ============================================================
# Logging Filter
# ============================================================

class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that automatically redacts sensitive data.

    Install on a logger to automatically redact all log records.

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> logger.addFilter(SensitiveDataFilter())
        >>> logger.error("API key: sk-abc123")
        # Logs: API key: sk-REDACTED
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record to redact sensitive data."""
        record.msg = redact_sensitive_data(str(record.msg))
        if record.args:
            record.args = tuple(redact_sensitive_data(str(arg)) for arg in record.args)
        return True


def install_security_filter(logger: logging.Logger) -> None:
    """
    Install sensitive data filter on a logger.

    Args:
        logger: Logger to install filter on

    Example:
        >>> from shared.security import install_security_filter
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> install_security_filter(logger)
    """
    logger.addFilter(SensitiveDataFilter())


# ============================================================
# FastAPI Dependencies
# ============================================================

def create_security_middleware():
    """
    Create security middleware for FastAPI applications.

    Returns:
        List of middleware for FastAPI app

    Example:
        >>> from shared.security import create_security_middleware
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> for middleware in create_security_middleware():
        ...     app.add_middleware(middleware)
    """
    # This can be expanded to include additional security middleware
    # For now, return empty list - filter-based approach used instead
    return []


# ============================================================
# Export
# ============================================================

__all__ = [
    "redact_sensitive_data",
    "redact_dict",
    "redact_exception",
    "SensitiveDataFilter",
    "install_security_filter",
    "create_security_middleware",
]
