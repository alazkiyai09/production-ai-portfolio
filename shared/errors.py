# ============================================================
# Error Handling for All Projects
# ============================================================
"""
Standardized error handling module for FastAPI applications.

This module provides comprehensive error handling functionality including:
- Custom exception classes with error codes
- FastAPI exception handlers for consistent error responses
- Pydantic models for API error responses
- Error logging utilities
- Error tracking integration helpers

Example:
    >>> from shared.errors import (
    ...     AuthenticationError,
    ...     ValidationError,
    ...     ErrorResponse,
    ...     register_error_handlers,
    ...     log_error,
    ... )
    >>> from fastapi import FastAPI
    >>>
    >>> app = FastAPI()
    >>> register_error_handlers(app)
    >>>
    >>> @app.get("/api/resource")
    >>> async def get_resource():
    ...     if not valid:
    ...         raise ValidationError(
    ...             field_name="id",
    ...             message="Invalid ID format"
    ...         )
"""

import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Optional, Type, Union, Callable
from functools import wraps
from enum import Enum

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# ============================================================
# Error Code Enum
# ============================================================

class ErrorCode(str, Enum):
    """Standard error codes for API responses."""

    # Authentication & Authorization
    AUTHENTICATION_FAILED = "AUTH_001"
    INVALID_CREDENTIALS = "AUTH_002"
    TOKEN_EXPIRED = "AUTH_003"
    TOKEN_INVALID = "AUTH_004"
    AUTHORIZATION_FAILED = "AUTH_005"
    INSUFFICIENT_PERMISSIONS = "AUTH_006"
    ACCOUNT_DISABLED = "AUTH_007"
    SESSION_EXPIRED = "AUTH_008"

    # Validation
    VALIDATION_ERROR = "VAL_001"
    INVALID_INPUT = "VAL_002"
    MISSING_REQUIRED_FIELD = "VAL_003"
    INVALID_FORMAT = "VAL_004"
    CONSTRAINT_VIOLATION = "VAL_005"
    DUPLICATE_RESOURCE = "VAL_006"

    # Rate Limiting
    RATE_LIMIT_EXCEEDED = "RATE_001"
    TOO_MANY_REQUESTS = "RATE_002"
    TEMPORARILY_BLOCKED = "RATE_003"

    # Database
    DATABASE_ERROR = "DB_001"
    CONNECTION_FAILED = "DB_002"
    QUERY_FAILED = "DB_003"
    TRANSACTION_FAILED = "DB_004"
    RECORD_NOT_FOUND = "DB_005"
    RECORD_CONFLICT = "DB_006"

    # External API
    EXTERNAL_API_ERROR = "API_001"
    EXTERNAL_SERVICE_UNAVAILABLE = "API_002"
    EXTERNAL_TIMEOUT = "API_003"
    EXTERNAL_RATE_LIMIT = "API_004"

    # Configuration
    CONFIGURATION_ERROR = "CFG_001"
    MISSING_ENVIRONMENT = "CFG_002"
    INVALID_CONFIG = "CFG_003"

    # Server
    INTERNAL_ERROR = "SRV_001"
    SERVICE_UNAVAILABLE = "SRV_002"
    MAINTENANCE_MODE = "SRV_003"

    # Not Found
    NOT_FOUND = "NF_001"
    RESOURCE_NOT_FOUND = "NF_002"
    ENDPOINT_NOT_FOUND = "NF_003"


# ============================================================
# HTTP Status Code Mapping
# ============================================================

ERROR_STATUS_MAPPING: dict[ErrorCode, int] = {
    # Authentication - 401
    ErrorCode.AUTHENTICATION_FAILED: status.HTTP_401_UNAUTHORIZED,
    ErrorCode.INVALID_CREDENTIALS: status.HTTP_401_UNAUTHORIZED,
    ErrorCode.TOKEN_EXPIRED: status.HTTP_401_UNAUTHORIZED,
    ErrorCode.TOKEN_INVALID: status.HTTP_401_UNAUTHORIZED,
    ErrorCode.ACCOUNT_DISABLED: status.HTTP_401_UNAUTHORIZED,
    ErrorCode.SESSION_EXPIRED: status.HTTP_401_UNAUTHORIZED,

    # Authorization - 403
    ErrorCode.AUTHORIZATION_FAILED: status.HTTP_403_FORBIDDEN,
    ErrorCode.INSUFFICIENT_PERMISSIONS: status.HTTP_403_FORBIDDEN,

    # Validation - 400
    ErrorCode.VALIDATION_ERROR: status.HTTP_400_BAD_REQUEST,
    ErrorCode.INVALID_INPUT: status.HTTP_400_BAD_REQUEST,
    ErrorCode.MISSING_REQUIRED_FIELD: status.HTTP_400_BAD_REQUEST,
    ErrorCode.INVALID_FORMAT: status.HTTP_400_BAD_REQUEST,
    ErrorCode.CONSTRAINT_VIOLATION: status.HTTP_400_BAD_REQUEST,
    ErrorCode.DUPLICATE_RESOURCE: status.HTTP_409_CONFLICT,

    # Rate Limiting - 429
    ErrorCode.RATE_LIMIT_EXCEEDED: status.HTTP_429_TOO_MANY_REQUESTS,
    ErrorCode.TOO_MANY_REQUESTS: status.HTTP_429_TOO_MANY_REQUESTS,
    ErrorCode.TEMPORARILY_BLOCKED: status.HTTP_429_TOO_MANY_REQUESTS,

    # Database - 500/404/409
    ErrorCode.DATABASE_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ErrorCode.CONNECTION_FAILED: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.QUERY_FAILED: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ErrorCode.TRANSACTION_FAILED: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ErrorCode.RECORD_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.RECORD_CONFLICT: status.HTTP_409_CONFLICT,

    # External API - 502/503/504
    ErrorCode.EXTERNAL_API_ERROR: status.HTTP_502_BAD_GATEWAY,
    ErrorCode.EXTERNAL_SERVICE_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.EXTERNAL_TIMEOUT: status.HTTP_504_GATEWAY_TIMEOUT,
    ErrorCode.EXTERNAL_RATE_LIMIT: status.HTTP_429_TOO_MANY_REQUESTS,

    # Configuration - 500
    ErrorCode.CONFIGURATION_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ErrorCode.MISSING_ENVIRONMENT: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ErrorCode.INVALID_CONFIG: status.HTTP_500_INTERNAL_SERVER_ERROR,

    # Server - 500/503
    ErrorCode.INTERNAL_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ErrorCode.SERVICE_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.MAINTENANCE_MODE: status.HTTP_503_SERVICE_UNAVAILABLE,

    # Not Found - 404
    ErrorCode.NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.RESOURCE_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.ENDPOINT_NOT_FOUND: status.HTTP_404_NOT_FOUND,
}


def get_status_for_error(code: ErrorCode) -> int:
    """
    Get the HTTP status code for a given error code.

    Args:
        code: Error code enum

    Returns:
        HTTP status code (default: 500)

    Example:
        >>> get_status_for_error(ErrorCode.INVALID_CREDENTIALS)
        401
    """
    return ERROR_STATUS_MAPPING.get(code, status.HTTP_500_INTERNAL_SERVER_ERROR)


# ============================================================
# Base Exception Class
# ============================================================

class BaseAppError(Exception):
    """
    Base exception class for all application errors.

    Provides consistent error handling with:
    - Error codes for programmatic handling
    - HTTP status code mapping
    - Detailed error messages
    - Additional context data
    - Request ID tracking

    Attributes:
        error_code: Standard error code from ErrorCode enum
        message: Human-readable error message
        status_code: HTTP status code
        details: Additional error context
        request_id: Unique identifier for the request

    Example:
        >>> raise BaseAppError(
        ...     error_code=ErrorCode.VALIDATION_ERROR,
        ...     message="Invalid input data",
        ...     status_code=400
        ... )
    """

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the base error exception.

        Args:
            error_code: Standard error code from ErrorCode enum
            message: Human-readable error message
            status_code: HTTP status code (auto-mapped from error_code if not provided)
            details: Additional error context
            request_id: Unique identifier for the request
        """
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.request_id = request_id or str(uuid.uuid4())

        if status_code is None:
            status_code = get_status_for_error(error_code)

        self.status_code = status_code
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert error to dictionary for API response.

        Returns:
            Dictionary representation of the error

        Example:
            >>> error = BaseAppError(ErrorCode.VALIDATION_ERROR, "Invalid input")
            >>> error.to_dict()
            {
                'error_code': 'VAL_001',
                'message': 'Invalid input',
                'status_code': 400,
                'request_id': '...',
                'details': {}
            }
        """
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "status_code": self.status_code,
            "request_id": self.request_id,
            "details": self.details,
        }

    def __str__(self) -> str:
        """Return string representation of the error."""
        return f"[{self.error_code.value}] {self.message}"

    def __repr__(self) -> str:
        """Return detailed representation of the error."""
        return (
            f"{self.__class__.__name__}("
            f"error_code={self.error_code.value}, "
            f"message='{self.message}', "
            f"status_code={self.status_code}"
            f")"
        )


# ============================================================
# Custom Exception Classes
# ============================================================

class AuthenticationError(BaseAppError):
    """
    Exception raised for authentication failures.

    Use when credentials are invalid, token is expired, etc.

    Attributes:
        username: Optional username that failed authentication
        reason: Specific reason for authentication failure

    Example:
        >>> raise AuthenticationError(
        ...     username="john_doe",
        ...     reason="Invalid password"
        ... )
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        username: Optional[str] = None,
        reason: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.AUTHENTICATION_FAILED,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Initialize authentication error.

        Args:
            message: Human-readable error message
            username: Username that failed authentication
            reason: Specific reason for failure
            error_code: Specific authentication error code
            request_id: Unique request identifier
        """
        details: dict[str, Any] = {}
        if username:
            details["username"] = username
        if reason:
            details["reason"] = reason

        super().__init__(
            error_code=error_code,
            message=message,
            status_code=get_status_for_error(error_code),
            details=details,
            request_id=request_id,
        )


class AuthorizationError(BaseAppError):
    """
    Exception raised for authorization failures.

    Use when user is authenticated but lacks permission for an action.

    Attributes:
        required_permission: Permission that was required
        user_role: Role of the user
        resource: Resource being accessed

    Example:
        >>> raise AuthorizationError(
        ...     required_permission="admin",
        ...     user_role="user",
        ...     resource="users"
        ... )
    """

    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_permission: Optional[str] = None,
        user_role: Optional[str] = None,
        resource: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.INSUFFICIENT_PERMISSIONS,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Initialize authorization error.

        Args:
            message: Human-readable error message
            required_permission: Permission that was required
            user_role: Role of the user
            resource: Resource being accessed
            error_code: Specific authorization error code
            request_id: Unique request identifier
        """
        details: dict[str, Any] = {}
        if required_permission:
            details["required_permission"] = required_permission
        if user_role:
            details["user_role"] = user_role
        if resource:
            details["resource"] = resource

        super().__init__(
            error_code=error_code,
            message=message,
            status_code=get_status_for_error(error_code),
            details=details,
            request_id=request_id,
        )


class ValidationError(BaseAppError):
    """
    Exception raised for validation errors.

    Use when input data fails validation rules.

    Attributes:
        field_name: Name of the field that failed validation
        field_value: Value that failed validation
        constraint: Validation constraint that was violated

    Example:
        >>> raise ValidationError(
        ...     field_name="email",
        ...     field_value="invalid-email",
        ...     constraint="valid email format"
        ... )
    """

    def __init__(
        self,
        message: str = "Validation failed",
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        constraint: Optional[str] = None,
        errors: Optional[list[dict[str, Any]]] = None,
        error_code: ErrorCode = ErrorCode.VALIDATION_ERROR,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Initialize validation error.

        Args:
            message: Human-readable error message
            field_name: Name of the field that failed validation
            field_value: Value that failed validation
            constraint: Validation constraint that was violated
            errors: List of validation errors for multiple fields
            error_code: Specific validation error code
            request_id: Unique request identifier
        """
        details: dict[str, Any] = {}
        if field_name:
            details["field"] = field_name
        if field_value is not None:
            details["value"] = str(field_value)[:100]  # Limit length
        if constraint:
            details["constraint"] = constraint
        if errors:
            details["errors"] = errors

        super().__init__(
            error_code=error_code,
            message=message,
            status_code=get_status_for_error(error_code),
            details=details,
            request_id=request_id,
        )


class RateLimitError(BaseAppError):
    """
    Exception raised when rate limit is exceeded.

    Attributes:
        retry_after: Seconds until retry is allowed
        limit: Rate limit that was exceeded
        current_usage: Current usage count

    Example:
        >>> raise RateLimitError(
        ...     retry_after=60,
        ...     limit="10/minute"
        ... )
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        limit: Optional[str] = None,
        current_usage: Optional[int] = None,
        error_code: ErrorCode = ErrorCode.RATE_LIMIT_EXCEEDED,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Initialize rate limit error.

        Args:
            message: Human-readable error message
            retry_after: Seconds until retry is allowed
            limit: Rate limit that was exceeded
            current_usage: Current usage count
            error_code: Specific rate limit error code
            request_id: Unique request identifier
        """
        details: dict[str, Any] = {}
        if retry_after is not None:
            details["retry_after"] = retry_after
        if limit:
            details["limit"] = limit
        if current_usage is not None:
            details["current_usage"] = current_usage

        super().__init__(
            error_code=error_code,
            message=message,
            status_code=get_status_for_error(error_code),
            details=details,
            request_id=request_id,
        )


class DatabaseError(BaseAppError):
    """
    Exception raised for database-related errors.

    Attributes:
        query: Query that failed (sanitized)
        table: Table being accessed
        operation: Operation being performed

    Example:
        >>> raise DatabaseError(
        ...     operation="SELECT",
        ...     table="users",
        ...     reason="Connection timeout"
        ... )
    """

    def __init__(
        self,
        message: str = "Database operation failed",
        operation: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        reason: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.DATABASE_ERROR,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Initialize database error.

        Args:
            message: Human-readable error message
            operation: Operation being performed
            table: Table being accessed
            query: Query that failed (will be sanitized)
            reason: Specific reason for failure
            error_code: Specific database error code
            request_id: Unique request identifier
        """
        from shared.security import redact_sensitive_data

        details: dict[str, Any] = {}
        if operation:
            details["operation"] = operation
        if table:
            details["table"] = table
        if query:
            details["query"] = redact_sensitive_data(query)[:200]
        if reason:
            details["reason"] = reason

        super().__init__(
            error_code=error_code,
            message=message,
            status_code=get_status_for_error(error_code),
            details=details,
            request_id=request_id,
        )


class ExternalAPIError(BaseAppError):
    """
    Exception raised for external API failures.

    Attributes:
        service: Name of the external service
        status_code: HTTP status from external service
        response_body: Response body from external service

    Example:
        >>> raise ExternalAPIError(
        ...     service="OpenAI",
        ...     status_code=429,
        ...     reason="Rate limit exceeded"
        ... )
    """

    def __init__(
        self,
        message: str = "External API call failed",
        service: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        reason: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.EXTERNAL_API_ERROR,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Initialize external API error.

        Args:
            message: Human-readable error message
            service: Name of the external service
            status_code: HTTP status from external service
            response_body: Response body from external service
            reason: Specific reason for failure
            error_code: Specific external API error code
            request_id: Unique request identifier
        """
        from shared.security import redact_sensitive_data

        details: dict[str, Any] = {}
        if service:
            details["service"] = service
        if status_code is not None:
            details["external_status"] = status_code
        if response_body:
            details["response"] = redact_sensitive_data(response_body)[:200]
        if reason:
            details["reason"] = reason

        # Map to appropriate error code based on status
        if status_code == 429:
            error_code = ErrorCode.EXTERNAL_RATE_LIMIT
        elif status_code == 503:
            error_code = ErrorCode.EXTERNAL_SERVICE_UNAVAILABLE
        elif status_code is None or status_code >= 504:
            error_code = ErrorCode.EXTERNAL_TIMEOUT

        super().__init__(
            error_code=error_code,
            message=message,
            status_code=get_status_for_error(error_code),
            details=details,
            request_id=request_id,
        )


class ConfigurationError(BaseAppError):
    """
    Exception raised for configuration-related errors.

    Attributes:
        config_key: Configuration key that is invalid
        expected_type: Expected type of configuration value
        provided_value: Value that was provided

    Example:
        >>> raise ConfigurationError(
        ...     config_key="DATABASE_URL",
        ...     reason="Missing environment variable"
        ... )
    """

    def __init__(
        self,
        message: str = "Configuration error",
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        provided_value: Optional[Any] = None,
        reason: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.CONFIGURATION_ERROR,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Initialize configuration error.

        Args:
            message: Human-readable error message
            config_key: Configuration key that is invalid
            expected_type: Expected type of configuration value
            provided_value: Value that was provided
            reason: Specific reason for failure
            error_code: Specific configuration error code
            request_id: Unique request identifier
        """
        from shared.security import redact_sensitive_data

        details: dict[str, Any] = {}
        if config_key:
            details["config_key"] = config_key
        if expected_type:
            details["expected_type"] = expected_type
        if provided_value is not None:
            details["provided"] = redact_sensitive_data(str(provided_value))[:50]
        if reason:
            details["reason"] = reason

        super().__init__(
            error_code=error_code,
            message=message,
            status_code=get_status_for_error(error_code),
            details=details,
            request_id=request_id,
        )


class NotFoundError(BaseAppError):
    """
    Exception raised when a resource is not found.

    Attributes:
        resource_type: Type of resource that was not found
        resource_id: ID of the resource

    Example:
        >>> raise NotFoundError(
        ...     resource_type="User",
        ...     resource_id="123"
        ... )
    """

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.RESOURCE_NOT_FOUND,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Initialize not found error.

        Args:
            message: Human-readable error message
            resource_type: Type of resource that was not found
            resource_id: ID of the resource
            error_code: Specific not found error code
            request_id: Unique request identifier
        """
        details: dict[str, Any] = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id

        super().__init__(
            error_code=error_code,
            message=message,
            status_code=get_status_for_error(error_code),
            details=details,
            request_id=request_id,
        )


# ============================================================
# Pydantic Error Response Models
# ============================================================

class ErrorDetail(BaseModel):
    """Single error detail item."""

    field: Optional[str] = Field(None, description="Field name that caused the error")
    message: str = Field(..., description="Error message for this field")
    constraint: Optional[str] = Field(None, description="Validation constraint that was violated")


class ValidationErrorDetail(BaseModel):
    """Detailed validation error information."""

    field: str = Field(..., description="Field name that caused the error")
    value: Optional[str] = Field(None, description="Value that failed validation")
    message: str = Field(..., description="Error message")
    constraint: Optional[str] = Field(None, description="Validation constraint that was violated")
    location: Optional[str] = Field(None, description="Location of the field (e.g., body, query, path)")


class ErrorResponse(BaseModel):
    """
    Standard error response model for all API errors.

    This model ensures consistent error response structure across all endpoints.

    Attributes:
        success: Always false for error responses
        error_code: Standard error code for programmatic handling
        message: Human-readable error message
        status_code: HTTP status code
        request_id: Unique identifier for the request
        details: Additional error context
        errors: List of validation errors (for validation failures)
        timestamp: ISO timestamp of when the error occurred
        path: Request path that caused the error

    Example:
        >>> response = ErrorResponse(
        ...     error_code=ErrorCode.VALIDATION_ERROR,
        ...     message="Invalid input",
        ...     status_code=400,
        ...     request_id="abc-123",
        ...     path="/api/users"
        ... )
    """

    success: bool = Field(False, description="Always false for error responses")
    error_code: str = Field(..., description="Standard error code for programmatic handling")
    message: str = Field(..., description="Human-readable error message")
    status_code: int = Field(..., description="HTTP status code")
    request_id: str = Field(..., description="Unique identifier for the request")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional error context")
    errors: Optional[list[ValidationErrorDetail]] = Field(None, description="List of validation errors")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO timestamp of when the error occurred"
    )
    path: Optional[str] = Field(None, description="Request path that caused the error")

    @validator("success")
    def success_must_be_false(cls, v: bool) -> bool:
        """Ensure success is always false for error responses."""
        return False

    class Config:
        """Pydantic config."""

        from_attributes = True
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "VAL_001",
                "message": "Validation failed",
                "status_code": 400,
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "details": {"field": "email"},
                "timestamp": "2024-01-01T00:00:00",
                "path": "/api/users"
            }
        }


# ============================================================
# Error Logging Utilities
# ============================================================

class ErrorContext(BaseModel):
    """
    Context information for error logging.

    Attributes:
        request_id: Unique identifier for the request
        user_id: ID of the user if authenticated
        path: Request path
        method: HTTP method
        ip_address: Client IP address (sanitized)
        user_agent: Client user agent
        extra: Additional context data
    """

    request_id: str
    user_id: Optional[str] = None
    path: Optional[str] = None
    method: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


def log_error(
    error: Exception,
    level: str = "ERROR",
    context: Optional[ErrorContext] = None,
    include_traceback: bool = True,
) -> None:
    """
    Log an error with standardized formatting.

    Args:
        error: Exception to log
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        context: Additional error context
        include_traceback: Whether to include stack trace

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_error(e, context=ErrorContext(request_id="123"))
    """
    from shared.security import redact_exception

    log_level = getattr(logging, level.upper(), logging.ERROR)
    log_func = logger.log

    # Build log message
    error_type = type(error).__name__
    error_message = redact_exception(error)

    log_parts = [f"[{error_type}]", error_message]

    # Add error code if it's a BaseAppError
    if isinstance(error, BaseAppError):
        log_parts.insert(1, f"[{error.error_code.value}]")
        if context is None:
            context = ErrorContext(request_id=error.request_id)

    # Add context information
    if context:
        context_parts = []
        if context.user_id:
            context_parts.append(f"user={context.user_id}")
        if context.path:
            context_parts.append(f"path={context.path}")
        if context.method:
            context_parts.append(f"method={context.method}")
        if context.request_id:
            context_parts.append(f"request_id={context.request_id}")

        if context_parts:
            log_parts.append(f"({', '.join(context_parts)})")

    # Add extra context
    if context and context.extra:
        log_parts.append(f"extra={context.extra}")

    log_message = " ".join(log_parts)

    # Add traceback if requested
    exc_info = include_traceback
    if include_traceback and isinstance(error, BaseAppError):
        # For our custom errors, only add traceback if it's not a known error
        if error.error_code in (
            ErrorCode.VALIDATION_ERROR,
            ErrorCode.AUTHENTICATION_FAILED,
            ErrorCode.INSUFFICIENT_PERMISSIONS,
            ErrorCode.NOT_FOUND,
        ):
            exc_info = False

    log_func(log_level, log_message, exc_info=exc_info)


def log_validation_error(
    errors: list[dict[str, Any]],
    context: Optional[ErrorContext] = None,
) -> None:
    """
    Log validation errors with formatted output.

    Args:
        errors: List of validation errors
        context: Error context

    Example:
        >>> errors = [{"field": "email", "message": "Invalid format"}]
        >>> log_validation_error(errors, context=ErrorContext(request_id="123"))
    """
    error_count = len(errors)
    context_str = f"request_id={context.request_id}" if context else "no_context"

    logger.warning(
        f"Validation failed with {error_count} error(s) ({context_str}): {errors}"
    )


def extract_error_context(request: Request) -> ErrorContext:
    """
    Extract error context from a FastAPI request.

    Args:
        request: FastAPI request object

    Returns:
        ErrorContext with request information

    Example:
        >>> @app.exception_handler(Exception)
        >>> async def handle_exception(request: Request, exc: Exception):
        ...     context = extract_error_context(request)
        ...     log_error(exc, context=context)
    """
    from shared.security import redact_sensitive_data

    # Get request ID from state or headers
    request_id = getattr(request.state, "request_id", None)
    if not request_id:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Get user info from state
    user_id = getattr(request.state, "user_id", None)

    # Sanitize IP address
    ip_address = request.client.host if request.client else None
    if ip_address:
        ip_address = redact_sensitive_data(ip_address)

    return ErrorContext(
        request_id=request_id,
        user_id=user_id,
        path=request.url.path,
        method=request.method,
        ip_address=ip_address,
        user_agent=request.headers.get("user-agent"),
    )


# ============================================================
# Exception Handlers
# ============================================================

def create_error_response(
    error: Union[BaseAppError, Exception],
    request: Optional[Request] = None,
    status_code_override: Optional[int] = None,
) -> JSONResponse:
    """
    Create a standardized JSON error response.

    Args:
        error: Exception to convert to response
        request: Optional FastAPI request object
        status_code_override: Override the status code

    Returns:
        JSONResponse with standardized error format

    Example:
        >>> try:
        ...     operation()
        ... except ValidationError as e:
        ...     return create_error_response(e, request)
    """
    from shared.security import redact_exception

    # Handle BaseAppError
    if isinstance(error, BaseAppError):
        error_dict = error.to_dict()
        status_code = status_code_override or error.status_code

        response_data = ErrorResponse(
            error_code=error.error_code.value,
            message=error.message,
            status_code=status_code,
            request_id=error.request_id,
            details=error.details,
            path=request.url.path if request else None,
        )

    # Handle HTTPException
    elif isinstance(error, HTTPException):
        status_code = error.status_code
        response_data = ErrorResponse(
            error_code="HTTP_EXCEPTION",
            message=str(error.detail),
            status_code=status_code,
            request_id=str(uuid.uuid4()),
            path=request.url.path if request else None,
        )

    # Handle RequestValidationError
    elif isinstance(error, RequestValidationError):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        validation_errors = []

        for err in error.errors():
            field = ".".join(str(loc) for loc in err["loc"] if loc != "body")
            validation_errors.append(
                ValidationErrorDetail(
                    field=field,
                    message=err["msg"],
                    constraint=err.get("type"),
                    location=err["loc"][0] if err["loc"] else None,
                )
            )

        response_data = ErrorResponse(
            error_code=ErrorCode.VALIDATION_ERROR.value,
            message="Validation failed",
            status_code=status_code,
            request_id=str(uuid.uuid4()),
            errors=validation_errors,
            path=request.url.path if request else None,
        )

    # Generic exception
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        sanitized_message = redact_exception(error)

        response_data = ErrorResponse(
            error_code=ErrorCode.INTERNAL_ERROR.value,
            message="An internal error occurred",
            status_code=status_code,
            request_id=str(uuid.uuid4()),
            details={"error": sanitized_message} if logger.isEnabledFor(logging.DEBUG) else {},
            path=request.url.path if request else None,
        )

    return JSONResponse(
        status_code=status_code,
        content=response_data.model_dump(exclude_none=True),
    )


async def base_app_error_handler(request: Request, exc: BaseAppError) -> JSONResponse:
    """
    FastAPI exception handler for BaseAppError and subclasses.

    Args:
        request: FastAPI request object
        exc: BaseAppError exception

    Returns:
        JSONResponse with standardized error format
    """
    context = extract_error_context(request)

    # Update request_id in exception
    if exc.request_id != context.request_id:
        exc.request_id = context.request_id

    # Log the error
    log_error(exc, context=context)

    # Create response
    return create_error_response(exc, request)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    FastAPI exception handler for HTTPException.

    Args:
        request: FastAPI request object
        exc: HTTPException

    Returns:
        JSONResponse with standardized error format
    """
    context = extract_error_context(request)
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail} ({context.request_id})")

    return create_error_response(exc, request)


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    FastAPI exception handler for RequestValidationError.

    Args:
        request: FastAPI request object
        exc: RequestValidationError

    Returns:
        JSONResponse with standardized error format
    """
    context = extract_error_context(request)

    # Format validation errors
    validation_errors = []
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err["loc"] if loc != "body")
        validation_errors.append({
            "field": field,
            "message": err["msg"],
            "type": err.get("type"),
        })

    log_validation_error(validation_errors, context)

    return create_error_response(exc, request)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler for unhandled exceptions.

    Args:
        request: FastAPI request object
        exc: Unhandled exception

    Returns:
        JSONResponse with standardized error format
    """
    context = extract_error_context(request)

    # Log the error with traceback
    log_error(exc, context=context, include_traceback=True)

    # Create generic error response
    return create_error_response(exc, request)


# ============================================================
# Error Handler Registration
# ============================================================

def register_error_handlers(
    app: FastAPI,
    custom_handlers: Optional[dict[Type[Exception], Callable]] = None,
    log_unhandled: bool = True,
) -> None:
    """
    Register all standard error handlers with a FastAPI application.

    Args:
        app: FastAPI application instance
        custom_handlers: Optional dict of custom exception handlers
        log_unhandled: Whether to log unhandled exceptions

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> register_error_handlers(app)
    """
    # Register base app error handler (catches all custom exceptions)
    app.add_exception_handler(BaseAppError, base_app_error_handler)

    # Register HTTP exception handler
    app.add_exception_handler(HTTPException, http_exception_handler)

    # Register validation exception handler
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # Register generic exception handler (must be last)
    if log_unhandled:
        app.add_exception_handler(Exception, generic_exception_handler)

    # Register custom handlers if provided
    if custom_handlers:
        for exc_type, handler in custom_handlers.items():
            app.add_exception_handler(exc_type, handler)

    logger.info("Error handlers registered successfully")


# ============================================================
# Decorator for Error Handling
# ============================================================

def handle_errors(
    error_map: Optional[dict[Type[Exception], Type[BaseAppError]]] = None,
    default_error: Type[BaseAppError] = ConfigurationError,
    reraise: bool = False,
):
    """
    Decorator to convert exceptions to standardized application errors.

    Args:
        error_map: Mapping of exception types to application error types
        default_error: Default error type for unmapped exceptions
        reraise: Whether to re-raise the converted error

    Example:
        >>> @handle_errors({
        ...     ValueError: ValidationError,
        ...     KeyError: NotFoundError,
        ... })
        >>> async def my_endpoint():
        ...     data = get_data()  # May raise ValueError
        ...     return {"data": data}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except BaseAppError:
                # Re-raise our custom errors
                raise
            except Exception as e:
                # Map to custom error or use default
                error_type = default_error
                for exc_type, mapped_error in (error_map or {}).items():
                    if isinstance(e, exc_type):
                        error_type = mapped_error
                        break

                # Create and raise/log the error
                error = error_type(
                    message=str(e),
                    reason=type(e).__name__
                )

                if reraise:
                    log_error(error, context=None)
                    raise error
                else:
                    log_error(error, context=None)
                    return create_error_response(error)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except BaseAppError:
                raise
            except Exception as e:
                error_type = default_error
                for exc_type, mapped_error in (error_map or {}).items():
                    if isinstance(e, exc_type):
                        error_type = mapped_error
                        break

                error = error_type(
                    message=str(e),
                    reason=type(e).__name__
                )

                if reraise:
                    log_error(error, context=None)
                    raise error
                else:
                    log_error(error, context=None)
                    return create_error_response(error)

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ============================================================
# Error Tracking Integration Helpers
# ============================================================

class ErrorTracker:
    """
    Base class for error tracking integration.

    Subclass this to integrate with error tracking services like
    Sentry, Rollbar, DataDog, etc.

    Attributes:
        enabled: Whether tracking is enabled
        environment: Application environment (dev, staging, prod)
        service_name: Name of the service
    """

    def __init__(
        self,
        enabled: bool = True,
        environment: str = "development",
        service_name: str = "api",
    ) -> None:
        """
        Initialize error tracker.

        Args:
            enabled: Whether tracking is enabled
            environment: Application environment
            service_name: Name of the service
        """
        self.enabled = enabled
        self.environment = environment
        self.service_name = service_name

    def track_exception(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        level: str = "error",
    ) -> str:
        """
        Track an exception with the error tracking service.

        Args:
            error: Exception to track
            context: Error context
            level: Error level (debug, info, warning, error, fatal)

        Returns:
            Event ID or tracking identifier

        Example:
            >>> tracker = ErrorTracker()
            >>> try:
            ...     risky_operation()
            ... except Exception as e:
            ...     event_id = tracker.track_exception(e)
        """
        if not self.enabled:
            return "disabled"

        # Base implementation - override in subclasses
        logger.error(
            f"Error tracking not configured. Error: {type(error).__name__}: {error}"
        )
        return "not_configured"

    def track_message(
        self,
        message: str,
        level: str = "info",
        context: Optional[ErrorContext] = None,
    ) -> str:
        """
        Track a message with the error tracking service.

        Args:
            message: Message to track
            level: Message level
            context: Error context

        Returns:
            Event ID or tracking identifier
        """
        if not self.enabled:
            return "disabled"

        logger.log(
            getattr(logging, level.upper(), logging.INFO),
            message
        )
        return "logged"

    def add_breadcrumb(
        self,
        message: str,
        category: str = "custom",
        level: str = "info",
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Add a breadcrumb for error context.

        Args:
            message: Breadcrumb message
            category: Breadcrumb category
            level: Breadcrumb level
            data: Additional data

        Example:
            >>> tracker.addBreadcrumb(
            ...     message="User clicked button",
            ...     category="user",
            ...     data={"button": "submit"}
            ... )
        """
        if not self.enabled:
            return

        logger.debug(f"Breadcrumb [{category}]: {message}")

    def set_user_context(self, user_id: str, **kwargs: Any) -> None:
        """
        Set user context for error tracking.

        Args:
            user_id: User ID
            **kwargs: Additional user context

        Example:
            >>> tracker.set_user_context(
            ...     user_id="user123",
            ...     email="user@example.com",
            ...     username="john_doe"
            ... )
        """
        if not self.enabled:
            return

        logger.debug(f"User context set: {user_id}")

    def clear_context(self) -> None:
        """Clear all context (user, tags, etc.)."""
        if not self.enabled:
            return

        logger.debug("Error tracking context cleared")


# Global error tracker instance
_error_tracker: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """
    Get the global error tracker instance.

    Returns:
        Global ErrorTracker instance

    Example:
        >>> tracker = get_error_tracker()
        >>> tracker.track_exception(exc)
    """
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker(enabled=False)
    return _error_tracker


def configure_error_tracking(
    tracker: Optional[ErrorTracker] = None,
    enabled: bool = True,
    environment: str = "production",
    service_name: str = "api",
) -> None:
    """
    Configure error tracking for the application.

    Args:
        tracker: Custom ErrorTracker instance
        enabled: Whether tracking should be enabled
        environment: Application environment
        service_name: Name of the service

    Example:
        >>> configure_error_tracking(
        ...     enabled=True,
        ...     environment="production",
        ...     service_name="my-api"
        ... )
    """
    global _error_tracker

    if tracker is not None:
        _error_tracker = tracker
    else:
        _error_tracker = ErrorTracker(
            enabled=enabled,
            environment=environment,
            service_name=service_name,
        )

    logger.info(f"Error tracking configured: enabled={enabled}, env={environment}")


def track_error(
    error: Exception,
    context: Optional[ErrorContext] = None,
    level: str = "error",
) -> str:
    """
    Convenience function to track an error.

    Args:
        error: Exception to track
        context: Error context
        level: Error level

    Returns:
        Event ID or tracking identifier

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     track_error(e)
    """
    return get_error_tracker().track_exception(error, context, level)


# ============================================================
# Middleware for Request ID Generation
# ============================================================

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add unique request IDs to all requests.

    This middleware:
    - Generates a unique ID for each request
    - Adds it to request state for error handlers
    - Includes it in response headers

    Example:
        >>> app = FastAPI()
        >>> app.add_middleware(RequestIDMiddleware)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and add ID.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response with X-Request-ID header
        """
        # Check for existing request ID in header
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store in request state
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


# ============================================================
# Utility Functions
# ============================================================

def raise_http_error(
    status_code: int,
    message: str,
    error_code: Optional[ErrorCode] = None,
    details: Optional[dict[str, Any]] = None,
) -> None:
    """
    Raise an HTTPException with standardized format.

    Args:
        status_code: HTTP status code
        message: Error message
        error_code: Optional error code
        details: Optional error details

    Raises:
        HTTPException

    Example:
        >>> if not user:
        ...     raise_http_error(404, "User not found", ErrorCode.NOT_FOUND)
    """
    if error_code is None:
        # Map status code to error code
        if status_code == 401:
            error_code = ErrorCode.AUTHENTICATION_FAILED
        elif status_code == 403:
            error_code = ErrorCode.INSUFFICIENT_PERMISSIONS
        elif status_code == 404:
            error_code = ErrorCode.NOT_FOUND
        elif status_code == 429:
            error_code = ErrorCode.RATE_LIMIT_EXCEEDED
        elif status_code == 500:
            error_code = ErrorCode.INTERNAL_ERROR
        else:
            error_code = ErrorCode.VALIDATION_ERROR

    raise HTTPException(
        status_code=status_code,
        detail={
            "error_code": error_code.value,
            "message": message,
            "details": details or {},
        },
    )


def format_validation_errors(
    errors: list[dict[str, Any]],
) -> list[ValidationErrorDetail]:
    """
    Format validation errors into ValidationErrorDetail objects.

    Args:
        errors: List of error dictionaries

    Returns:
        List of ValidationErrorDetail objects

    Example:
        >>> errors = [
        ...     {"field": "email", "message": "Invalid format"}
        ... ]
        >>> formatted = format_validation_errors(errors)
    """
    formatted = []
    for error in errors:
        formatted.append(
            ValidationErrorDetail(
                field=error.get("field", "unknown"),
                message=error.get("message", "Validation failed"),
                constraint=error.get("constraint"),
                location=error.get("location"),
            )
        )
    return formatted


# ============================================================
# Export
# ============================================================

__all__ = [
    # Enums
    "ErrorCode",
    # Base Exception
    "BaseAppError",
    # Custom Exceptions
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError",
    "RateLimitError",
    "DatabaseError",
    "ExternalAPIError",
    "ConfigurationError",
    "NotFoundError",
    # Pydantic Models
    "ErrorDetail",
    "ValidationErrorDetail",
    "ErrorResponse",
    "ErrorContext",
    # Error Tracking
    "ErrorTracker",
    "get_error_tracker",
    "configure_error_tracking",
    "track_error",
    # Logging
    "log_error",
    "log_validation_error",
    "extract_error_context",
    # Exception Handlers
    "create_error_response",
    "base_app_error_handler",
    "http_exception_handler",
    "validation_exception_handler",
    "generic_exception_handler",
    "register_error_handlers",
    # Middleware
    "RequestIDMiddleware",
    # Decorators
    "handle_errors",
    # Utilities
    "raise_http_error",
    "format_validation_errors",
    "get_status_for_error",
]
