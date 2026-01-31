# ============================================================
# Enterprise-RAG: Custom Exception Definitions
# ============================================================
"""
Custom exception hierarchy for the RAG system with error codes.

This module defines a comprehensive exception hierarchy for handling
errors across the RAG pipeline, from document processing through
retrieval, generation, and evaluation.

Each exception includes:
- A descriptive error message
- Optional original exception for debugging
- Error code for API responses
- HTTP status code mapping

Example:
    >>> try:
    ...     process_document(file_path)
    ... except DocumentProcessingError as e:
    ...     print(f"Error {e.error_code}: {e.message}")
    ...     print(f"Original: {e.__cause__}")
    >>> try:
    ...     retrieve_documents(query)
    ... except RetrievalError as e:
    ...     print(f"Retrieval failed: {e}")

Error Code Format: RAG_XXX_YYY
- XXX: Module code (DOC, RET, GEN, EVAL, CFG)
- YYY: Specific error number
"""

import logging
from enum import Enum
from typing import Any, Optional
from functools import wraps

from src.config import settings


# ============================================================
# Error Codes Enumeration
# ============================================================

class ErrorCode(str, Enum):
    """
    Standardized error codes for the RAG system.

    Format: RAG_<MODULE>_<NUMBER>
    Modules:
        - DOC: Document Processing
        - RET: Retrieval
        - GEN: Generation
        - EVAL: Evaluation
        - CFG: Configuration
        - API: API Layer
        - VAL: Validation
    """

    # Document Processing Errors
    DOC_UNSUPPORTED_FORMAT = "RAG_DOC_001"
    DOC_FILE_NOT_FOUND = "RAG_DOC_002"
    DOC_FILE_TOO_LARGE = "RAG_DOC_003"
    DOC_PARSE_ERROR = "RAG_DOC_004"
    DOC_CHUNK_ERROR = "RAG_DOC_005"
    DOC_EMPTY_CONTENT = "RAG_DOC_006"
    DOC_METADATA_ERROR = "RAG_DOC_007"

    # Retrieval Errors
    RET_NO_INDEX = "RAG_RET_001"
    RET_QUERY_EMPTY = "RAG_RET_002"
    RET_EMBEDDING_FAILED = "RAG_RET_003"
    RET_VECTOR_STORE_ERROR = "RAG_RET_004"
    RET_RERANK_FAILED = "RAG_RET_005"
    RET_NO_RESULTS = "RAG_RET_006"
    RET_BM25_ERROR = "RAG_RET_007"

    # Generation Errors
    GEN_LLM_NOT_CONFIGURED = "RAG_GEN_001"
    GEN_LLM_API_ERROR = "RAG_GEN_002"
    GEN_TIMEOUT = "RAG_GEN_003"
    GEN_PROMPT_ERROR = "RAG_GEN_004"
    GEN_CONTEXT_TOO_LONG = "RAG_GEN_005"
    GENERATION_FAILED = "RAG_GEN_006"

    # Evaluation Errors
    EVAL_NO_DATA = "RAG_EVAL_001"
    EVAL_METRIC_ERROR = "RAG_EVAL_002"
    EVAL_API_ERROR = "RAG_EVAL_003"
    EVAL_VALIDATION_ERROR = "RAG_EVAL_004"

    # Configuration Errors
    CFG_MISSING_KEY = "RAG_CFG_001"
    CFG_INVALID_VALUE = "RAG_CFG_002"
    CFG_MODEL_NOT_AVAILABLE = "RAG_CFG_003"
    CFG_PATH_NOT_FOUND = "RAG_CFG_004"

    # API Errors
    API_INVALID_REQUEST = "RAG_API_001"
    API_RATE_LIMITED = "RAG_API_002"
    API_NOT_AUTHORIZED = "RAG_API_003"
    API_SERVICE_UNAVAILABLE = "RAG_API_004"

    # Validation Errors
    VAL_INVALID_INPUT = "RAG_VAL_001"
    VAL_MISSING_FIELD = "RAG_VAL_002"
    VAL_INVALID_FORMAT = "RAG_VAL_003"


# ============================================================
# HTTP Status Code Mapping
# ============================================================

def get_http_status(error_code: ErrorCode) -> int:
    """
    Map error codes to appropriate HTTP status codes.

    Args:
        error_code: The error code to map

    Returns:
        HTTP status code (400-599)

    Example:
        >>> get_http_status(ErrorCode.DOC_FILE_NOT_FOUND)
        404
        >>> get_http_status(ErrorCode.API_RATE_LIMITED)
        429
    """
    status_map = {
        # 400 Bad Request
        ErrorCode.DOC_UNSUPPORTED_FORMAT: 400,
        ErrorCode.DOC_FILE_TOO_LARGE: 400,
        ErrorCode.DOC_EMPTY_CONTENT: 400,
        ErrorCode.RET_QUERY_EMPTY: 400,
        ErrorCode.GEN_CONTEXT_TOO_LONG: 400,
        ErrorCode.CFG_INVALID_VALUE: 400,
        ErrorCode.API_INVALID_REQUEST: 400,
        ErrorCode.VAL_INVALID_INPUT: 400,
        ErrorCode.VAL_MISSING_FIELD: 400,
        ErrorCode.VAL_INVALID_FORMAT: 400,

        # 401 Unauthorized
        ErrorCode.API_NOT_AUTHORIZED: 401,

        # 404 Not Found
        ErrorCode.DOC_FILE_NOT_FOUND: 404,
        ErrorCode.RET_NO_INDEX: 404,
        ErrorCode.CFG_PATH_NOT_FOUND: 404,

        # 422 Unprocessable Entity
        ErrorCode.DOC_PARSE_ERROR: 422,
        ErrorCode.DOC_CHUNK_ERROR: 422,
        ErrorCode.DOC_METADATA_ERROR: 422,
        ErrorCode.RET_EMBEDDING_FAILED: 422,
        ErrorCode.RET_RERANK_FAILED: 422,
        ErrorCode.GEN_PROMPT_ERROR: 422,

        # 429 Rate Limited
        ErrorCode.API_RATE_LIMITED: 429,

        # 500 Internal Server Error
        ErrorCode.RET_VECTOR_STORE_ERROR: 500,
        ErrorCode.RET_BM25_ERROR: 500,
        ErrorCode.GEN_LLM_NOT_CONFIGURED: 500,
        ErrorCode.GEN_LLM_API_ERROR: 500,
        ErrorCode.GEN_TIMEOUT: 500,
        ErrorCode.GENERATION_FAILED: 500,
        ErrorCode.EVAL_NO_DATA: 500,
        ErrorCode.EVAL_METRIC_ERROR: 500,
        ErrorCode.EVAL_API_ERROR: 500,
        ErrorCode.EVAL_VALIDATION_ERROR: 500,
        ErrorCode.CFG_MISSING_KEY: 500,
        ErrorCode.CFG_MODEL_NOT_AVAILABLE: 500,
        ErrorCode.API_SERVICE_UNAVAILABLE: 503,
    }

    return status_map.get(error_code, 500)


# ============================================================
# Base Exception Class
# ============================================================

class RAGException(Exception):
    """
    Base exception class for all RAG system errors.

    Provides common functionality for all custom exceptions including
    error codes, detailed messages, and original exception tracking.

    Attributes:
        message: Human-readable error description
        error_code: Standardized error code from ErrorCode enum
        details: Additional error context (optional)
        http_status: HTTP status code for API responses

    Example:
        >>> try:
        ...     raise RAGException("Something went wrong", ErrorCode.API_INVALID_REQUEST)
        ... except RAGException as e:
        ...     print(f"{e.error_code}: {e.message}")
        ...     print(f"HTTP {e.http_status}")
    """

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the base RAG exception.

        Args:
            message: Human-readable error description
            error_code: Standardized error code
            details: Additional context about the error
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.http_status = get_http_status(error_code)

        # Build full error message
        full_message = f"[{error_code.value}] {message}"
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            full_message += f" ({detail_str})"

        super().__init__(full_message)

        # Log the exception automatically
        self._log_exception()

    def _log_exception(self) -> None:
        """Log the exception using the configured logger."""
        logger = logging.getLogger(__name__)

        if self.http_status >= 500:
            # Server errors - log as ERROR
            logger.error(
                f"{self.error_code.value}: {self.message}",
                extra={
                    "error_code": self.error_code.value,
                    "details": self.details,
                    "http_status": self.http_status,
                },
                exc_info=self.__cause__ is not None,
            )
        elif self.http_status >= 400:
            # Client errors - log as WARNING
            logger.warning(
                f"{self.error_code.value}: {self.message}",
                extra={
                    "error_code": self.error_code.value,
                    "details": self.details,
                    "http_status": self.http_status,
                },
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert exception to dictionary for API responses.

        Returns:
            Dictionary with error information

        Example:
            >>> try:
            ...     raise DocumentProcessingError("Invalid format", "pdf")
            ... except RAGException as e:
            ...     error_dict = e.to_dict()
            ...     print(error_dict)
        """
        return {
            "error": True,
            "error_code": self.error_code.value,
            "message": self.message,
            "http_status": self.http_status,
            "details": self.details,
        }

    def __str__(self) -> str:
        """Return string representation of the exception."""
        return f"[{self.error_code.value}] {self.message}"

    def __repr__(self) -> str:
        """Return detailed representation of the exception."""
        return f"{self.__class__.__name__}(code={self.error_code.value}, message='{self.message}')"


# ============================================================
# Document Processing Exceptions
# ============================================================

class DocumentProcessingError(RAGException):
    """
    Exception raised for errors during document processing.

    Covers file reading, parsing, chunking, and metadata extraction.

    Example:
        >>> try:
        ...     process_document("invalid.xyz")
        ... except DocumentProcessingError as e:
        ...     print(f"Document error: {e}")
        ...     print(f"Suggested format: {e.details.get('supported_formats')}")
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_format: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize document processing error.

        Args:
            message: Error description
            file_path: Path to the problematic file
            file_format: Format of the problematic file
            details: Additional error context
        """
        error_details = details or {}
        if file_path:
            error_details["file_path"] = file_path
        if file_format:
            error_details["file_format"] = file_format

        super().__init__(
            message=message,
            error_code=ErrorCode.DOC_PARSE_ERROR,
            details=error_details,
        )


class UnsupportedFormatError(DocumentProcessingError):
    """Exception raised for unsupported document formats."""

    def __init__(
        self,
        file_format: str,
        supported_formats: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize unsupported format error.

        Args:
            file_format: The unsupported format
            supported_formats: List of supported formats
        """
        super().__init__(
            message=f"Unsupported document format: '{file_format}'",
            file_format=file_format,
            details={
                "supported_formats": supported_formats or settings.supported_formats_list,
            },
        )
        self.error_code = ErrorCode.DOC_UNSUPPORTED_FORMAT


class FileSizeExceededError(DocumentProcessingError):
    """Exception raised when file size exceeds limit."""

    def __init__(
        self,
        file_path: str,
        file_size: int,
        max_size: int,
    ) -> None:
        """
        Initialize file size exceeded error.

        Args:
            file_path: Path to the file
            file_size: Actual file size in bytes
            max_size: Maximum allowed size in bytes
        """
        super().__init__(
            message=f"File size ({file_size} bytes) exceeds maximum ({max_size} bytes)",
            file_path=file_path,
            details={
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "max_size_mb": round(max_size / (1024 * 1024), 2),
            },
        )
        self.error_code = ErrorCode.DOC_FILE_TOO_LARGE


class DocumentChunkError(DocumentProcessingError):
    """Exception raised during document chunking."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        """Initialize document chunking error."""
        super().__init__(
            message=message,
            file_path=file_path,
            details={"chunk_size": chunk_size} if chunk_size else None,
        )
        self.error_code = ErrorCode.DOC_CHUNK_ERROR


# ============================================================
# Retrieval Exceptions
# ============================================================

class RetrievalError(RAGException):
    """
    Exception raised for errors during document retrieval.

    Covers vector search, BM25, embeddings, and reranking failures.

    Example:
        >>> try:
        ...     retriever.retrieve(query)
        ... except RetrievalError as e:
        ...     print(f"Retrieval failed: {e}")
    """

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize retrieval error.

        Args:
            message: Error description
            query: The query that failed
            details: Additional error context
        """
        error_details = details or {}
        if query:
            error_details["query"] = query[:100] + "..." if len(query) > 100 else query

        super().__init__(
            message=message,
            error_code=ErrorCode.RET_VECTOR_STORE_ERROR,
            details=error_details,
        )


class EmbeddingError(RetrievalError):
    """Exception raised during embedding generation."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        text_length: Optional[int] = None,
    ) -> None:
        """Initialize embedding error."""
        super().__init__(
            message=message,
            details={
                "model": model,
                "text_length": text_length,
            },
        )
        self.error_code = ErrorCode.RET_EMBEDDING_FAILED


class RerankingError(RetrievalError):
    """Exception raised during reranking."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        num_results: Optional[int] = None,
    ) -> None:
        """Initialize reranking error."""
        super().__init__(
            message=message,
            details={
                "reranker_model": model,
                "num_results": num_results,
            },
        )
        self.error_code = ErrorCode.RET_RERANK_FAILED


class NoResultsFoundError(RetrievalError):
    """Exception raised when no results are found."""

    def __init__(
        self,
        query: str,
        top_k: int,
    ) -> None:
        """Initialize no results error."""
        super().__init__(
            message=f"No documents found matching query",
            query=query,
            details={
                "top_k": top_k,
                "suggestion": "Try adjusting your query or indexing more documents",
            },
        )
        self.error_code = ErrorCode.RET_NO_RESULTS


# ============================================================
# Generation Exceptions
# ============================================================

class GenerationError(RAGException):
    """
    Exception raised during response generation.

    Covers LLM API errors, timeouts, and prompt construction.

    Example:
        >>> try:
        ...     rag_engine.generate_answer(query, context)
        ... except GenerationError as e:
        ...     print(f"Generation failed: {e}")
    """

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize generation error."""
        error_details = details or {}
        if model:
            error_details["model"] = model

        super().__init__(
            message=message,
            error_code=ErrorCode.GENERATION_FAILED,
            details=error_details,
        )


class LLMNotConfiguredError(GenerationError):
    """Exception raised when LLM is not properly configured."""

    def __init__(self, missing_config: str) -> None:
        """Initialize LLM configuration error."""
        super().__init__(
            message=f"LLM not configured: {missing_config}",
            details={"missing_config": missing_config},
        )
        self.error_code = ErrorCode.GEN_LLM_NOT_CONFIGURED


class LLMTimeoutError(GenerationError):
    """Exception raised when LLM request times out."""

    def __init__(
        self,
        model: str,
        timeout: float,
    ) -> None:
        """Initialize timeout error."""
        super().__init__(
            message=f"LLM request timed out after {timeout} seconds",
            model=model,
            details={"timeout_seconds": timeout},
        )
        self.error_code = ErrorCode.GEN_TIMEOUT


# ============================================================
# Evaluation Exceptions
# ============================================================

class EvaluationError(RAGException):
    """
    Exception raised during RAG evaluation.

    Covers metric calculation, data validation, and evaluation API errors.

    Example:
        >>> try:
        ...     evaluator.evaluate(dataset)
        ... except EvaluationError as e:
        ...     print(f"Evaluation failed: {e}")
    """

    def __init__(
        self,
        message: str,
        metric: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize evaluation error."""
        error_details = details or {}
        if metric:
            error_details["metric"] = metric

        super().__init__(
            message=message,
            error_code=ErrorCode.EVAL_METRIC_ERROR,
            details=error_details,
        )


class EvaluationDataError(EvaluationError):
    """Exception raised when evaluation data is invalid."""

    def __init__(
        self,
        message: str,
        dataset_size: Optional[int] = None,
    ) -> None:
        """Initialize evaluation data error."""
        super().__init__(
            message=message,
            details={"dataset_size": dataset_size} if dataset_size else None,
        )
        self.error_code = ErrorCode.EVAL_NO_DATA


# ============================================================
# Configuration Exceptions
# ============================================================

class ConfigurationError(RAGException):
    """
    Exception raised for configuration-related errors.

    Covers missing keys, invalid values, and model availability issues.

    Example:
        >>> try:
        ...     settings = get_settings()
        ...     if not settings.OPENAI_API_KEY:
        ...         raise ConfigurationError("Missing API key", "OPENAI_API_KEY")
        ... except ConfigurationError as e:
        ...     print(f"Configuration error: {e}")
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize configuration error."""
        error_details = details or {}
        if config_key:
            error_details["config_key"] = config_key

        super().__init__(
            message=message,
            error_code=ErrorCode.CFG_MISSING_KEY,
            details=error_details,
        )


class InvalidConfigValueError(ConfigurationError):
    """Exception raised for invalid configuration values."""

    def __init__(
        self,
        config_key: str,
        value: Any,
        expected_type: Optional[str] = None,
    ) -> None:
        """Initialize invalid config value error."""
        super().__init__(
            message=f"Invalid value for '{config_key}': {value}",
            config_key=config_key,
            details={
                "provided_value": str(value),
                "expected_type": expected_type,
            },
        )
        self.error_code = ErrorCode.CFG_INVALID_VALUE


class ModelNotAvailableError(ConfigurationError):
    """Exception raised when a model is not available."""

    def __init__(
        self,
        model_name: str,
        reason: Optional[str] = None,
    ) -> None:
        """Initialize model not available error."""
        super().__init__(
            message=f"Model '{model_name}' is not available: {reason or 'Unknown reason'}",
            details={"model": model_name, "reason": reason},
        )
        self.error_code = ErrorCode.CFG_MODEL_NOT_AVAILABLE


# ============================================================
# Decorator for Exception Handling
# ============================================================

def handle_exceptions(
    default_message: str = "An error occurred",
    reraise: bool = True,
) -> callable:
    """
    Decorator to handle and log exceptions in functions.

    Args:
        default_message: Default message if no specific error message
        reraise: Whether to reraise the exception after logging

    Example:
        >>> @handle_exceptions("Failed to process document")
        ... def process_document(path: str):
        ...     # Processing logic
        ...     pass
    """

    def decorator(func: callable) -> callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except RAGException:
                # RAG exceptions are already logged
                if reraise:
                    raise
                return None
            except Exception as e:
                # Unexpected exceptions
                logger = logging.getLogger(func.__module__)
                logger.error(
                    f"{default_message} in {func.__name__}: {str(e)}",
                    exc_info=True,
                    extra={"function": func.__name__, "args": str(args)[:200]},
                )
                if reraise:
                    raise
                return None

        return wrapper

    return decorator


# ============================================================
# Exception Utilities
# ============================================================

def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        True if the error is potentially retryable

    Example:
        >>> try:
        ...     api_call()
        ... except Exception as e:
        ...     if is_retryable_error(e):
        ...         retry()
    """
    if isinstance(error, RAGException):
        # Retry client errors (4xx) and some server errors
        return error.http_status in (429, 500, 502, 503, 504)
    return False


def format_error_for_api(error: Exception) -> dict[str, Any]:
    """
    Format any exception for API response.

    Args:
        error: The exception to format

    Returns:
        Dictionary suitable for JSON API response

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     return format_error_for_api(e)
    """
    if isinstance(error, RAGException):
        return error.to_dict()

    # Handle unexpected exceptions
    logger = logging.getLogger(__name__)
    logger.error(f"Unexpected error: {str(error)}", exc_info=True)

    return {
        "error": True,
        "error_code": "RAG_API_500",
        "message": "An unexpected error occurred",
        "http_status": 500,
        "details": {"type": type(error).__name__},
    }


# Export all exceptions
__all__ = [
    # Error codes
    "ErrorCode",
    "get_http_status",
    # Base exception
    "RAGException",
    # Document processing
    "DocumentProcessingError",
    "UnsupportedFormatError",
    "FileSizeExceededError",
    "DocumentChunkError",
    # Retrieval
    "RetrievalError",
    "EmbeddingError",
    "RerankingError",
    "NoResultsFoundError",
    # Generation
    "GenerationError",
    "LLMNotConfiguredError",
    "LLMTimeoutError",
    # Evaluation
    "EvaluationError",
    "EvaluationDataError",
    # Configuration
    "ConfigurationError",
    "InvalidConfigValueError",
    "ModelNotAvailableError",
    # Utilities
    "handle_exceptions",
    "is_retryable_error",
    "format_error_for_api",
]
