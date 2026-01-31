# ============================================================
# Enterprise-RAG: Logging Configuration
# ============================================================
"""
Structured logging configuration for the RAG system.

Provides comprehensive logging with:
- Colored console output for development
- File logging with rotation for production
- JSON format for structured logging
- Request ID tracking for distributed tracing
- Module-specific log levels
- Performance timing utilities

Example:
    >>> from src.logging_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Document processed", extra={"doc_id": "123", "pages": 5})
    >>> logger.error("Retrieval failed", extra={"query": "test", "error": "timeout"})
"""

import logging
import logging.handlers
import sys
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import jsonlog

from src.config import settings
# Import security filter from shared utilities
try:
    from shared.security import SensitiveDataFilter
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False


# ============================================================
# Context Variables for Request Tracking
# ============================================================

REQUEST_ID_CTX: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
SESSION_ID_CTX: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


# ============================================================
# Custom Formatters
# ============================================================

class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for development.

    Provides color-coded log levels and formatted output for readability.

    Colors:
        DEBUG: Blue
        INFO: Green
        WARNING: Yellow
        ERROR: Red
        CRITICAL: Red + Bold
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[1;31m", # Bold Red
    }
    RESET = "\033[0m"

    def __init__(self) -> None:
        """Initialize colored formatter with format string."""
        super().__init__()
        self.format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        )
        self.datefmt = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.

        Args:
            record: The log record to format

        Returns:
            Formatted and colored log message
        """
        # Get color for level
        level_color = self.COLORS.get(record.levelname, "")

        # Format the message
        formatter = logging.Formatter(
            f"{level_color}{{}}{self.RESET}".format(self.format_string),
            datefmt=self.datefmt,
        )
        result = formatter.format(record)

        # Add exception info if present
        if record.exc_info:
            result += "\n" + formatter.formatException(record.exc_info)

        return result


class StructuredFormatter(logging.Formatter):
    """
    Structured formatter that outputs logs as JSON for production.

    Provides machine-readable log format suitable for:
    - Log aggregation systems (ELK, Splunk)
    - Cloud logging (AWS CloudWatch, GCP Cloud Logging)
    - Distributed tracing systems

    Output format:
    {
        "timestamp": "2024-01-15T10:30:45.123Z",
        "level": "INFO",
        "logger": "src.core.rag_engine",
        "message": "Document processed successfully",
        "module": "rag_engine",
        "function": "process_document",
        "line": 123,
        "request_id": "abc-123",
        "session_id": "session-xyz",
        "extra": {...}
    }
    """

    def __init__(self) -> None:
        """Initialize structured formatter."""
        super().__init__()
        self.required_fields = [
            "timestamp",
            "level",
            "logger",
            "message",
            "module",
            "function",
            "line",
        ]

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: The log record to format

        Returns:
            JSON-formatted log message
        """
        # Create base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request/session tracking if available
        request_id = REQUEST_ID_CTX.get()
        if request_id:
            log_entry["request_id"] = request_id

        session_id = SESSION_ID_CTX.get()
        if session_id:
            log_entry["session_id"] = session_id

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created", "msecs",
                "relativeCreated", "thread", "threadName", "processName",
                "process", "getMessage", "exc_info", "exc_text", "stack_info",
            }:
                # Serialize value
                try:
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class DetailedFormatter(logging.Formatter):
    """
    Detailed text formatter for file logging.

    Provides comprehensive information in human-readable text format.
    Suitable for log files that need to be readable by humans.
    """

    def __init__(self) -> None:
        """Initialize detailed formatter."""
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | "
            "RequestID=%(request_id)s | %(message)s"
        )
        super().__init__(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with request ID."""
        # Add request_id to record
        request_id = REQUEST_ID_CTX.get()
        record.request_id = request_id or "N/A"

        # Format the message
        result = super().format(record)

        # Add extra fields as key=value pairs
        extra_fields = []
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created", "msecs",
                "relativeCreated", "thread", "threadName", "processName",
                "process", "getMessage", "exc_info", "exc_text", "stack_info",
                "message", "asctime", "request_id",
            }:
                extra_fields.append(f"{key}={value}")

        if extra_fields:
            result += " | " + " ".join(extra_fields)

        # Add exception if present
        if record.exc_info:
            result += "\n" + self.formatException(record.exc_info)

        return result


# ============================================================
# Logging Configuration
# ============================================================

def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    log_format: str = "text",
) -> None:
    """
    Configure logging for the application.

    Sets up:
    - Root logger with specified level
    - Console handler with colored output
    - File handler with rotation
    - Module-specific log levels

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_format: Format type (text, json)

    Example:
        >>> setup_logging(log_level="DEBUG", log_format="json")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Get settings
    level = log_level or settings.LOG_LEVEL
    log_path = log_file or settings.log_file_path
    format_type = log_format or settings.LOG_FORMAT

    # Ensure log directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # ============================================================
    # Security Filter (API Key Redaction)
    # ============================================================
    if SECURITY_AVAILABLE:
        security_filter = SensitiveDataFilter()
        root_logger.addFilter(security_filter)

    # ============================================================
    # Console Handler
    # ============================================================
    if settings.CONSOLE_LOGGING:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        # Use colored formatter for text format
        if format_type == "json":
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(ColoredFormatter())

        root_logger.addHandler(console_handler)

    # ============================================================
    # File Handler with Rotation
    # ============================================================
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_path),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # File gets all logs

    # Use appropriate formatter
    if format_type == "json":
        file_handler.setFormatter(StructuredFormatter())
    else:
        file_handler.setFormatter(DetailedFormatter())

    root_logger.addHandler(file_handler)

    # ============================================================
    # Module-Specific Log Levels
    # ============================================================
    loggers_config = {
        # Our application
        "src": logging.DEBUG,
        # External libraries (reduce noise)
        "uvicorn": logging.INFO,
        "uvicorn.access": logging.WARNING,
        "uvicorn.error": logging.ERROR,
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "sentence_transformers": logging.WARNING,
        "chromadb": logging.WARNING,
        "qdrant_client": logging.WARNING,
        "faiss": logging.WARNING,
        "langchain": logging.WARNING,
        "ragas": logging.INFO,
        # Suppress verbose loggers
        "PIL": logging.WARNING,
        "matplotlib": logging.WARNING,
        "numba": logging.WARNING,
    }

    for logger_name, logger_level in loggers_config.items():
        module_logger = logging.getLogger(logger_name)
        module_logger.setLevel(logger_level)
        # Prevent propagation to avoid duplicate logs
        module_logger.propagate = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    This is the preferred way to get loggers in the application.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured logger instance

    Example:
        >>> from src.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing document", extra={"doc_id": "123"})
    """
    return logging.getLogger(name)


# ============================================================
# Logging Context Managers
# ============================================================

class RequestContext:
    """
    Context manager for request-scoped logging context.

    Automatically sets and clears request_id and session_id.

    Example:
        >>> with RequestContext(request_id="abc-123", session_id="session-xyz"):
        ...     logger.info("Processing request")
        ...     # All logs in this block will have request_id and session_id
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Initialize request context.

        Args:
            request_id: Unique request identifier
            session_id: Session identifier
        """
        self.request_id = request_id
        self.session_id = session_id
        self.token = None
        self.session_token = None

    def __enter__(self) -> "RequestContext":
        """Set context variables on enter."""
        if self.request_id:
            self.token = REQUEST_ID_CTX.set(self.request_id)
        if self.session_id:
            self.session_token = SESSION_ID_CTX.set(self.session_id)
        return self

    def __exit__(self, *args: Any) -> None:
        """Clear context variables on exit."""
        if self.token:
            REQUEST_ID_CTX.reset(self.token)
        if self.session_token:
            SESSION_ID_CTX.reset(self.session_token)


# ============================================================
# Logging Decorators
# ============================================================

def log_execution(
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    include_args: bool = False,
    include_result: bool = False,
) -> callable:
    """
    Decorator to log function execution with timing.

    Args:
        logger: Logger instance (uses module logger if None)
        level: Log level for the message
        include_args: Whether to include function arguments
        include_result: Whether to include return value

    Example:
        >>> @log_execution(include_args=True)
        ... def process_document(doc_id: str):
        ...     # Function logic
        ...     return {"status": "success"}
    """

    def decorator(func: callable) -> callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get logger
            func_logger = logger or logging.getLogger(func.__module__)

            # Build log message
            msg = f"Executing {func.__name__}"

            # Add args if requested
            extra: dict[str, Any] = {"function": func.__name__}
            if include_args:
                extra["args"] = str(args)[:200]
                extra["kwargs"] = str(kwargs)[:200]

            func_logger.log(level, msg, extra=extra)

            # Execute and time
            import time
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                execution_time = time.time() - start_time
                extra["execution_time_ms"] = round(execution_time * 1000, 2)

                if include_result:
                    extra["result"] = str(result)[:200]

                func_logger.log(
                    level,
                    f"Completed {func.__name__} in {execution_time:.2f}s",
                    extra=extra,
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                extra["execution_time_ms"] = round(execution_time * 1000, 2)
                extra["error"] = str(e)

                func_logger.error(
                    f"Failed {func.__name__} after {execution_time:.2f}s: {str(e)}",
                    extra=extra,
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


def log_async_execution(
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
) -> callable:
    """
    Decorator to log async function execution with timing.

    Args:
        logger: Logger instance (uses module logger if None)
        level: Log level for the message

    Example:
        >>> @log_async_execution()
        ... async def process_document_async(doc_id: str):
        ...     # Async logic
        ...     return {"status": "success"}
    """

    import asyncio

    def decorator(func: callable) -> callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_logger = logger or logging.getLogger(func.__module__)

            msg = f"Executing async {func.__name__}"
            extra: dict[str, Any] = {"function": func.__name__, "async": True}

            func_logger.log(level, msg, extra=extra)

            import time
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)

                execution_time = time.time() - start_time
                extra["execution_time_ms"] = round(execution_time * 1000, 2)

                func_logger.log(
                    level,
                    f"Completed async {func.__name__} in {execution_time:.2f}s",
                    extra=extra,
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                extra["execution_time_ms"] = round(execution_time * 1000, 2)
                extra["error"] = str(e)

                func_logger.error(
                    f"Failed async {func.__name__} after {execution_time:.2f}s: {str(e)}",
                    extra=extra,
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


# ============================================================
# Utility Functions
# ============================================================

def set_request_context(request_id: str, session_id: Optional[str] = None) -> None:
    """
    Set request-scoped logging context.

    Use this in FastAPI middleware or request handlers.

    Args:
        request_id: Unique request identifier
        session_id: Optional session identifier

    Example:
        >>> @app.middleware("http")
        ... async def add_request_id(request: Request, call_next):
        ...     request_id = generate_id()
        ...     set_request_context(request_id)
        ...     response = await call_next(request)
        ...     response.headers["X-Request-ID"] = request_id
        ...     return response
    """
    REQUEST_ID_CTX.set(request_id)
    if session_id:
        SESSION_ID_CTX.set(session_id)


def clear_request_context() -> None:
    """Clear request-scoped logging context."""
    REQUEST_ID_CTX.set(None)
    SESSION_ID_CTX.set(None)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return REQUEST_ID_CTX.get()


def get_session_id() -> Optional[str]:
    """Get the current session ID from context."""
    return SESSION_ID_CTX.get()


# ============================================================
# JSON Log Handler (for production systems)
# ============================================================

try:
    # Try to import python-json-logger for better JSON formatting
    from pythonjsonlogger import jsonlogger

    class JSONFormatter(jsonlogger.JsonFormatter):
        """Enhanced JSON formatter with request tracking."""

        def add_fields(
            self,
            log_record: dict[str, Any],
            record: logging.LogRecord,
            message_dict: dict[str, Any],
        ) -> None:
            """Add custom fields to JSON log."""
            super().add_fields(log_record, record, message_dict)

            # Add request tracking
            request_id = REQUEST_ID_CTX.get()
            if request_id:
                log_record["request_id"] = request_id

            session_id = SESSION_ID_CTX.get()
            if session_id:
                log_record["session_id"] = session_id

            # Add performance info
            if hasattr(record, "execution_time_ms"):
                log_record["execution_time_ms"] = record.execution_time_ms

except ImportError:
    # Fall back to basic JSON formatter
    JSONFormatter = StructuredFormatter


# ============================================================
# Initialize Logging on Import
# ============================================================

# Setup logging when module is imported
setup_logging()

# Create module logger
logger = get_logger(__name__)

# Log initialization
logger.info(
    "Logging system initialized",
    extra={
        "log_level": settings.LOG_LEVEL,
        "log_format": settings.LOG_FORMAT,
        "log_file": str(settings.log_file_path),
    },
)


# Export public API
__all__ = [
    # Core functions
    "setup_logging",
    "get_logger",
    # Context management
    "RequestContext",
    "set_request_context",
    "clear_request_context",
    "get_request_id",
    "get_session_id",
    # Decorators
    "log_execution",
    "log_async_execution",
    # Context variables
    "REQUEST_ID_CTX",
    "SESSION_ID_CTX",
    # Formatters
    "ColoredFormatter",
    "StructuredFormatter",
    "DetailedFormatter",
    "JSONFormatter",
]
