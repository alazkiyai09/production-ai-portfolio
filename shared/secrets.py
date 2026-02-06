# ============================================================
# Secrets and Configuration Management for All Projects
# ============================================================
"""
Centralized secrets management using Pydantic BaseSettings.

This module provides secure environment variable loading with validation,
support for different environments (dev, staging, prod), and centralized
configuration for API keys, databases, Redis, Celery, and JWT settings.

Example:
    >>> from shared.secrets import get_settings, Settings
    >>>
    >>> # Get the singleton settings instance
    >>> settings = get_settings()
    >>>
    >>> # Access configuration
    >>> print(settings.OPENAI_API_KEY)
    >>> print(settings.database_url)
    >>>
    >>> # Check environment
    >>> if settings.is_production:
    ...     print("Running in production mode")
    >>>
    >>> # Get database connection string
    >>> db_url = settings.get_database_url()
"""

import os
import secrets as python_secrets
import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional, Literal

from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ============================================================
# Environment Enum
# ============================================================

class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"
    PROD = "prod"
    TEST = "test"

    @classmethod
    def normalize(cls, env: str) -> "Environment":
        """
        Normalize environment string to standard value.

        Args:
            env: Environment string to normalize

        Returns:
            Normalized Environment enum value

        Example:
            >>> Environment.normalize("prod")
            <Environment.PRODUCTION: 'production'>
            >>> Environment.normalize("dev")
            <Environment.DEVELOPMENT: 'development'>
        """
        env_lower = env.lower()
        if env_lower in ("prod", "production"):
            return cls.PRODUCTION
        elif env_lower in ("dev", "development"):
            return cls.DEVELOPMENT
        elif env_lower == "staging":
            return cls.STAGING
        elif env_lower == "test":
            return cls.TEST
        else:
            return cls.DEVELOPMENT


# ============================================================
# Base Settings Class
# ============================================================

class BaseSecretSettings(BaseSettings):
    """
    Base settings class with common configuration.

    This class provides the foundation for all settings with
    standard Pydantic Settings configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def __init__(self, **kwargs):
        """Initialize settings and log environment."""
        super().__init__(**kwargs)
        logger.info(f"Settings loaded for environment: {self.ENVIRONMENT}")


# ============================================================
# Main Settings Class
# ============================================================

class Settings(BaseSecretSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden by environment variables.
    Missing required secrets will raise a validation error.

    Attributes:
        ENVIRONMENT: Application environment (dev, staging, prod)
        DEBUG: Enable debug mode
        API Keys: OpenAI, Anthropic, Cohere, etc.
        Database: Connection strings for PostgreSQL, Redis
        JWT: Secret key and algorithm configuration
        Celery: Broker and backend configuration
        Security: Various security settings

    Example:
        >>> settings = Settings()
        >>> print(settings.OPENAI_API_KEY)
        >>> print(settings.database_url)
    """

    # ============================================================
    # Environment Configuration
    # ============================================================
    ENVIRONMENT: str = Field(
        default="development",
        description="Application environment (development, staging, production, test)",
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode with additional logging",
    )
    APP_NAME: str = Field(
        default="agenticflow",
        description="Application name",
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application version",
    )

    # ============================================================
    # API Keys - LLM Providers
    # ============================================================
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key for GPT models",
    )
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude models",
    )
    COHERE_API_KEY: Optional[str] = Field(
        default=None,
        description="Cohere API key for Cohere models",
    )
    HUGGINGFACE_API_KEY: Optional[str] = Field(
        default=None,
        description="HuggingFace API key for model access",
    )
    TOGETHER_API_KEY: Optional[str] = Field(
        default=None,
        description="Together AI API key",
    )
    OPENROUTER_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenRouter API key",
    )

    # ============================================================
    # API Keys - Other Services
    # ============================================================
    TAVILY_API_KEY: Optional[str] = Field(
        default=None,
        description="Tavily Search API key for web search",
    )
    SERPAPI_KEY: Optional[str] = Field(
        default=None,
        description="SerpAPI key for Google Search",
    )
    LANGCHAIN_API_KEY: Optional[str] = Field(
        default=None,
        description="LangSmith API key for tracing",
    )
    LANGCHAIN_TRACING: bool = Field(
        default=False,
        description="Enable LangSmith tracing",
    )
    LANGCHAIN_PROJECT: str = Field(
        default="agenticflow",
        description="LangSmith project name",
    )

    # ============================================================
    # Database Configuration
    # ============================================================
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection string",
    )
    DATABASE_HOST: str = Field(
        default="localhost",
        description="Database host",
    )
    DATABASE_PORT: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="Database port",
    )
    DATABASE_NAME: str = Field(
        default="agenticflow",
        description="Database name",
    )
    DATABASE_USER: str = Field(
        default="postgres",
        description="Database user",
    )
    DATABASE_PASSWORD: Optional[SecretStr] = Field(
        default=None,
        description="Database password",
    )

    # SQLite fallback
    SQLITE_DB_PATH: str = Field(
        default="./data/app.db",
        description="SQLite database path for fallback",
    )

    # ============================================================
    # Redis Configuration
    # ============================================================
    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Redis connection string (overrides individual settings)",
    )
    REDIS_HOST: str = Field(
        default="localhost",
        description="Redis host",
    )
    REDIS_PORT: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis port",
    )
    REDIS_DB: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database number",
    )
    REDIS_PASSWORD: Optional[SecretStr] = Field(
        default=None,
        description="Redis password",
    )
    REDIS_USE_SSL: bool = Field(
        default=False,
        description="Use SSL for Redis connection",
    )
    REDIS_MAX_CONNECTIONS: int = Field(
        default=50,
        ge=1,
        description="Maximum Redis connection pool size",
    )

    # ============================================================
    # Celery Configuration
    # ============================================================
    CELERY_BROKER_URL: Optional[str] = Field(
        default=None,
        description="Celery broker URL (Redis or RabbitMQ)",
    )
    CELERY_RESULT_BACKEND: Optional[str] = Field(
        default=None,
        description="Celery result backend URL",
    )
    CELERY_TASK_SERIALIZER: str = Field(
        default="json",
        description="Celery task serializer",
    )
    CELERY_RESULT_SERIALIZER: str = Field(
        default="json",
        description="Celery result serializer",
    )
    CELERY_ACCEPT_CONTENT: list[str] = Field(
        default_factory=lambda: ["json"],
        description="Accepted content types",
    )
    CELERY_TIMEZONE: str = Field(
        default="UTC",
        description="Celery timezone",
    )
    CELERY_TASK_TRACK_STARTED: bool = Field(
        default=True,
        description="Track when tasks start",
    )
    CELERY_TASK_TIME_LIMIT: int = Field(
        default=30 * 60,
        ge=0,
        description="Task time limit in seconds",
    )
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = Field(
        default=4,
        ge=1,
        description="Worker prefetch multiplier",
    )
    CELERY_WORKER_MAX_TASKS_PER_CHILD: int = Field(
        default=1000,
        ge=1,
        description="Max tasks per worker before restart",
    )

    # ============================================================
    # JWT Configuration
    # ============================================================
    SECRET_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Secret key for JWT token signing",
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT algorithm for token signing",
    )
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=1,
        description="Access token expiration in minutes",
    )
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        ge=1,
        description="Refresh token expiration in days",
    )

    # ============================================================
    # API Configuration
    # ============================================================
    API_HOST: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    API_PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API server port",
    )
    API_PREFIX: str = Field(
        default="/api",
        description="API URL prefix",
    )
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8501",
        description="Comma-separated CORS allowed origins",
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS",
    )
    CORS_ALLOW_METHODS: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed CORS methods",
    )
    CORS_ALLOW_HEADERS: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS headers",
    )

    # ============================================================
    # Vector Store Configuration
    # ============================================================
    CHROMA_PATH: str = Field(
        default="./data/chroma",
        description="ChromaDB persistence directory",
    )
    QDRANT_HOST: str = Field(
        default="localhost",
        description="Qdrant host",
    )
    QDRANT_PORT: int = Field(
        default=6333,
        ge=1,
        le=65535,
        description="Qdrant HTTP port",
    )
    QDRANT_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Qdrant API key",
    )
    QDRANT_COLLECTION: str = Field(
        default="agenticflow",
        description="Default Qdrant collection name",
    )

    # ============================================================
    # Model Configuration
    # ============================================================
    DEFAULT_OPENAI_MODEL: str = Field(
        default="gpt-4-turbo-preview",
        description="Default OpenAI model",
    )
    DEFAULT_ANTHROPIC_MODEL: str = Field(
        default="claude-3-opus-20240229",
        description="Default Anthropic model",
    )
    MODEL_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default model temperature",
    )
    MODEL_MAX_TOKENS: int = Field(
        default=2048,
        ge=1,
        description="Default max tokens for generation",
    )

    # ============================================================
    # Logging Configuration
    # ============================================================
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    LOG_FORMAT: Literal["text", "json"] = Field(
        default="text",
        description="Log output format",
    )
    LOG_FILE: Optional[str] = Field(
        default=None,
        description="Log file path (None for stdout only)",
    )
    LOG_ROTATION: str = Field(
        default="10 MB",
        description="Log rotation size",
    )
    LOG_RETENTION: str = Field(
        default="30 days",
        description="Log retention period",
    )

    # ============================================================
    # Rate Limiting Configuration
    # ============================================================
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting",
    )
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=60,
        ge=1,
        description="Requests per minute per user",
    )
    RATE_LIMIT_BURST: int = Field(
        default=10,
        ge=1,
        description="Burst size for rate limiting",
    )

    # ============================================================
    # Feature Flags
    # ============================================================
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )
    ENABLE_TRACING: bool = Field(
        default=False,
        description="Enable distributed tracing",
    )
    ENABLE_CACHE: bool = Field(
        default=True,
        description="Enable response caching",
    )
    CACHE_TTL: int = Field(
        default=3600,
        ge=0,
        description="Cache TTL in seconds",
    )

    # ============================================================
    # Security Settings
    # ============================================================
    SECURE_COOKIES: bool = Field(
        default=True,
        description="Set secure flag on cookies",
    )
    COOKIE_SAMESITE: Literal["lax", "strict", "none"] = Field(
        default="lax",
        description="Cookie same-site policy",
    )
    MAX_REQUEST_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Max request size in MB",
    )

    # ============================================================
    # Environment-specific Overrides
    # ============================================================
    @field_validator("ENVIRONMENT", mode="before")
    @classmethod
    def normalize_environment(cls, v: str) -> str:
        """Normalize environment name."""
        if isinstance(v, str):
            return Environment.normalize(v).value
        return v

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        """Normalize and validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = str(v).upper()
        if v_upper not in valid_levels:
            logger.warning(f"Invalid LOG_LEVEL: {v}, using INFO")
            return "INFO"
        return v_upper

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> str:
        """Ensure CORS origins is a string."""
        if isinstance(v, list):
            return ",".join(v)
        return v

    @field_validator("CELERY_ACCEPT_CONTENT", mode="before")
    @classmethod
    def parse_celery_content(cls, v: str | list[str]) -> list[str]:
        """Parse Celery accepted content types."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v

    @field_validator("CORS_ALLOW_METHODS", mode="before")
    @classmethod
    def parse_cors_methods(cls, v: str | list[str]) -> list[str]:
        """Parse CORS allowed methods."""
        if isinstance(v, str):
            return [item.strip().upper() for item in v.split(",")]
        return v

    @field_validator("CORS_ALLOW_HEADERS", mode="before")
    @classmethod
    def parse_cors_headers(cls, v: str | list[str]) -> list[str]:
        """Parse CORS allowed headers."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v

    # ============================================================
    # Environment Detection Properties
    # ============================================================
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT.lower() in ("dev", "development")

    @property
    def is_staging(self) -> bool:
        """Check if running in staging mode."""
        return self.ENVIRONMENT.lower() == "staging"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT.lower() in ("prod", "production")

    @property
    def is_test(self) -> bool:
        """Check if running in test mode."""
        return self.ENVIRONMENT.lower() == "test"

    @property
    def should_debug(self) -> bool:
        """Check if debug mode should be enabled."""
        return self.DEBUG or self.is_development

    # ============================================================
    # Secret Access with Validation
    # ============================================================
    @property
    def secret_key(self) -> str:
        """
        Get JWT secret key.

        Returns:
            Secret key string

        Raises:
            ValueError: If SECRET_KEY is not set in production
        """
        if self.SECRET_KEY is None:
            if self.is_production:
                raise ValueError(
                    "SECRET_KEY must be set in production. "
                    "Set the SECRET_KEY environment variable."
                )
            # Generate a warning for non-production
            logger.warning(
                "SECRET_KEY not set, using generated key. "
                "This will cause tokens to be invalidated on restart."
            )
            return python_secrets.token_urlsafe(32)
        return self.SECRET_KEY.get_secret_value()

    # ============================================================
    # Database Connection Methods
    # ============================================================
    @property
    def database_url(self) -> str:
        """
        Get the database connection URL.

        Returns:
            Database connection string

        Example:
            >>> settings = get_settings()
            >>> url = settings.database_url
            >>> engine = create_engine(url)
        """
        if self.DATABASE_URL:
            return self.DATABASE_URL

        # Build connection string from components
        password = (
            self.DATABASE_PASSWORD.get_secret_value()
            if self.DATABASE_PASSWORD
            else ""
        )
        if password:
            password = f":{password}"

        return (
            f"postgresql://{self.DATABASE_USER}{password}@"
            f"{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
        )

    @property
    def sqlite_url(self) -> str:
        """Get SQLite database URL for fallback."""
        return f"sqlite:///{self.SQLITE_DB_PATH}"

    @property
    def use_postgres(self) -> bool:
        """Check if PostgreSQL should be used."""
        return bool(self.DATABASE_URL or self.DATABASE_PASSWORD)

    def get_database_url(self, force_sqlite: bool = False) -> str:
        """
        Get database URL with optional fallback to SQLite.

        Args:
            force_sqlite: Force SQLite even if PostgreSQL is configured

        Returns:
            Database connection string

        Example:
            >>> settings = get_settings()
            >>> # Use SQLite
            >>> url = settings.get_database_url(force_sqlite=True)
            >>> # Use PostgreSQL if available
            >>> url = settings.get_database_url()
        """
        if force_sqlite or not self.use_postgres:
            return self.sqlite_url
        return self.database_url

    # ============================================================
    # Redis Connection Methods
    # ============================================================
    @property
    def redis_url(self) -> str:
        """
        Get Redis connection URL.

        Returns:
            Redis connection string

        Example:
            >>> settings = get_settings()
            >>> url = settings.redis_url
            >>> redis = Redis.from_url(url)
        """
        if self.REDIS_URL:
            return self.REDIS_URL

        # Build connection string from components
        password = (
            f":{self.REDIS_PASSWORD.get_secret_value()}@"
            if self.REDIS_PASSWORD
            else ""
        )
        scheme = "rediss" if self.REDIS_USE_SSL else "redis"

        return f"{scheme}://{password}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def use_redis(self) -> bool:
        """Check if Redis should be used."""
        return bool(self.REDIS_URL or self.is_production)

    # ============================================================
    # Celery Configuration Methods
    # ============================================================
    @property
    def celery_broker_url(self) -> str:
        """
        Get Celery broker URL.

        Returns:
            Celery broker connection string

        Example:
            >>> settings = get_settings()
            >>> broker = settings.celery_broker_url
            >>> app = Celery(broker=broker)
        """
        if self.CELERY_BROKER_URL:
            return self.CELERY_BROKER_URL
        # Default to Redis
        return self.redis_url

    @property
    def celery_result_backend(self) -> str:
        """
        Get Celery result backend URL.

        Returns:
            Celery result backend connection string
        """
        if self.CELERY_RESULT_BACKEND:
            return self.CELERY_RESULT_BACKEND
        # Default to Redis
        return self.redis_url

    def get_celery_config(self) -> dict:
        """
        Get complete Celery configuration dictionary.

        Returns:
            Dictionary with all Celery settings

        Example:
            >>> settings = get_settings()
            >>> celery_config = settings.get_celery_config()
            >>> app = Celery('tasks', **celery_config)
        """
        return {
            "broker_url": self.celery_broker_url,
            "result_backend": self.celery_result_backend,
            "task_serializer": self.CELERY_TASK_SERIALIZER,
            "result_serializer": self.CELERY_RESULT_SERIALIZER,
            "accept_content": self.CELERY_ACCEPT_CONTENT,
            "timezone": self.CELERY_TIMEZONE,
            "task_track_started": self.CELERY_TASK_TRACK_STARTED,
            "task_time_limit": self.CELERY_TASK_TIME_LIMIT,
            "worker_prefetch_multiplier": self.CELERY_WORKER_PREFETCH_MULTIPLIER,
            "worker_max_tasks_per_child": self.CELERY_WORKER_MAX_TASKS_PER_CHILD,
        }

    # ============================================================
    # Vector Store Configuration
    # ============================================================
    @property
    def qdrant_url(self) -> str:
        """
        Get Qdrant connection URL.

        Returns:
            Qdrant connection string
        """
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    @property
    def qdrant_location(self) -> str:
        """
        Get Qdrant location for client configuration.

        Returns:
            Qdrant location (URL or :memory:)
        """
        if self.is_test:
            return ":memory:"
        return self.qdrant_url

    @property
    def chroma_persist_path(self) -> Path:
        """
        Get ChromaDB persist directory as Path.

        Returns:
            Path to ChromaDB storage directory
        """
        return Path(self.CHROMA_PATH)

    # ============================================================
    # Utility Methods
    # ============================================================
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    def get_openai_client_config(self) -> dict:
        """
        Get OpenAI client configuration.

        Returns:
            Dictionary with OpenAI client settings

        Example:
            >>> settings = get_settings()
            >>> config = settings.get_openai_client_config()
            >>> client = OpenAI(**config)
        """
        return {
            "api_key": self.OPENAI_API_KEY,
        }

    def get_anthropic_client_config(self) -> dict:
        """
        Get Anthropic client configuration.

        Returns:
            Dictionary with Anthropic client settings

        Example:
            >>> settings = get_settings()
            >>> config = settings.get_anthropic_client_config()
            >>> client = Anthropic(**config)
        """
        return {
            "api_key": self.ANTHROPIC_API_KEY,
        }

    def validate_required_secrets(self) -> list[str]:
        """
        Validate that required secrets are set.

        Returns:
            List of missing required secret names

        Example:
            >>> settings = get_settings()
            >>> missing = settings.validate_required_secrets()
            >>> if missing:
            ...     print(f"Missing secrets: {missing}")
        """
        missing = []

        if self.is_production:
            if not self.OPENAI_API_KEY:
                missing.append("OPENAI_API_KEY")
            if not self.SECRET_KEY:
                missing.append("SECRET_KEY")
            if not self.use_postgres:
                missing.append("DATABASE_URL or DATABASE_PASSWORD")

        return missing

    def create_directories(self) -> None:
        """
        Create necessary directories for data storage.

        Creates directories for:
        - SQLite database
        - ChromaDB storage
        - Log files

        Example:
            >>> settings = get_settings()
            >>> settings.create_directories()
        """
        directories = [
            Path(self.SQLITE_DB_PATH).parent,
            self.chroma_persist_path,
        ]

        if self.LOG_FILE:
            directories.append(Path(self.LOG_FILE).parent)

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created directories: {[str(d) for d in directories]}")

    def get_logging_config(self) -> dict:
        """
        Get logging configuration dictionary.

        Returns:
            Dictionary suitable for logging.config.dictConfig

        Example:
            >>> settings = get_settings()
            >>> config = settings.get_logging_config()
            >>> logging.config.dictConfig(config)
        """
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": (
                        "%(asctime)s - %(name)s - %(levelname)s - "
                        "%(funcName)s:%(lineno)d - %(message)s"
                    ),
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": (
                        "%(asctime)s - %(name)s - %(levelname)s - "
                        "%(funcName)s:%(lineno)d - %(message)s"
                    ),
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json" if self.LOG_FORMAT == "json" else "default",
                    "level": self.LOG_LEVEL,
                },
            },
            "root": {
                "level": self.LOG_LEVEL,
                "handlers": ["console"],
            },
            "loggers": {
                "uvicorn": {"level": "INFO"},
                "uvicorn.access": {"level": "INFO"},
                "fastapi": {"level": "INFO"},
            },
        }


# ============================================================
# Environment-Specific Settings Classes
# ============================================================

class DevelopmentSettings(Settings):
    """Development environment settings with dev defaults."""

    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    ENABLE_TRACING: bool = False
    SECURE_COOKIES: bool = False


class StagingSettings(Settings):
    """Staging environment settings."""

    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    SECURE_COOKIES: bool = True


class ProductionSettings(Settings):
    """Production environment settings with strict defaults."""

    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    SECURE_COOKIES: bool = True
    COOKIE_SAMESITE: Literal["lax", "strict", "none"] = "strict"
    ENABLE_TRACING: bool = True
    ENABLE_METRICS: bool = True


class TestSettings(Settings):
    """Test environment settings."""

    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    SQLITE_DB_PATH: str = ":memory:"
    CHROMA_PATH: str = ":memory:"
    RATE_LIMIT_ENABLED: bool = False


# ============================================================
# Settings Factory
# ============================================================

@lru_cache
def get_settings(env: Optional[str] = None) -> Settings:
    """
    Get cached settings instance for the specified environment.

    This function uses lru_cache to ensure settings are loaded only once.
    Subsequent calls with the same environment return the same instance.

    Args:
        env: Optional environment override (development, staging, production, test)

    Returns:
        Settings: Cached settings instance

    Example:
        >>> settings = get_settings()
        >>> print(settings.ENVIRONMENT)
        'development'

        >>> # Override environment
        >>> prod_settings = get_settings(env="production")
        >>> print(prod_settings.ENVIRONMENT)
        'production'
    """
    # Determine which settings class to use
    env_to_use = env or os.getenv("ENVIRONMENT", "development")
    env_normalized = Environment.normalize(env_to_use)

    settings_classes = {
        Environment.DEVELOPMENT: DevelopmentSettings,
        Environment.STAGING: StagingSettings,
        Environment.PRODUCTION: ProductionSettings,
        Environment.TEST: TestSettings,
    }

    settings_class = settings_classes.get(env_normalized, Settings)
    return settings_class()


def reload_settings() -> Settings:
    """
    Force reload of settings, clearing the cache.

    Use this when environment variables may have changed.

    Returns:
        Settings: Fresh settings instance

    Example:
        >>> os.environ["OPENAI_API_KEY"] = "new-key"
        >>> settings = reload_settings()
    """
    get_settings.cache_clear()
    return get_settings()


# ============================================================
# Global Settings Instance
# ============================================================

settings = get_settings()

# Create necessary directories on import
try:
    settings.create_directories()
except Exception as e:
    logger.warning(f"Could not create directories: {e}")


# ============================================================
# Validation on Import
# ============================================================

def _validate_on_import() -> None:
    """Validate required settings on module import."""
    missing = settings.validate_required_secrets()
    if missing and settings.is_production:
        raise ValueError(
            f"Missing required secrets for production: {', '.join(missing)}. "
            "Please set these environment variables before deploying."
        )


# Run validation in production
if settings.is_production:
    try:
        _validate_on_import()
    except ValueError as e:
        logger.error(str(e))
        raise


# ============================================================
# Export
# ============================================================

__all__ = [
    # Settings Classes
    "Settings",
    "DevelopmentSettings",
    "StagingSettings",
    "ProductionSettings",
    "TestSettings",
    "BaseSecretSettings",
    # Environment Enum
    "Environment",
    # Functions
    "get_settings",
    "reload_settings",
    # Global Instance
    "settings",
]
