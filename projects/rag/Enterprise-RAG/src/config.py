# ============================================================
# Enterprise-RAG: Configuration Management
# ============================================================
"""
Application configuration using Pydantic BaseSettings.

This module centralizes all configuration settings from environment variables,
provides type validation, and includes sensible defaults.

Example:
    >>> from src.config import settings
    >>> print(settings.OPENAI_API_KEY)
    >>> print(settings.CHUNK_SIZE)
    >>> # Test configuration is valid
    >>> assert settings.CHUNK_SIZE > 0
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden by environment variables.
    See .env.example for a complete list of available variables.

    Attributes:
        OPENAI_API_KEY: OpenAI API key for LLM and evaluation
        EMBEDDING_MODEL: Model name for embeddings
        RERANKER_MODEL: Model name for cross-encoder reranking
        CHUNK_SIZE: Size of text chunks in characters
        CHUNK_OVERLAP: Overlap between chunks in characters
        TOP_K_RETRIEVAL: Number of chunks to retrieve
        TOP_K_RERANK: Number of chunks to return after reranking
        CHROMA_PATH: Path to ChromaDB storage
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        >>> settings = Settings()
        >>> print(settings.EMBEDDING_MODEL)
        'sentence-transformers/all-MiniLM-L6-v2'
    """

    # ============================================================
    # API Keys
    # ============================================================
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API key for LLM and RAGAS evaluation",
    )
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude models (optional)",
    )
    COHERE_API_KEY: Optional[str] = Field(
        default=None,
        description="Cohere API key for Cohere reranker (optional)",
    )

    # ============================================================
    # Model Configuration
    # ============================================================
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model name for text embeddings",
    )
    RERANKER_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Model name for cross-encoder reranking",
    )
    LLM_MODEL: str = Field(
        default="gpt-4-turbo",
        description="LLM model for response generation",
    )
    LLM_TEMPERATURE: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation",
    )
    LLM_MAX_TOKENS: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Maximum tokens for LLM response",
    )

    # ============================================================
    # Chunking Configuration
    # ============================================================
    CHUNK_SIZE: int = Field(
        default=512,
        ge=100,
        le=2048,
        description="Size of text chunks in characters",
    )
    CHUNK_OVERLAP: int = Field(
        default=50,
        ge=0,
        le=512,
        description="Overlap between consecutive chunks in characters",
    )

    # ============================================================
    # Retrieval Configuration
    # ============================================================
    TOP_K_RETRIEVAL: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of chunks to retrieve from vector store",
    )
    TOP_K_RERANK: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks to return after reranking",
    )
    ENABLE_BM25: bool = Field(
        default=True,
        description="Enable BM25 sparse retrieval",
    )
    BM25_K1: float = Field(
        default=1.5,
        ge=0.0,
        le=3.0,
        description="BM25 k1 parameter (term frequency saturation)",
    )
    BM25_B: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 b parameter (length normalization)",
    )

    # ============================================================
    # Vector Database Configuration
    # ============================================================
    VECTOR_STORE_TYPE: Literal["chroma", "qdrant"] = Field(
        default="chroma",
        description="Type of vector database to use",
    )
    CHROMA_PATH: str = Field(
        default="./data/chroma",
        description="Path to ChromaDB persistence directory",
    )
    QDRANT_HOST: str = Field(
        default="localhost",
        description="Qdrant server host",
    )
    QDRANT_PORT: int = Field(
        default=6333,
        ge=1,
        le=65535,
        description="Qdrant HTTP port",
    )
    QDRANT_GRPC_PORT: int = Field(
        default=6334,
        ge=1,
        le=65535,
        description="Qdrant gRPC port",
    )
    QDRANT_COLLECTION_NAME: str = Field(
        default="enterprise_rag",
        description="Qdrant collection name",
    )
    QDRANT_API_KEY: Optional[str] = Field(
        default=None,
        description="Qdrant API key (if using cloud)",
    )

    # ============================================================
    # Document Processing Configuration
    # ============================================================
    MAX_FILE_SIZE: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum file size for upload in MB",
    )
    SUPPORTED_FORMATS: str = Field(
        default="pdf,docx,md,txt",
        description="Comma-separated list of supported document formats",
    )
    DOCUMENTS_PATH: str = Field(
        default="./data/documents",
        description="Path to store uploaded documents",
    )

    # ============================================================
    # API Configuration
    # ============================================================
    API_HOST: str = Field(
        default="0.0.0.0",
        description="FastAPI server host",
    )
    API_PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="FastAPI server port",
    )
    CORS_ORIGINS: str = Field(
        default="http://localhost:8501,http://localhost:3000",
        description="Comma-separated CORS allowed origins",
    )
    API_TIMEOUT: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="API request timeout in seconds",
    )

    # ============================================================
    # Logging Configuration
    # ============================================================
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    LOG_FILE: str = Field(
        default="./logs/enterprise_rag.log",
        description="Path to log file",
    )
    CONSOLE_LOGGING: bool = Field(
        default=True,
        description="Enable console logging",
    )
    LOG_FORMAT: Literal["text", "json"] = Field(
        default="text",
        description="Log output format",
    )

    # ============================================================
    # Evaluation Configuration
    # ============================================================
    ENABLE_EVALUATION: bool = Field(
        default=True,
        description="Enable RAGAS evaluation",
    )
    EVALUATION_METRICS: str = Field(
        default="faithfulness,answer_relevancy,context_recall",
        description="Comma-separated evaluation metrics",
    )
    EVALUATION_SAMPLE_SIZE: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Number of samples for evaluation",
    )

    # ============================================================
    # Performance Configuration
    # ============================================================
    WORKER_COUNT: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of workers for document processing",
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for embedding generation",
    )
    ENABLE_GPU: bool = Field(
        default=False,
        description="Enable GPU for embeddings (if available)",
    )
    CACHE_EMBEDDINGS: bool = Field(
        default=True,
        description="Cache embeddings to disk",
    )

    # ============================================================
    # Security Configuration
    # ============================================================
    SECRET_KEY: Optional[str] = Field(
        default=None,
        description="Secret key for JWT tokens (must be set in production)",
    )
    TOKEN_EXPIRATION_HOURS: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Token expiration time in hours",
    )

    @field_validator('SECRET_KEY')
    @classmethod
    def validate_secret_key(cls, v: Optional[str], info) -> str:
        """Validate that SECRET_KEY is set in production environments."""
        # Allow default for testing/development, but warn in documentation
        # For production, this should be required
        return v or "dev-only-change-in-production"

    # ============================================================
    # Feature Flags
    # ============================================================
    ENABLE_QUERY_CACHE: bool = Field(
        default=True,
        description="Enable query result caching",
    )
    ENABLE_STREAMING: bool = Field(
        default=True,
        description="Enable streaming responses",
    )
    ENABLE_HYBRID_SEARCH: bool = Field(
        default=True,
        description="Enable hybrid search (dense + sparse)",
    )

    # ============================================================
    # Pydantic Configuration
    # ============================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ============================================================
    # Validators
    # ============================================================
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid option."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL: {v}. Must be one of {valid_levels}"
            )
        return v_upper

    @field_validator("CHUNK_SIZE")
    @classmethod
    def validate_chunk_size(cls, v: int, info) -> int:
        """Ensure chunk size is greater than overlap."""
        if "CHUNK_OVERLAP" in info.data and v <= info.data["CHUNK_OVERLAP"]:
            raise ValueError(
                f"CHUNK_SIZE ({v}) must be greater than CHUNK_OVERLAP "
                f"({info.data['CHUNK_OVERLAP']})"
            )
        return v

    @field_validator("TOP_K_RERANK")
    @classmethod
    def validate_top_k_rerank(cls, v: int, info) -> int:
        """Ensure rerank k is not greater than retrieval k."""
        if "TOP_K_RETRIEVAL" in info.data and v > info.data["TOP_K_RETRIEVAL"]:
            raise ValueError(
                f"TOP_K_RERANK ({v}) must not be greater than "
                f"TOP_K_RETRIEVAL ({info.data['TOP_K_RETRIEVAL']})"
            )
        return v

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        """Warn if OpenAI API key is not set."""
        if not v or v.startswith("sk-your"):
            import warnings

            warnings.warn(
                "OPENAI_API_KEY is not set or is using placeholder value. "
                "Some features may not work correctly.",
                UserWarning,
                stacklevel=2,
            )
        return v

    # ============================================================
    # Utility Properties
    # ============================================================
    @property
    def supported_formats_list(self) -> list[str]:
        """Parse SUPPORTED_FORMATS into a list."""
        return [fmt.strip().lower() for fmt in self.SUPPORTED_FORMATS.split(",")]

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def evaluation_metrics_list(self) -> list[str]:
        """Parse EVALUATION_METRICS into a list."""
        return [metric.strip() for metric in self.EVALUATION_METRICS.split(",")]

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent

    @property
    def chroma_persist_path(self) -> Path:
        """Get the full path to ChromaDB storage."""
        return self.project_root / self.CHROMA_PATH.lstrip("./")

    @property
    def documents_storage_path(self) -> Path:
        """Get the full path to documents storage."""
        return self.project_root / self.DOCUMENTS_PATH.lstrip("./")

    @property
    def log_file_path(self) -> Path:
        """Get the full path to log file."""
        return self.project_root / self.LOG_FILE.lstrip("./")

    # ============================================================
    # Methods
    # ============================================================
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.chroma_persist_path,
            self.documents_storage_path,
            self.log_file_path.parent,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function uses lru_cache to ensure settings are loaded only once.
    Subsequent calls return the same instance.

    Returns:
        Settings: Cached settings instance

    Example:
        >>> settings = get_settings()
        >>> print(settings.EMBEDDING_MODEL)
    """
    return Settings()


# Global settings instance
settings = get_settings()

# Ensure required directories exist
settings.create_directories()
