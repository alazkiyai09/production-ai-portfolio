"""
Configuration Module for FraudDocs-RAG.

This module provides centralized configuration management using Pydantic settings.
All configuration is loaded from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Settings are loaded from:
    1. Environment variables
    2. .env file (if present)
    3. Default values defined here

    Example:
        >>> from fraud_docs_rag.config import settings
        >>> print(settings.APP_NAME)
        'FraudDocs-RAG'
    """

    # =========================================================================
    # Pydantic Configuration
    # =========================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Application Settings
    # =========================================================================
    APP_NAME: str = Field(
        default="FraudDocs-RAG",
        description="Application name"
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application version"
    )
    ENVIRONMENT: Literal["development", "demo", "production", "testing"] = Field(
        default="development",
        description="Environment type"
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    # =========================================================================
    # Server Settings
    # =========================================================================
    HOST: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port"
    )
    WORKERS: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Number of worker processes"
    )
    RELOAD: bool = Field(
        default=False,
        description="Enable auto-reload in development"
    )

    # =========================================================================
    # Data Paths
    # =========================================================================
    DATA_DIR: Path = Field(
        default=Path("./data"),
        description="Base data directory"
    )
    DOCUMENTS_DIR: Path = Field(
        default=Path("./data/documents"),
        description="Processed documents directory"
    )
    RAW_DOCUMENTS_DIR: Path = Field(
        default=Path("./data/raw"),
        description="Raw uploaded documents directory"
    )
    VECTOR_STORE_DIR: Path = Field(
        default=Path("./data/vector_store"),
        description="Vector store directory"
    )
    LOGS_DIR: Path = Field(
        default=Path("./logs"),
        description="Logs directory"
    )

    # =========================================================================
    # ChromaDB Settings
    # =========================================================================
    CHROMA_HOST: str = Field(
        default="localhost",
        description="ChromaDB host"
    )
    CHROMA_PORT: int = Field(
        default=8001,
        ge=1,
        le=65535,
        description="ChromaDB HTTP port"
    )
    CHROMA_PERSIST_DIRECTORY: Path = Field(
        default=Path("./data/chroma_db"),
        description="ChromaDB persistent storage directory"
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="financial_documents",
        description="ChromaDB collection name"
    )
    CHROMA_DISTANCE_FUNCTION: str = Field(
        default="cosine",
        description="Distance function for similarity"
    )

    # =========================================================================
    # Ollama Settings (Development)
    # =========================================================================
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    OLLAMA_MODEL: str = Field(
        default="llama3.2:3b",
        description="Ollama model name"
    )
    OLLAMA_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Ollama temperature"
    )
    OLLAMA_MAX_TOKENS: int = Field(
        default=2048,
        ge=1,
        description="Ollama max tokens"
    )
    OLLAMA_TIMEOUT: int = Field(
        default=300,
        ge=1,
        description="Ollama request timeout (seconds)"
    )

    # =========================================================================
    # GLM-4 Settings (Demo/Production)
    # =========================================================================
    ZHIPUAI_API_KEY: str = Field(
        default="",
        description="ZhipuAI API key for GLM-4"
    )
    GLM_BASE_URL: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4",
        description="GLM-4 API base URL"
    )
    GLM_MODEL: str = Field(
        default="glm-4-plus",
        description="GLM-4 model name"
    )
    GLM_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="GLM-4 temperature"
    )
    GLM_MAX_TOKENS: int = Field(
        default=4096,
        ge=1,
        description="GLM-4 max tokens"
    )
    GLM_TIMEOUT: int = Field(
        default=300,
        ge=1,
        description="GLM-4 request timeout (seconds)"
    )

    # =========================================================================
    # OpenAI Settings (Production)
    # =========================================================================
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API key"
    )
    OPENAI_BASE_URL: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL"
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name"
    )
    OPENAI_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="OpenAI temperature"
    )
    OPENAI_MAX_TOKENS: int = Field(
        default=4096,
        ge=1,
        description="OpenAI max tokens"
    )
    OPENAI_TIMEOUT: int = Field(
        default=300,
        ge=1,
        description="OpenAI request timeout (seconds)"
    )

    # =========================================================================
    # Embedding Settings
    # =========================================================================
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace embedding model"
    )
    EMBEDDING_DEVICE: str = Field(
        default="cpu",
        description="Device for embeddings (cpu, cuda)"
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        ge=1,
        description="Embedding batch size"
    )
    EMBEDDING_DIMENSION: int = Field(
        default=384,
        ge=1,
        description="Embedding vector dimension"
    )

    # =========================================================================
    # Chunking Settings
    # =========================================================================
    CHUNKING_STRATEGY: str = Field(
        default="semantic",
        description="Chunking strategy (semantic, fixed)"
    )
    CHUNK_SIZE: int = Field(
        default=512,
        ge=128,
        description="Target chunk size in characters"
    )
    CHUNK_OVERLAP: int = Field(
        default=50,
        ge=0,
        description="Chunk overlap in characters"
    )
    SEMANTIC_CHUNK_BUFFER_SIZE: int = Field(
        default=1,
        ge=1,
        description="Buffer size for semantic chunking"
    )
    SEMANTIC_CHUNK_BREAKPOINT_THRESHOLD: int = Field(
        default=60,
        ge=0,
        le=100,
        description="Breakpoint percentile threshold"
    )

    # =========================================================================
    # Retrieval Settings
    # =========================================================================
    TOP_K_RETRIEVAL: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of documents to retrieve"
    )
    SIMILARITY_THRESHOLD: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )
    RERANK_ENABLED: bool = Field(
        default=True,
        description="Enable cross-encoder reranking"
    )
    RERANK_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    RERANK_TOP_N: int = Field(
        default=5,
        ge=1,
        description="Number of results after reranking"
    )

    # =========================================================================
    # Document Processing Settings
    # =========================================================================
    SUPPORTED_DOCUMENT_FORMATS: str = Field(
        default="pdf,docx,doc,pptx,ppt,txt,md,html,htm",
        description="Supported document formats (comma-separated)"
    )
    MAX_FILE_SIZE_MB: int = Field(
        default=100,
        ge=1,
        description="Maximum file size in MB"
    )
    PARSING_STRATEGY: str = Field(
        default="auto",
        description="Document parsing strategy"
    )

    # =========================================================================
    # API Settings
    # =========================================================================
    API_V1_PREFIX: str = Field(
        default="/api/v1",
        description="API v1 prefix"
    )
    API_TITLE: str = Field(
        default="FraudDocs RAG API",
        description="API title"
    )
    API_DESCRIPTION: str = Field(
        default="RAG system for financial fraud detection documents",
        description="API description"
    )
    ALLOWED_ORIGINS: str = Field(
        default="*",
        description="CORS allowed origins (comma-separated)"
    )
    ENABLE_CORS: bool = Field(
        default=True,
        description="Enable CORS middleware"
    )
    MAX_UPLOAD_FILES: int = Field(
        default=10,
        ge=1,
        description="Maximum files per upload"
    )

    # =========================================================================
    # Rate Limiting Settings
    # =========================================================================
    RATE_LIMIT_ENABLED: bool = Field(
        default=False,
        description="Enable rate limiting"
    )
    RATE_LIMIT_REQUESTS: int = Field(
        default=100,
        ge=1,
        description="Requests per time window"
    )
    RATE_LIMIT_WINDOW: int = Field(
        default=60,
        ge=1,
        description="Time window in seconds"
    )

    # =========================================================================
    # Cache Settings
    # =========================================================================
    CACHE_ENABLED: bool = Field(
        default=True,
        description="Enable response caching"
    )
    CACHE_TTL: int = Field(
        default=3600,
        ge=1,
        description="Cache TTL in seconds"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for caching"
    )

    # =========================================================================
    # Monitoring Settings
    # =========================================================================
    ENABLE_METRICS: bool = Field(
        default=False,
        description="Enable Prometheus metrics"
    )
    METRICS_PORT: int = Field(
        default=9090,
        ge=1,
        le=65535,
        description="Metrics endpoint port"
    )
    ENABLE_TRACING: bool = Field(
        default=False,
        description="Enable distributed tracing"
    )
    JAEGER_HOST: str = Field(
        default="localhost",
        description="Jaeger host for tracing"
    )
    JAEGER_PORT: int = Field(
        default=6831,
        ge=1,
        le=65535,
        description="Jaeger port"
    )

    # =========================================================================
    # Security Settings
    # =========================================================================
    SECRET_KEY: Optional[str] = Field(
        default=None,
        description="Secret key for signing (must be set in production)"
    )
    API_KEY_HEADER: str = Field(
        default="X-API-Key",
        description="API key header name"
    )
    REQUIRE_API_KEY: bool = Field(
        default=False,
        description="Require API key for all endpoints"
    )
    ALLOWED_API_KEYS: str = Field(
        default="",
        description="Comma-separated list of allowed API keys"
    )

    # =========================================================================
    # Feature Flags
    # =========================================================================
    ENABLE_STREAMING: bool = Field(
        default=False,
        description="Enable streaming responses"
    )
    ENABLE_CITATIONS: bool = Field(
        default=True,
        description="Enable source citations"
    )
    ENABLE_QUERY_EXPANSION: bool = Field(
        default=False,
        description="Enable query expansion"
    )
    ENABLE_HYBRID_SEARCH: bool = Field(
        default=False,
        description="Enable hybrid search (vector + keyword)"
    )

    # =========================================================================
    # Validators
    # =========================================================================

    @field_validator("DATA_DIR", "DOCUMENTS_DIR", "RAW_DOCUMENTS_DIR",
                     "VECTOR_STORE_DIR", "LOGS_DIR", "CHROMA_PERSIST_DIRECTORY")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Create directories if they don't exist."""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate and normalize environment."""
        valid = {"development", "demo", "production", "testing"}
        if v.lower() not in valid:
            raise ValueError(f"ENVIRONMENT must be one of {valid}")
        return v.lower()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {valid}")
        return v.upper()

    @field_validator("SUPPORTED_DOCUMENT_FORMATS")
    @classmethod
    def validate_formats(cls, v: str) -> str:
        """Validate and normalize document formats."""
        formats = [f.strip().lower().lstrip(".") for f in v.split(",")]
        return ",".join(formats)

    @field_validator("ALLOWED_ORIGINS")
    @classmethod
    def validate_origins(cls, v: str) -> list[str]:
        """Convert origins string to list."""
        if v == "*":
            return ["*"]
        return [origin.strip() for origin in v.split(",")]

    @field_validator("ALLOWED_API_KEYS")
    @classmethod
    def validate_api_keys(cls, v: str) -> list[str]:
        """Convert API keys string to list."""
        if not v:
            return []
        return [key.strip() for key in v.split(",") if key.strip()]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT == "production"

    @property
    def is_demo(self) -> bool:
        """Check if running in demo mode."""
        return self.ENVIRONMENT == "demo"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.ENVIRONMENT == "testing"

    @property
    def supported_formats_list(self) -> list[str]:
        """Get supported formats as a list."""
        return [
            f".{f.strip()}" if not f.startswith(".") else f
            for f in self.SUPPORTED_DOCUMENT_FORMATS.split(",")
        ]

    @property
    def llm_provider(self) -> str:
        """Get the LLM provider based on environment."""
        if self.is_development or self.is_testing:
            return "ollama"
        elif self.is_demo:
            return "glm"
        else:
            return "openai"

    @property
    def llm_model(self) -> str:
        """Get the LLM model based on environment."""
        if self.is_development or self.is_testing:
            return self.OLLAMA_MODEL
        elif self.is_demo:
            return self.GLM_MODEL
        else:
            return self.OPENAI_MODEL

    def get_llm_config(self) -> dict:
        """Get LLM configuration as a dictionary."""
        if self.is_development or self.is_testing:
            return {
                "provider": "ollama",
                "model": self.OLLAMA_MODEL,
                "base_url": self.OLLAMA_BASE_URL,
                "temperature": self.OLLAMA_TEMPERATURE,
                "max_tokens": self.OLLAMA_MAX_TOKENS,
                "timeout": self.OLLAMA_TIMEOUT,
            }
        elif self.is_demo:
            return {
                "provider": "glm",
                "model": self.GLM_MODEL,
                "base_url": self.GLM_BASE_URL,
                "api_key": self.ZHIPUAI_API_KEY,
                "temperature": self.GLM_TEMPERATURE,
                "max_tokens": self.GLM_MAX_TOKENS,
                "timeout": self.GLM_TIMEOUT,
            }
        else:
            return {
                "provider": "openai",
                "model": self.OPENAI_MODEL,
                "base_url": self.OPENAI_BASE_URL,
                "api_key": self.OPENAI_API_KEY,
                "temperature": self.OPENAI_TEMPERATURE,
                "max_tokens": self.OPENAI_MAX_TOKENS,
                "timeout": self.OPENAI_TIMEOUT,
            }


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function caches the settings to avoid repeated loading
    from environment variables.

    Returns:
        Settings: Cached settings instance

    Example:
        >>> from fraud_docs_rag.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.APP_NAME)
    """
    return Settings()


# Export settings instance for easy importing
settings = get_settings()


# ===========================================================================
# Usage Examples
# ===========================================================================

if __name__ == "__main__":
    # Example 1: Display all settings
    print("FraudDocs-RAG Configuration")
    print("=" * 60)
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"App Name: {settings.APP_NAME}")
    print(f"Version: {settings.APP_VERSION}")
    print(f"Debug: {settings.DEBUG}")
    print()

    # Example 2: Display paths
    print("Paths:")
    print(f"  Data: {settings.DATA_DIR}")
    print(f"  Documents: {settings.DOCUMENTS_DIR}")
    print(f"  ChromaDB: {settings.CHROMA_PERSIST_DIRECTORY}")
    print(f"  Logs: {settings.LOGS_DIR}")
    print()

    # Example 3: Display LLM configuration
    print("LLM Configuration:")
    llm_config = settings.get_llm_config()
    print(f"  Provider: {llm_config['provider']}")
    print(f"  Model: {llm_config['model']}")
    print(f"  Temperature: {llm_config['temperature']}")
    print()

    # Example 4: Display retrieval settings
    print("Retrieval Settings:")
    print(f"  Top K: {settings.TOP_K_RETRIEVAL}")
    print(f"  Rerank: {settings.RERANK_ENABLED}")
    print(f"  Rerank Model: {settings.RERANK_MODEL}")
    print(f"  Rerank Top N: {settings.RERANK_TOP_N}")
    print()

    # Example 5: Display supported formats
    print("Supported Document Formats:")
    for fmt in settings.supported_formats_list:
        print(f"  - {fmt}")
