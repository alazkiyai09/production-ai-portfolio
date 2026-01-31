"""
Configuration management for LLMOps-Eval using Pydantic Settings.

This module centralizes all configuration including API keys, model settings,
evaluation parameters, and deployment configuration.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Attributes:
        API keys for LLM providers
        Model configurations
        Evaluation parameters
        API and dashboard settings
        Monitoring configuration
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===========================================
    # LLM Provider API Keys
    # ===========================================
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    cohere_api_key: str = Field(default="", description="Cohere API key")
    huggingface_api_key: str = Field(default="", description="HuggingFace API key")

    # ===========================================
    # Model Configuration
    # ===========================================
    default_openai_model: str = Field(
        default="gpt-4-turbo-preview",
        description="Default OpenAI model to evaluate",
    )
    default_anthropic_model: str = Field(
        default="claude-3-opus-20240229",
        description="Default Anthropic model to evaluate",
    )
    default_cohere_model: str = Field(
        default="command",
        description="Default Cohere model to evaluate",
    )
    default_local_model_path: str = Field(
        default="",
        description="Path to local model for evaluation",
    )

    # ===========================================
    # Evaluation Settings
    # ===========================================
    max_concurrent_evaluations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent evaluation tasks",
    )
    request_timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed requests",
    )
    retry_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Exponential backoff multiplier for retries",
    )

    # ===========================================
    # Judge Model Configuration
    # ===========================================
    judge_model_provider: Literal["openai", "anthropic", "cohere"] = Field(
        default="openai",
        description="Provider for LLM-as-judge evaluations",
    )
    judge_model_name: str = Field(
        default="gpt-4-turbo-preview",
        description="Model name for LLM-as-judge evaluations",
    )

    # ===========================================
    # Evaluation Metrics
    # ===========================================
    enable_accuracy_metrics: bool = Field(
        default=True,
        description="Enable accuracy and quality metrics",
    )
    enable_latency_metrics: bool = Field(
        default=True,
        description="Enable latency metrics (TTFT, total time)",
    )
    enable_cost_metrics: bool = Field(
        default=True,
        description="Enable cost tracking metrics",
    )
    enable_hallucination_metrics: bool = Field(
        default=True,
        description="Enable hallucination detection metrics",
    )
    enable_safety_metrics: bool = Field(
        default=True,
        description="Enable toxicity and safety checks",
    )
    enable_format_compliance_metrics: bool = Field(
        default=True,
        description="Enable format compliance validation",
    )
    semantic_similarity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Threshold for semantic similarity matching",
    )

    # ===========================================
    # Paths
    # ===========================================
    data_dir: Path = Field(default=Path("./data"), description="Base data directory")
    datasets_dir: Path = Field(
        default=Path("./data/datasets"),
        description="Directory for test datasets",
    )
    results_dir: Path = Field(
        default=Path("./data/results"),
        description="Directory for evaluation results",
    )
    cache_dir: Path = Field(
        default=Path("./data/cache"),
        description="Directory for cached responses",
    )
    local_models_dir: Path = Field(
        default=Path("./data/models"),
        description="Directory for local models",
    )

    # ===========================================
    # API Configuration
    # ===========================================
    api_host: str = Field(default="0.0.0.0", description="FastAPI server host")
    api_port: int = Field(default=8000, ge=1024, le=65535, description="FastAPI server port")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        description="CORS allowed origins",
    )
    api_key: str = Field(default="", description="Optional API key for authentication")
    max_request_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Max request body size in MB",
    )

    # ===========================================
    # Dashboard Configuration
    # ===========================================
    dashboard_host: str = Field(default="0.0.0.0", description="Streamlit server host")
    dashboard_port: int = Field(
        default=8501,
        ge=1024,
        le=65535,
        description="Streamlit server port",
    )

    # ===========================================
    # Monitoring & Observability
    # ===========================================
    metrics_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Prometheus metrics server port",
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics collection",
    )
    metrics_endpoint: str = Field(
        default="/metrics",
        description="Prometheus metrics endpoint path",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity level",
    )
    log_file: str = Field(
        default="",
        description="Log file path (empty for stdout only)",
    )

    # ===========================================
    # Database Configuration
    # ===========================================
    database_url: str = Field(
        default="",
        description="PostgreSQL connection string (optional)",
    )
    sqlite_db_path: str = Field(
        default="./data/results.db",
        description="SQLite database path for fallback",
    )

    # ===========================================
    # Rate Limiting
    # ===========================================
    rate_limit_rpm: int = Field(
        default=60,
        ge=1,
        description="Requests per minute per API key",
    )
    max_concurrent_requests: int = Field(
        default=5,
        ge=1,
        description="Maximum concurrent requests per user",
    )

    # ===========================================
    # Feature Flags
    # ===========================================
    enable_cache: bool = Field(
        default=True,
        description="Enable response caching",
    )
    cache_ttl: int = Field(
        default=3600,
        ge=0,
        description="Cache TTL in seconds",
    )
    enable_async: bool = Field(
        default=True,
        description="Enable async evaluation",
    )
    enable_debug_logging: bool = Field(
        default=False,
        description="Enable detailed debug logging",
    )

    # ===========================================
    # Security
    # ===========================================
    secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key for session encryption",
    )
    enable_https: bool = Field(
        default=False,
        description="Enable HTTPS for production",
    )
    ssl_cert_path: str = Field(
        default="",
        description="Path to SSL certificate",
    )
    ssl_key_path: str = Field(
        default="",
        description="Path to SSL private key",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("data_dir", "datasets_dir", "results_dir", "cache_dir", "local_models_dir")
    @classmethod
    def ensure_path_exists(cls, v: Path) -> Path:
        """Ensure directory paths are created if they don't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def is_prod(self) -> bool:
        """Check if running in production mode."""
        return self.enable_https

    @property
    def use_database(self) -> bool:
        """Check if PostgreSQL database is configured."""
        return bool(self.database_url)

    @property
    def judge_model(self) -> str:
        """Get the judge model identifier."""
        return f"{self.judge_model_provider}:{self.judge_model_name}"

    def get_model_config(self, provider: str) -> dict:
        """
        Get configuration for a specific model provider.

        Args:
            provider: Model provider name (openai, anthropic, cohere)

        Returns:
            Dictionary with provider-specific configuration
        """
        configs = {
            "openai": {
                "api_key": self.openai_api_key,
                "default_model": self.default_openai_model,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
            },
            "anthropic": {
                "api_key": self.anthropic_api_key,
                "default_model": self.default_anthropic_model,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
            },
            "cohere": {
                "api_key": self.cohere_api_key,
                "default_model": self.default_cohere_model,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
            },
        }
        return configs.get(provider.lower(), {})


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance.

    Returns:
        Settings: Application settings singleton
    """
    return Settings()


# Convenience export
settings = get_settings()
