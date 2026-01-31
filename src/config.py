"""
Configuration management for AgenticFlow.

Uses Pydantic BaseSettings to load and validate configuration from
environment variables and .env files.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings have defaults and can be overridden via environment variables
    or a .env file in the project root.
    """

    # =============================================================================
    # API Keys
    # =============================================================================

    # OpenAI API Key (Required)
    openai_api_key: str = Field(
        ..., description="OpenAI API key for GPT models"
    )

    # Anthropic API Key (Optional - for Claude models)
    anthropic_api_key: Optional[str] = Field(
        None, description="Anthropic API key for Claude models"
    )

    # Tavily Search API Key (Required)
    tavily_api_key: str = Field(
        ..., description="Tavily API key for web search"
    )

    # =============================================================================
    # Model Configuration
    # =============================================================================

    # Default model to use for agents
    default_model: str = Field(
        default="gpt-4-turbo-preview",
        description="Default LLM model for agents"
    )

    # Fallback model if primary fails
    fallback_model: str = Field(
        default="gpt-3.5-turbo",
        description="Fallback model if primary model fails"
    )

    # Temperature for generation
    model_temperature: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Temperature for model generation (0.0-1.0)"
    )

    # =============================================================================
    # Workflow Configuration
    # =============================================================================

    # Maximum iterations per workflow
    max_iterations: int = Field(
        default=10, gt=0, le=100,
        description="Maximum number of iterations/steps per workflow"
    )

    # Timeout per step
    timeout_seconds: int = Field(
        default=300, gt=0, le=3600,
        description="Timeout in seconds for each agent step"
    )

    # Enable human-in-the-loop checkpoints
    enable_human_checkpoint: bool = Field(
        default=True,
        description="Enable human-in-the-loop checkpoints for review"
    )

    # =============================================================================
    # API Configuration
    # =============================================================================

    # API host
    api_host: str = Field(
        default="0.0.0.0",
        description="Host for FastAPI server"
    )

    # API port
    api_port: int = Field(
        default=8000, gt=0, lt=65536,
        description="Port for FastAPI server"
    )

    # Enable CORS
    cors_enabled: bool = Field(
        default=True,
        description="Enable CORS for API"
    )

    # CORS origins
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )

    # =============================================================================
    # Checkpoint Storage Configuration
    # =============================================================================

    # Storage backend type
    checkpoint_storage: Literal["memory", "sqlite", "postgresql"] = Field(
        default="memory",
        description="Checkpoint storage backend"
    )

    # SQLite checkpoint path
    sqlite_checkpoint_path: str = Field(
        default="./data/checkpoints.db",
        description="Path to SQLite checkpoint database"
    )

    # PostgreSQL connection string
    postgres_connection_string: Optional[str] = Field(
        None,
        description="PostgreSQL connection string for checkpoints"
    )

    # =============================================================================
    # Logging Configuration
    # =============================================================================

    # Log level
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )

    # Log file path
    log_file: Optional[str] = Field(
        None,
        description="Log file path (None for stdout only)"
    )

    # Enable JSON logging
    log_json: bool = Field(
        default=False,
        description="Use JSON format for logs"
    )

    # =============================================================================
    # Development Settings
    # =============================================================================

    # Debug mode
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )

    # Hot reload
    reload: bool = Field(
        default=True,
        description="Enable hot reload during development"
    )

    # Number of workers
    workers: int = Field(
        default=1, gt=0, le=16,
        description="Number of worker processes"
    )

    # =============================================================================
    # Pydantic Configuration
    # =============================================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # =============================================================================
    # Validators
    # =============================================================================

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("log_file")
    @classmethod
    def ensure_log_directory(cls, v: Optional[str]) -> Optional[str]:
        """Ensure log directory exists if log file is specified."""
        if v:
            log_path = Path(v)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        return v

    # =============================================================================
    # Computed Properties
    # =============================================================================

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug

    @property
    def api_url(self) -> str:
        """Get the full API URL."""
        return f"http://{self.api_host}:{self.api_port}"

    @property
    def data_directory(self) -> Path:
        """Get or create the data directory."""
        data_dir = Path("./data")
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once and reused
    across the application.

    Returns:
        Settings: Cached settings instance
    """
    return Settings()


# Export settings instance for easy importing
settings = get_settings()
