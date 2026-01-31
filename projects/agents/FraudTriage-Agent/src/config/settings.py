"""
Application settings using Pydantic Settings.

Loads configuration from environment variables and .env file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # API Configuration
    # -------------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=False, description="Enable auto-reload")
    log_level: str = Field(default="info", description="Log level")

    # -------------------------------------------------------------------------
    # GLM-4.7 Configuration (Primary LLM)
    # -------------------------------------------------------------------------
    glm_api_key: str = Field(default="", description="GLM API key")
    glm_base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4",
        description="GLM API base URL"
    )
    glm_model: str = Field(default="glm-4-plus", description="GLM model name")
    glm_temperature: float = Field(default=0.1, description="GLM temperature")
    glm_max_tokens: int = Field(default=4096, description="GLM max tokens")

    # -------------------------------------------------------------------------
    # OpenAI Configuration (Fallback/Optional)
    # -------------------------------------------------------------------------
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL"
    )
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    openai_temperature: float = Field(default=0.1, description="OpenAI temperature")
    openai_max_tokens: int = Field(default=4096, description="OpenAI max tokens")

    # -------------------------------------------------------------------------
    # LangSmith Configuration (Optional - for tracing)
    # -------------------------------------------------------------------------
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangSmith tracing")
    langchain_api_key: str = Field(default="", description="LangSmith API key")
    langchain_project: str = Field(default="fraud-triage-agent", description="LangSmith project name")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith endpoint"
    )

    # -------------------------------------------------------------------------
    # Agent Configuration
    # -------------------------------------------------------------------------
    agent_type: str = Field(default="langgraph", description="Agent type")
    max_iterations: int = Field(default=10, description="Maximum agent iterations")
    timeout_seconds: int = Field(default=300, description="Agent timeout in seconds")

    # Risk thresholds
    human_in_loop_enabled: bool = Field(default=True, description="Enable human-in-the-loop")
    high_risk_threshold: int = Field(default=70, description="High risk threshold (0-100)")
    medium_risk_threshold: int = Field(default=40, description="Medium risk threshold (0-100)")

    # -------------------------------------------------------------------------
    # Database Configuration
    # -------------------------------------------------------------------------
    database_url: str = Field(default="sqlite:///./data/fraud_triage.db", description="Database URL")

    # Redis (optional - for caching)
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")

    # -------------------------------------------------------------------------
    # External API Configuration
    # -------------------------------------------------------------------------
    transaction_service_url: str = Field(
        default="http://localhost:8001/api/transactions",
        description="Transaction service URL"
    )
    customer_service_url: str = Field(
        default="http://localhost:8002/api/customers",
        description="Customer service URL"
    )
    device_fingerprint_url: str = Field(
        default="http://localhost:8003/api/devices",
        description="Device fingerprint service URL"
    )
    case_management_url: str = Field(
        default="http://localhost:8004/api/cases",
        description="Case management URL"
    )

    # -------------------------------------------------------------------------
    # Security
    # -------------------------------------------------------------------------
    secret_key: str = Field(default="change-me-in-production", description="Secret key for signing")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="CORS allowed origins"
    )

    # -------------------------------------------------------------------------
    # Feature Flags
    # -------------------------------------------------------------------------
    enable_async_tools: bool = Field(default=True, description="Enable async tools")
    enable_cache: bool = Field(default=True, description="Enable caching")
    enable_audit_logging: bool = Field(default=True, description="Enable audit logging")
    mock_external_apis: bool = Field(default=True, description="Use mock external APIs")

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    @property
    def base_dir(self) -> Path:
        """Base directory of the project."""
        return Path(__file__).parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Data directory."""
        return self.base_dir / "data"

    @property
    def sample_alerts_dir(self) -> Path:
        """Sample alerts directory."""
        return self.data_dir / "sample_alerts"

    @property
    def mock_data_dir(self) -> Path:
        """Mock data directory."""
        return self.data_dir / "mock_data"

    @property
    def log_dir(self) -> Path:
        """Log directory."""
        return self.base_dir / "logs"

    def get_allowed_origins_list(self) -> list[str]:
        """Get allowed origins as a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function is cached to ensure settings are loaded only once.
    """
    return Settings()


# Create a global settings instance
settings = get_settings()
