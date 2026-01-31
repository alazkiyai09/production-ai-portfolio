"""
Configuration models for AIGuard middleware.

Uses Pydantic for type-safe configuration management.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GuardrailsConfig(BaseSettings):
    """
    Configuration for AIGuard middleware.

    All settings can be overridden via environment variables or .env file.
    """

    # Detection Feature Flags
    enable_injection_detection: bool = Field(
        default=True,
        description="Enable prompt injection and jailbreak detection",
    )
    enable_pii_detection: bool = Field(
        default=True,
        description="Enable PII detection in inputs/outputs",
    )
    enable_pii_redaction: bool = Field(
        default=True,
        description="Enable automatic PII redaction",
    )
    enable_output_filtering: bool = Field(
        default=True,
        description="Enable output filtering for data leakage and toxicity",
    )
    enable_encoding_detection: bool = Field(
        default=True,
        description="Enable encoding attack detection",
    )

    # Detection Thresholds
    injection_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for injection detection",
    )
    jailbreak_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for jailbreak detection",
    )
    pii_confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for PII detection",
    )
    toxicity_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Threshold for blocking toxic content",
    )

    # Response Behavior
    block_on_detection: bool = Field(
        default=True,
        description="Block requests when threats are detected (vs just logging)",
    )
    sanitize_on_detection: bool = Field(
        default=False,
        description="Sanitize detected content instead of blocking",
    )
    return_details: bool = Field(
        default=True,
        description="Include detection details in error responses",
    )

    # Logging Configuration
    log_blocked_requests: bool = Field(
        default=True,
        description="Log all blocked requests",
    )
    log_all_requests: bool = Field(
        default=False,
        description="Log all requests (not just blocked)",
    )
    log_file_path: Optional[str] = Field(
        default="logs/aiguard.log",
        description="Path to log file",
    )

    # Model Configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for semantic detection",
    )
    spacy_model: str = Field(
        default="en_core_web_sm",
        description="spaCy model for NER",
    )
    device: str = Field(
        default="cpu",
        description="Device for model inference (cpu/cuda)",
    )
    cache_dir: Optional[str] = Field(
        default="./cache",
        description="Directory for model caching",
    )

    # PII Redaction Mode
    pii_redaction_mode: str = Field(
        default="full",
        description="PII redaction mode: full, partial, token, mask, hash",
    )

    # Rate Limiting
    enable_rate_limiting: bool = Field(
        default=False,
        description="Enable rate limiting",
    )
    rate_limit_requests: int = Field(
        default=100,
        ge=1,
        description="Max requests per time window",
    )
    rate_limit_window: int = Field(
        default=60,
        ge=1,
        description="Time window in seconds",
    )

    # Input/Output Limits
    max_prompt_length: int = Field(
        default=10000,
        ge=1,
        description="Maximum allowed prompt length in characters",
    )
    max_output_length: int = Field(
        default=50000,
        ge=1,
        description="Maximum allowed output length in characters",
    )
    truncate_long_inputs: bool = Field(
        default=False,
        description="Truncate inputs that exceed max length (vs reject)",
    )

    # API Keys
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for LLM-based detection (optional)",
    )

    # Blocked IPs (comma-separated)
    blocked_ips: str = Field(
        default="",
        description="Comma-separated list of blocked IP addresses",
    )

    # Allowed endpoints (regex pattern)
    allowed_endpoints: Optional[str] = Field(
        default=None,
        description="Regex pattern for allowed endpoints (empty = all)",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AIGUARD_",
        extra="allow",
    )

    def get_blocked_ips_list(self) -> list[str]:
        """Parse blocked IPs comma-separated string."""
        return [ip.strip() for ip in self.blocked_ips.split(",") if ip.strip()]


@lru_cache()
def get_config() -> GuardrailsConfig:
    """
    Get cached configuration instance.

    Returns:
        GuardrailsConfig singleton
    """
    return GuardrailsConfig()
