"""
Configuration management for PromptOptimizer.

This module provides configuration for prompt optimization experiments,
statistical testing parameters, and integration with LLMOps-Eval.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, List
import logging

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.config import settings as llmops_settings

logger = logging.getLogger(__name__)


class OptimizerSettings(BaseSettings):
    """
    Configuration settings for PromptOptimizer.

    Attributes:
        # Statistical Testing Parameters
        significance_level: Alpha threshold for statistical tests
        minimum_sample_size: Minimum samples per variant for valid tests
        power_analysis: Target statistical power (1-beta)
        effect_size_threshold: Minimum effect size to consider meaningful
        multiple_testing_correction: Method for correcting multiple comparisons

        # Experiment Parameters
        max_concurrent_experiments: Maximum parallel experiment runs
        experiment_timeout_seconds: Timeout per experiment
        auto_save_results: Automatically save experiment results
        default_iterations: Default number of optimization iterations

        # Template Management
        templates_dir: Directory for prompt templates
        template_cache_ttl: Cache duration for parsed templates

        # History & Tracking
        history_db_path: SQLite database for experiment history
        max_history_entries: Maximum history entries to keep per prompt
        enable_audit_log: Enable detailed audit logging

        # Selection Criteria
        selection_metric: Primary metric for selecting best prompt
        require_statistical_significance: Require significant differences
        min_confidence_score: Minimum confidence score for selection

        # Variation Generation
        max_variants_per_template: Maximum variations to generate
        variation_strategies: List of variation strategies to use
        enable_temperature_sweep: Enable temperature parameter sweeps
        temperature_range: Temperature range for sweeps (min, max, steps)

        # Integration with LLMOps-Eval
        eval_output_dir: Output directory from LLMOps-Eval
        default_metrics: Default evaluation metrics
        default_dataset: Default dataset for testing
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ========================================================================
    # Statistical Testing Parameters
    # ========================================================================

    significance_level: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description="Alpha threshold for statistical significance testing",
    )

    minimum_sample_size: int = Field(
        default=30,
        ge=5,
        le=1000,
        description="Minimum sample size per variant for valid statistical tests",
    )

    power_analysis_target: float = Field(
        default=0.8,
        ge=0.5,
        le=0.99,
        description="Target statistical power (1-beta)",
    )

    effect_size_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Minimum Cohen's d effect size to consider meaningful",
    )

    multiple_testing_correction: str = Field(
        default="bonferroni",
        description="Method for correcting multiple comparisons (bonferroni, holm, none)",
    )

    # ========================================================================
    # Experiment Parameters
    # ========================================================================

    max_concurrent_experiments: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum parallel experiment runs",
    )

    experiment_timeout_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Timeout per experiment in seconds",
    )

    auto_save_results: bool = Field(
        default=True,
        description="Automatically save experiment results",
    )

    default_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of optimization iterations",
    )

    # ========================================================================
    # Template Management
    # ========================================================================

    templates_dir: Path = Field(
        default=Path("./data/prompt_templates"),
        description="Directory for prompt template files",
    )

    template_cache_ttl: int = Field(
        default=3600,
        ge=0,
        description="Template cache TTL in seconds",
    )

    # ========================================================================
    # History & Tracking
    # ========================================================================

    history_db_path: Path = Field(
        default=Path("./data/prompt_history/experiments.db"),
        description="SQLite database for experiment history",
    )

    max_history_entries: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum history entries to keep per prompt",
    )

    enable_audit_log: bool = Field(
        default=True,
        description="Enable detailed audit logging",
    )

    audit_log_path: Optional[Path] = Field(
        default=Path("./data/prompt_history/audit.log"),
        description="Path to audit log file",
    )

    # ========================================================================
    # Selection Criteria
    # ========================================================================

    selection_metric: str = Field(
        default="overall_score",
        description="Primary metric for selecting best prompt variant",
    )

    require_statistical_significance: bool = Field(
        default=True,
        description="Require statistical significance for winner selection",
    )

    min_confidence_score: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Minimum confidence score for automated selection",
    )

    # ========================================================================
    # Variation Generation
    # ========================================================================

    max_variants_per_template: int = Field(
        default=10,
        ge=2,
        le=50,
        description="Maximum number of variations to generate per template",
    )

    variation_strategies: List[str] = Field(
        default=[
            "instruction_phrasing",
            "example_ordering",
            "output_format",
            "few_shot_selection",
            "system_prompt",
            "chain_of_thought",
        ],
        description="List of variation strategies to employ",
    )

    enable_temperature_sweep: bool = Field(
        default=True,
        description="Enable temperature parameter sweeps",
    )

    temperature_sweep_min: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum temperature for sweep",
    )

    temperature_sweep_max: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Maximum temperature for sweep",
    )

    temperature_sweep_steps: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of temperature steps",
    )

    # ========================================================================
    # Integration with LLMOps-Eval
    # ========================================================================

    eval_output_dir: Path = Field(
        default=llmops_settings.results_dir,
        description="Output directory from LLMOps-Eval",
    )

    default_metrics: List[str] = Field(
        default=["exact_match", "semantic_similarity", "latency"],
        description="Default evaluation metrics to use",
    )

    default_dataset: str = Field(
        default="qa_evaluation",
        description="Default dataset to use for testing",
    )

    # ========================================================================
    # Advanced Settings
    # ========================================================================

    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging",
    )

    parallel_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of parallel workers for analysis",
    )

    @field_validator("templates_dir", "history_db_path")
    @classmethod
    def ensure_paths_exist(cls, v: Path) -> Path:
        """Ensure directories exist, create if needed."""
        v.parent.mkdir(parents=True, exist_ok=True)
        v.parent.parent.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("multiple_testing_correction")
    @classmethod
    def validate_correction_method(cls, v: str) -> str:
        """Validate multiple testing correction method."""
        valid_methods = ["bonferroni", "holm", "sidak", "none"]
        v_lower = v.lower()
        if v_lower not in valid_methods:
            raise ValueError(
                f"Invalid correction method: {v}. "
                f"Must be one of {valid_methods}"
            )
        return v_lower

    @property
    def confidence_level(self) -> float:
        """Calculate confidence level from significance level."""
        return 1.0 - self.significance_level

    @property
    def critical_value(self) -> float:
        """
        Get critical value for two-tailed test at current alpha.

        Returns:
            Critical value for significance test
        """
        from scipy import stats

        # For two-tailed test
        return stats.norm.ppf(1 - self.significance_level / 2)

    @property
    def required_sample_size(self) -> int:
        """
        Calculate required sample size for target power and effect size.

        Uses Cohen's d for effect size.

        Returns:
            Required sample size per group
        """
        from scipy import stats

        # Approximate sample size calculation
        # n = 2 * (z_alpha + z_beta)^2 * d^2
        alpha = self.significance_level
        beta = 1 - self.power_analysis_target
        effect_size = self.effect_size_threshold

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(self.power_analysis_target)

        n = 2 * ((z_alpha + z_beta) ** 2) / (effect_size ** 2)
        return max(int(round(n)), self.minimum_sample_size)

    def get_correction_multiplier(self, num_comparisons: int) -> float:
        """
        Get correction multiplier for multiple comparisons.

        Args:
            num_comparisons: Number of comparisons/hypotheses tested

        Returns:
            Multiplier to apply to alpha threshold
        """
        if self.multiple_testing_correction == "bonferroni":
            return float(num_comparisons)
        elif self.multiple_testing_correction == "holm":
            # Holm-Bonferroni (handled in testing, return 1 here)
            return 1.0
        elif self.multiple_testing_correction == "sidak":
            return float(1 - (1 - self.significance_level) ** (1 / num_comparisons))
        else:
            return 1.0

    def get_temperature_steps(self) -> List[float]:
        """
        Generate temperature steps for sweeping.

        Returns:
            List of temperature values to test
        """
        if not self.enable_temperature_sweep:
            return [0.7]  # Default temperature

        start = self.temperature_sweep_min
        end = self.temperature_sweep_max
        steps = self.temperature_sweep_steps

        if steps == 1:
            return [start]

        step_size = (end - start) / (steps - 1)
        return [start + i * step_size for i in range(steps)]


@lru_cache
def get_optimizer_settings() -> OptimizerSettings:
    """
    Cached optimizer settings instance.

    Returns:
        OptimizerSettings: Configuration singleton
    """
    return OptimizerSettings()


# Convenience export
optimizer_settings = get_optimizer_settings()


# Export main classes and functions
__all__ = [
    "OptimizerSettings",
    "get_optimizer_settings",
    "optimizer_settings",
]
