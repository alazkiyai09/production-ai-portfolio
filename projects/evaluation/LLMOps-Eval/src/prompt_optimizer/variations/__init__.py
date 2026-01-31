"""
Prompt variation strategies and generators.

This module provides systematic prompt variation capabilities for A/B testing
with reproducibility and lineage tracking.
"""

from src.prompt_optimizer.variations.variation_generator import (
    # Enums
    VariationStrategy,
    # Data classes
    PromptVariation,
    VariationSet,
    # Strategies
    BaseVariationStrategy,
    InstructionRephraseStrategy,
    FewShotSelectionStrategy,
    FewShotOrderStrategy,
    OutputFormatStrategy,
    ChainOfThoughtStrategy,
    SystemPromptStrategy,
    EmphasisStrategy,
    VerbosityStrategy,
    TemperatureSweepStrategy,
    TopPSweepStrategy,
    ContextPositionStrategy,
    # Main class
    VariationGenerator,
    create_variation_generator,
)

__all__ = [
    # Enums
    "VariationStrategy",
    # Data classes
    "PromptVariation",
    "VariationSet",
    # Base
    "BaseVariationStrategy",
    # Strategies
    "InstructionRephraseStrategy",
    "FewShotSelectionStrategy",
    "FewShotOrderStrategy",
    "OutputFormatStrategy",
    "ChainOfThoughtStrategy",
    "SystemPromptStrategy",
    "EmphasisStrategy",
    "VerbosityStrategy",
    "TemperatureSweepStrategy",
    "TopPSweepStrategy",
    "ContextPositionStrategy",
    # Main
    "VariationGenerator",
    "create_variation_generator",
]
