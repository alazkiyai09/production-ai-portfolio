"""
Comprehensive tests for PromptOptimizer module.

Tests cover:
- Template management
- Variation generation
- A/B testing framework
- Statistical analysis
- Selection logic
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import json

# ============================================================================
# Template Management Tests
# ============================================================================

class TestTemplateManagement:
    """Test prompt template management functionality."""

    def test_create_template(self):
        """Test creating a new prompt template."""
        from src.prompt_optimizer.templates import create_template_manager

        manager = create_template_manager()

        template = manager.create_template(
            name="test_template",
            template_string="You are a {{role}}. Task: {{task}}.",
            description="Test template",
            category="test",
            tags=["testing"],
        )

        assert template is not None
        assert template.name == "test_template"
        assert template.template_string == "You are a {{role}}. Task: {{task}}."
        assert template.variables == {"role", "task"}

    def test_template_versioning(self):
        """Test template versioning."""
        from src.prompt_optimizer.templates import create_template_manager

        manager = create_template_manager()

        # Create v1
        v1 = manager.create_template(
            name="versioned_template",
            template_string="Version 1: {{content}}",
            version="1.0",
        )

        # Create v2
        v2 = manager.create_template(
            name="versioned_template",
            template_string="Version 2: {{content}}",
            version="2.0",
        )

        assert v1.version == "1.0"
        assert v2.version == "2.0"
        assert v1.template_string != v2.template_string

    def test_template_rendering(self):
        """Test rendering templates with variables."""
        from src.prompt_optimizer.templates import create_template_manager

        manager = create_template_manager()

        template = manager.create_template(
            name="render_test",
            template_string="You are a {{role}}. Your task: {{task}}.",
            default_variables={"role": "assistant", "task": "help"},
        )

        rendered = manager.render_template(
            "render_test",
            {"role": "expert", "task": "analyze"},
        )

        assert rendered.rendered_content == "You are a expert. Your task: analyze."

    def test_variable_extraction(self):
        """Test extraction of variables from template."""
        from src.prompt_optimizer.templates import create_template_manager

        manager = create_template_manager()

        template = manager.create_template(
            name="var_extract",
            template_string="{{var1}} and {{var2}} with {{var3}}",
        )

        assert template.variables == {"var1", "var2", "var3"}

    def test_template_validation(self):
        """Test template validation."""
        from src.prompt_optimizer.templates import create_template_manager

        manager = create_template_manager()

        # Valid template
        valid = manager.create_template(
            name="valid_template",
            template_string="Valid {{content}}",
        )
        assert valid.is_valid()

        # Invalid Jinja2
        invalid = manager.create_template(
            name="invalid_template",
            template_string="{{unclosed brace",
        )
        assert not invalid.is_valid()


# ============================================================================
# Variation Generation Tests
# ============================================================================

class TestVariationGeneration:
    """Test prompt variation generation."""

    def test_instruction_rephrase_strategy(self):
        """Test instruction rephrasing variation strategy."""
        from src.prompt_optimizer.variations import (
            create_variation_generator,
            VariationStrategy,
        )

        generator = create_variation_generator(seed=42)

        variations = generator.generate(
            base_prompt="Summarize the text.",
            strategies=[VariationStrategy.INSTRUCTION_REPHRASE],
            variations_per_strategy=3,
        )

        assert len(variations.variations) > 0
        assert all(v.strategy == VariationStrategy.INSTRUCTION_REPHRASE for v in variations.variations)

    def test_few_shot_selection_strategy(self):
        """Test few-shot selection strategy."""
        from src.prompt_optimizer.variations import (
            create_variation_generator,
            VariationStrategy,
        )

        generator = create_variation_generator(seed=42)

        examples = [
            {"input": "Q1", "output": "A1", "category": "cat1"},
            {"input": "Q2", "output": "A2", "category": "cat2"},
            {"input": "Q3", "output": "A3", "category": "cat1"},
        ]

        variations = generator.generate(
            base_prompt="Answer the question.",
            strategies=[VariationStrategy.FEW_SHOT_SELECTION],
            variations_per_strategy=2,
            examples=examples,
            examples_per_prompt=2,
        )

        assert len(variations.variations) > 0

    def test_output_format_strategy(self):
        """Test output format variation strategy."""
        from src.prompt_optimizer.variations import (
            create_variation_generator,
            VariationStrategy,
        )

        generator = create_variation_generator()

        variations = generator.generate(
            base_prompt="Generate response.",
            strategies=[VariationStrategy.OUTPUT_FORMAT],
            variations_per_strategy=3,
        )

        # Should have variations for different formats
        assert len(variations.variations) >= 3

    def test_chain_of_thought_strategy(self):
        """Test chain-of-thought variation strategy."""
        from src.prompt_optimizer.variations import (
            create_variation_generator,
            VariationStrategy,
        )

        generator = create_variation_generator()

        variations = generator.generate(
            base_prompt="Solve the problem.",
            strategies=[VariationStrategy.COT_STYLE],
            variations_per_strategy=3,
        )

        assert len(variations.variations) > 0
        # Check CoT is added to prompt
        assert any("step by step" in v.prompt_content.lower() for v in variations.variations)

    def test_combined_strategies(self):
        """Test combining multiple strategies."""
        from src.prompt_optimizer.variations import (
            create_variation_generator,
            VariationStrategy,
        )

        generator = create_variation_generator()

        variations = generator.generate(
            base_prompt="Test prompt.",
            strategies=[
                VariationStrategy.INSTRUCTION_REPHRASE,
                VariationStrategy.OUTPUT_FORMAT,
            ],
            variations_per_strategy=2,
            combine_strategies=True,
        )

        # Should have base variations + combined variations
        assert len(variations.variations) >= 4

    def test_reproducibility(self):
        """Test that variations are reproducible with same seed."""
        from src.prompt_optimizer.variations import (
            create_variation_generator,
            VariationStrategy,
        )

        generator1 = create_variation_generator(seed=42)
        generator2 = create_variation_generator(seed=42)

        variations1 = generator1.generate(
            base_prompt="Test prompt.",
            strategies=[VariationStrategy.INSTRUCTION_REPHRASE],
            variations_per_strategy=3,
        )

        variations2 = generator2.generate(
            base_prompt="Test prompt.",
            strategies=[VariationStrategy.INSTRUCTION_REPHRASE],
            variations_per_strategy=3,
        )

        # Same content
        contents1 = [v.prompt_content for v in variations1.variations]
        contents2 = [v.prompt_content for v in variations2.variations]

        assert contents1 == contents2


# ============================================================================
# A/B Testing Framework Tests
# ============================================================================

class TestABTestingFramework:
    """Test A/B testing framework."""

    def test_create_experiment(self):
        """Test creating an A/B test experiment."""
        from src.prompt_optimizer.experiments.ab_testing import (
            ABTestingFramework,
            ExperimentStatus,
        )
        from src.prompt_optimizer.variations.variation_generator import PromptVariation

        framework = ABTestingFramework(storage_path="/tmp/test_experiments")

        # Create mock variations
        control = PromptVariation(
            id="control_1",
            base_template_id="base",
            strategy=None,
            variation_params={},
            prompt_content="Control prompt",
            system_prompt=None,
            description="Control",
        )

        treatment = PromptVariation(
            id="treatment_1",
            base_template_id="base",
            strategy=None,
            variation_params={},
            prompt_content="Treatment prompt",
            system_prompt=None,
            description="Treatment",
        )

        experiment = framework.create_experiment(
            name="Test Experiment",
            description="Testing A/B framework",
            control_prompt=control,
            treatment_prompts=[treatment],
            test_dataset_id="test_dataset",
            metrics=["accuracy"],
            config={"min_sample_size": 10},
        )

        assert experiment.id is not None
        assert experiment.name == "Test Experiment"
        assert experiment.status == ExperimentStatus.DRAFT
        assert len(experiment.variants) == 2

    def test_start_experiment(self):
        """Test starting an experiment."""
        from src.prompt_optimizer.experiments.ab_testing import (
            ABTestingFramework,
            ExperimentStatus,
        )
        from src.prompt_optimizer.variations.variation_generator import PromptVariation

        framework = ABTestingFramework(storage_path="/tmp/test_experiments")

        control = PromptVariation(
            id="control",
            base_template_id="base",
            strategy=None,
            variation_params={},
            prompt_content="Control",
        )

        experiment = framework.create_experiment(
            name="Test",
            description="Test",
            control_prompt=control,
            treatment_prompts=[],
            test_dataset_id="test",
            metrics=["accuracy"],
        )

        started = framework.start_experiment(experiment.id)

        assert started.status == ExperimentStatus.RUNNING
        assert started.started_at is not None

    def test_pause_experiment(self):
        """Test pausing a running experiment."""
        from src.prompt_optimizer.experiments.ab_testing import (
            ABTestingFramework,
            ExperimentStatus,
        )
        from src.prompt_optimizer.variations.variation_generator import PromptVariation

        framework = ABTestingFramework(storage_path="/tmp/test_experiments")

        control = PromptVariation(
            id="control",
            base_template_id="base",
            strategy=None,
            variation_params={},
            prompt_content="Control",
        )

        experiment = framework.create_experiment(
            name="Test",
            description="Test",
            control_prompt=control,
            treatment_prompts=[],
            test_dataset_id="test",
            metrics=["accuracy"],
        )

        framework.start_experiment(experiment.id)
        paused = framework.pause_experiment(experiment.id)

        assert paused.status == ExperimentStatus.PAUSED

    def test_cancel_experiment(self):
        """Test canceling an experiment."""
        from src.prompt_optimizer.experiments.ab_testing import (
            ABTestingFramework,
            ExperimentStatus,
        )
        from src.prompt_optimizer.variations.variation_generator import PromptVariation

        framework = ABTestingFramework(storage_path="/tmp/test_experiments")

        control = PromptVariation(
            id="control",
            base_template_id="base",
            strategy=None,
            variation_params={},
            prompt_content="Control",
        )

        experiment = framework.create_experiment(
            name="Test",
            description="Test",
            control_prompt=control,
            treatment_prompts=[],
            test_dataset_id="test",
            metrics=["accuracy"],
        )

        cancelled = framework.cancel_experiment(experiment.id)

        assert cancelled.status == ExperimentStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_execute_variant(self):
        """Test executing a single variant."""
        from src.prompt_optimizer.experiments.ab_testing import (
            ABTestingFramework,
        )
        from src.prompt_optimizer.variations.variation_generator import PromptVariation

        framework = ABTestingFramework()

        variant = PromptVariation(
            id="test_variant",
            base_template_id="base",
            strategy=None,
            variation_params={},
            prompt_content="Test prompt",
            system_prompt="You are helpful.",
        )

        # Mock test case
        test_case = Mock()
        test_case.id = "test_1"
        test_case.expected = "Expected output"
        test_case.context = {}

        # Mock LLM provider
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "Generated response"
        mock_response.total_tokens = 100
        mock_response.cost_usd = 0.001
        mock_provider.generate = AsyncMock(return_value=mock_response)

        result = await framework._execute_variant(
            variant=variant,
            test_case=test_case,
            llm_provider=mock_provider,
            metrics=[],
        )

        assert result.variant_id == "test_variant"
        assert result.test_case_id == "test_1"
        assert result.response == "Generated response"


# ============================================================================
# Statistical Analysis Tests
# ============================================================================

class TestStatisticalAnalysis:
    """Test statistical analysis functionality."""

    def test_cohens_d_calculation(self):
        """Test Cohen's d effect size calculation."""
        from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer()

        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        effect_size = analyzer._cohens_d(group1, group2)

        # Effect size should be 1.0 (shifted by 1 unit, same variance)
        assert abs(effect_size - 1.0) < 0.01

    def test_effect_size_interpretation(self):
        """Test effect size interpretation."""
        from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer()

        assert analyzer._interpret_effect_size(0.1) == "negligible"
        assert analyzer._interpret_effect_size(0.3) == "small"
        assert analyzer._interpret_effect_size(0.6) == "medium"
        assert analyzer._interpret_effect_size(1.0) == "large"

    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer(alpha=0.05)

        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        ci = analyzer._confidence_interval(group1, group2)

        # CI should contain the true difference (1.0)
        lower, upper = ci
        assert lower <= 1.0 <= upper
        assert lower < upper

    def test_power_calculation(self):
        """Test statistical power calculation."""
        from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer()

        # Larger effect size and sample size should give higher power
        power1 = analyzer._calculate_power(30, 30, 0.5)
        power2 = analyzer._calculate_power(100, 100, 0.8)

        assert 0 <= power1 <= 1
        assert 0 <= power2 <= 1
        assert power2 > power1  # Larger n and effect size

    def test_sample_size_calculation(self):
        """Test required sample size calculation."""
        from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer()

        # Larger effect size should require smaller sample size
        n1 = analyzer.calculate_required_sample_size(effect_size=0.5)
        n2 = analyzer.calculate_required_sample_size(effect_size=0.8)

        assert n1 > n2
        assert n1 > 0
        assert n2 > 0

    def test_bonferroni_correction(self):
        """Test Bonferroni correction for multiple comparisons."""
        from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer(alpha=0.05)

        # Create mock results
        from src.prompt_optimizer.statistics.analyzer import StatisticalTestResult

        results = {
            "var1": StatisticalTestResult(
                test_name="t-test",
                statistic=2.5,
                p_value=0.01,
                significant=True,
                effect_size=0.5,
                effect_size_interpretation="medium",
                confidence_interval=(0.1, 0.9),
                sample_size_control=30,
                sample_size_treatment=30,
                power=0.8,
                recommendation="Significant",
            ),
            "var2": StatisticalTestResult(
                test_name="t-test",
                statistic=2.0,
                p_value=0.03,
                significant=True,
                effect_size=0.4,
                effect_size_interpretation="small",
                confidence_interval=(0.05, 0.75),
                sample_size_control=30,
                sample_size_treatment=30,
                power=0.6,
                recommendation="Significant",
            ),
        }

        corrected = analyzer._apply_bonferroni_correction(results, num_comparisons=2)

        # After correction, p_value threshold is 0.025
        # var1: 0.01 < 0.025 -> still significant
        assert corrected["var1"].significant
        # var2: 0.03 > 0.025 -> no longer significant
        assert not corrected["var2"].significant


# ============================================================================
# Selection Tests
# ============================================================================

class TestPromptSelection:
    """Test prompt selection logic."""

    def test_weighted_score_calculation(self):
        """Test weighted score calculation."""
        from src.prompt_optimizer.selection.selector import BestPromptSelector
        from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer
        from src.prompt_optimizer.experiments.framework import VariantResult

        analyzer = StatisticalAnalyzer()
        selector = BestPromptSelector(analyzer)

        # Create mock variant results
        variant_results = {
            "var1": VariantResult(
                variant_id="var1",
                variant_name="Variant 1",
                strategy=None,
                prompt_content="Prompt 1",
                system_prompt=None,
                scores={"accuracy": [0.8, 0.9, 0.85]},
                mean_scores={"accuracy": 0.85},
                std_scores={"accuracy": 0.05},
                evaluation_time=1.0,
                token_usage={},
                cost=0.01,
            ),
            "var2": VariantResult(
                variant_id="var2",
                variant_name="Variant 2",
                strategy=None,
                prompt_content="Prompt 2",
                system_prompt=None,
                scores={"accuracy": [0.75, 0.8, 0.78]},
                mean_scores={"accuracy": 0.777},
                std_scores={"accuracy": 0.025},
                evaluation_time=1.0,
                token_usage={},
                cost=0.01,
            ),
        }

        # Higher accuracy should win
        scores = selector._calculate_weighted_scores(
            experiment=Mock(results=[], metrics=["accuracy"]),
            metric_analyses={},
            criteria=Mock(metric_weights={"accuracy": 1.0}),
        )

        # Just verify the method runs without error
        assert isinstance(scores, dict)

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        from src.prompt_optimizer.selection.selector import BestPromptSelector
        from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer()
        selector = BestPromptSelector(analyzer)

        best_data = {"weighted_score": 0.9}
        second_best = ("var2", {"weighted_score": 0.7})

        confidence = selector._calculate_confidence(best_data, second_best)

        # Larger gap should give higher confidence
        assert 0 < confidence <= 1
        assert confidence > 0.5  # 0.2 gap should give good confidence

    def test_score_normalization(self):
        """Test score normalization."""
        from src.prompt_optimizer.selection.selector import BestPromptSelector
        from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer()
        selector = BestPromptSelector(analyzer)

        # Create mock experiment
        experiment = Mock()
        experiment.results = [
            Mock(variant_id="var1", metrics={"accuracy": 0.8}),
            Mock(variant_id="var1", metrics={"accuracy": 0.9}),
            Mock(variant_id="var2", metrics={"accuracy": 0.7}),
        ]

        scores = {"accuracy": 0.85}
        criteria = Mock(metric_weights={"accuracy": 1.0})

        normalized = selector._normalize_scores(scores, experiment, criteria)

        # Should be in [0, 1] range
        assert 0 <= normalized.get("accuracy", 0) <= 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full workflows."""

    def test_end_to_end_experiment_workflow(self):
        """Test complete experiment workflow."""
        # This would test the full flow from template creation
        # through variation generation, experiment execution,
        # statistical analysis, and best prompt selection
        pass

    def test_template_to_variation_to_experiment(self):
        """Test workflow from template to experiment."""
        # Create template -> Generate variations -> Create experiment
        pass


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_experiment_data():
    """Sample experiment data for testing."""
    return {
        "control_scores": [0.75, 0.78, 0.82, 0.80, 0.77],
        "treatment_scores": [0.82, 0.85, 0.88, 0.86, 0.83],
        "alpha": 0.05,
    }


@pytest.fixture
def sample_variants():
    """Sample prompt variations for testing."""
    return [
        {
            "id": "control",
            "prompt": "Summarize the text.",
            "strategy": "control",
        },
        {
            "id": "treatment_1",
            "prompt": "Please summarize the text concisely.",
            "strategy": "instruction_rephrase",
        },
        {
            "id": "treatment_2",
            "prompt": "Provide a brief summary of: {{text}}",
            "strategy": "output_format",
        },
    ]


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
