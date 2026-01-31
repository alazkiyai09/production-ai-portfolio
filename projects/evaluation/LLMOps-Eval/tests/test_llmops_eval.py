"""
Test suite for LLMOps-Eval.

This module provides comprehensive tests for the LLMOps-Eval system including
metric calculations, provider interfaces, dataset loading, runner execution,
report generation, and API endpoints.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import json
import tempfile
import yaml

# Import from our modules
from src.models.llm_providers import (
    LLMResponse,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    create_provider,
)
from src.evaluation.metrics import (
    MetricResult,
    ExactMatchMetric,
    ContainsMetric,
    SemanticSimilarityMetric,
    LLMJudgeMetric,
    FormatComplianceMetric,
    LatencyMetric,
    CostMetric,
    evaluate_metrics,
    create_metric,
)
from src.datasets.dataset_manager import (
    TestCase,
    TestDataset,
    DatasetManager,
    load_sample_dataset,
)
from src.runners.eval_runner import (
    ModelConfig,
    EvaluationConfig,
    TestResult,
    EvaluationRunner,
    EvaluationResult,
)
from src.reporting.report_generator import (
    ReportGenerator,
    generate_report,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_test_case():
    """Create a sample test case."""
    return TestCase(
        id="test_001",
        prompt="What is the capital of France?",
        expected="Paris",
        category="factual",
        tags=["geography", "simple"],
        metrics=["exact_match"],
    )


@pytest.fixture
def sample_dataset(sample_test_case):
    """Create a sample dataset."""
    return TestDataset(
        name="test_dataset",
        version="1.0",
        description="Test dataset",
        test_cases=[sample_test_case],
        default_metrics=["exact_match"],
    )


@pytest.fixture
def sample_llm_response():
    """Create a sample LLM response."""
    return LLMResponse(
        content="Paris",
        model="gpt-4o-mini",
        provider="openai",
        input_tokens=5,
        output_tokens=2,
        total_tokens=7,
        latency_ms=1250.0,
        time_to_first_token_ms=150.0,
        cost_usd=0.00001,
        raw_response={},
        finish_reason="stop",
    )


@pytest.fixture
def sample_dataset_yaml():
    """Sample dataset YAML content."""
    return """
name: test_dataset
version: "1.0"
description: "Test dataset"
default_metrics:
  - exact_match
test_cases:
  - id: test_001
    prompt: "What is 2+2?"
    expected: "4"
    category: "math"
    tags: ["simple"]
"""


# ============================================================================
# Tests: Models
# ============================================================================

class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_create_response(self):
        """Test creating a response."""
        response = LLMResponse(
            content="Test response",
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            latency_ms=1000.0,
            cost_usd=0.001,
        )

        assert response.content == "Test response"
        assert response.model == "gpt-4o-mini"
        assert response.total_tokens == 30
        assert response.latency_ms == 1000.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        response = LLMResponse(
            content="Test",
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=10,
            output_tokens=10,
            total_tokens=20,
            latency_ms=500.0,
            cost_usd=0.0001,
            raw_response={},
        )

        data = response.to_dict()

        assert data["content"] == "Test"
        assert data["model"] == "gpt-4o-mini"
        assert data["total_tokens"] == 20
        assert data["latency_ms"] == 500.0


class TestCreateProvider:
    """Test provider factory function."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        provider = create_provider(
            provider="openai",
            model="gpt-4o-mini",
            api_key="test-key",
        )

        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o-mini"

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        provider = create_provider(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

        assert isinstance(provider, AnthropicProvider)
        assert provider.model == "claude-3-5-sonnet-20241022"

    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        provider = create_provider(
            provider="ollama",
            model="llama3.2:3b",
        )

        assert isinstance(provider, OllamaProvider)
        assert provider.model == "llama3.2:3b"

    def test_create_invalid_provider(self):
        """Test creating invalid provider raises error."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_provider(
                provider="invalid",
                model="test-model",
            )


# ============================================================================
# Tests: Metrics
# ============================================================================

class TestExactMatchMetric:
    """Test ExactMatchMetric."""

    def test_exact_match_pass(self):
        """Test exact match that passes."""
        metric = ExactMatchMetric(threshold=1.0)

        result = asyncio.run(metric.evaluate(
            response="Paris",
            expected="Paris",
        ))

        assert result.name == "exact_match"
        assert result.value == 1.0
        assert result.passed is True

    def test_exact_match_fail(self):
        """Test exact match that fails."""
        metric = ExactMatchMetric(threshold=1.0)

        result = asyncio.run(metric.evaluate(
            response="paris",
            expected="Paris",
        ))

        assert result.value == 0.0
        assert result.passed is False

    def test_exact_match_case_insensitive(self):
        """Test case insensitive matching."""
        metric = ExactMatchMetric(threshold=1.0, case_sensitive=False)

        result = asyncio.run(metric.evaluate(
            response="paris",
            expected="PARIS",
        ))

        assert result.passed is True

    def test_exact_match_ignore_whitespace(self):
        """Test whitespace ignoring."""
        metric = ExactMatchMetric(
            threshold=1.0,
            ignore_whitespace=True,
        )

        result = asyncio.run(metric.evaluate(
            response="Paris  France",
            expected="Paris France",
        ))

        assert result.passed is True


class TestContainsMetric:
    """Test ContainsMetric."""

    def test_contains_single_keyword(self):
        """Test checking for single keyword."""
        metric = ContainsMetric(threshold=0.5)

        result = asyncio.run(metric.evaluate(
            response="The capital of France is Paris.",
            expected="Paris",
            context={"keywords": ["Paris"]},
        ))

        assert result.passed is True

    def test_contains_multiple_keywords(self):
        """Test checking for multiple keywords."""
        metric = ContainsMetric(threshold=0.5, require_all=False)

        result = asyncio.run(metric.evaluate(
            response="Paris is in France.",
            expected="",
            context={"keywords": ["Paris", "London", "Berlin"]},
        ))

        # Should have 1/3 keywords
        assert result.value == 1.0 / 3.0


class TestFormatComplianceMetric:
    """Test FormatComplianceMetric."""

    def test_json_format_valid(self):
        """Test valid JSON format."""
        metric = FormatComplianceMetric(expected_format="json")

        result = asyncio.run(metric.evaluate(
            response='{"name": "John", "age": 30}',
            expected="",
        ))

        assert result.passed is True
        assert result.value == 1.0

    def test_json_format_invalid(self):
        """Test invalid JSON format."""
        metric = FormatComplianceMetric(expected_format="json")

        result = asyncio.run(metric.evaluate(
            response='{"name": "John", age: 30}',  # Invalid JSON
            expected="",
        ))

        assert result.passed is False

    def test_json_extraction(self):
        """Test JSON extraction from text."""
        metric = FormatComplianceMetric(expected_format="json")

        result = asyncio.run(metric.evaluate(
            response='Here is the JSON: {"name": "John"}',
            expected="",
        ))

        assert result.passed is True


class TestLatencyMetric:
    """Test LatencyMetric."""

    def test_latency_within_threshold(self):
        """Test latency within threshold."""
        metric = LatencyMetric(threshold_ms=5000)

        result = asyncio.run(metric.evaluate(
            response="",
            expected="",
            context={"latency_ms": 1000},
        ))

        assert result.passed is True
        assert result.value == 1.0

    def test_latency_exceeds_threshold(self):
        """Test latency exceeds threshold."""
        metric = LatencyMetric(threshold_ms=1000)

        result = asyncio.run(metric.evaluate(
            response="",
            expected="",
            context={"latency_ms": 5000},
        ))

        assert result.passed is False


class TestCostMetric:
    """Test CostMetric."""

    def test_cost_within_budget(self):
        """Test cost within budget."""
        metric = CostMetric(threshold_usd=0.01)

        result = asyncio.run(metric.evaluate(
            response="",
            expected="",
            context={"cost_usd": 0.005},
        ))

        assert result.passed is True

    def test_cost_exceeds_budget(self):
        """Test cost exceeds budget."""
        metric = CostMetric(threshold_usd=0.01)

        result = asyncio.run(metric.evaluate(
            response="",
            expected="",
            context={"cost_usd": 0.02},
        ))

        assert result.passed is False


class TestCreateMetric:
    """Test metric factory function."""

    def test_create_exact_match_metric(self):
        """Test creating exact match metric."""
        metric = create_metric("exact_match")

        assert isinstance(metric, ExactMatchMetric)

    def test_create_contains_metric(self):
        """Test creating contains metric."""
        metric = create_metric("contains")

        assert isinstance(metric, ContainsMetric)

    def test_create_invalid_metric(self):
        """Test creating invalid metric raises error."""
        with pytest.raises(ValueError, match="Unknown metric"):
            create_metric("invalid_metric")


# ============================================================================
# Tests: Datasets
# ============================================================================

class TestTestCase:
    """Test TestCase dataclass."""

    def test_create_test_case(self):
        """Test creating a test case."""
        test_case = TestCase(
            id="test_001",
            prompt="Test prompt",
            expected="Expected answer",
        )

        assert test_case.id == "test_001"
        assert test_case.prompt == "Test prompt"
        assert test_case.category == "general"  # Default
        assert test_case.enabled is True  # Default

    def test_get_effective_metrics(self):
        """Test getting effective metrics."""
        test_case = TestCase(
            id="test_001",
            prompt="Test",
            expected="Test",
            metrics=["custom_metric"],
        )

        metrics = test_case.get_effective_metrics(["default_metric"])

        assert metrics == ["custom_metric"]

    def test_fallback_to_default_metrics(self):
        """Test falling back to default metrics."""
        test_case = TestCase(
            id="test_001",
            prompt="Test",
            expected="Test",
            metrics=[],
        )

        metrics = test_case.get_effective_metrics(["default_metric"])

        assert metrics == ["default_metric"]


class TestTestDataset:
    """Test TestDataset dataclass."""

    def test_create_dataset(self, sample_test_case):
        """Test creating a dataset."""
        dataset = TestDataset(
            name="test",
            version="1.0",
            description="Test dataset",
            test_cases=[sample_test_case],
        )

        assert dataset.name == "test"
        assert dataset.test_case_count == 1
        assert dataset.enabled_test_case_count == 1

    def test_get_test_case(self, sample_dataset):
        """Test getting a test case by ID."""
        test_case = sample_dataset.get_test_case("test_001")

        assert test_case is not None
        assert test_case.id == "test_001"

    def test_get_nonexistent_test_case(self, sample_dataset):
        """Test getting nonexistent test case."""
        test_case = sample_dataset.get_test_case("nonexistent")

        assert test_case is None

    def test_categories_property(self, sample_test_case):
        """Test categories property."""
        dataset = TestDataset(
            name="test",
            version="1.0",
            description="Test",
            test_cases=[sample_test_case],
        )

        assert "factual" in dataset.categories

    def test_tags_property(self, sample_test_case):
        """Test tags property."""
        dataset = TestDataset(
            name="test",
            version="1.0",
            description="Test",
            test_cases=[sample_test_case],
        )

        assert "geography" in dataset.tags


# ============================================================================
# Tests: Dataset Manager
# ============================================================================

class TestDatasetManager:
    """Test DatasetManager."""

    def test_create_from_string(self, sample_dataset_yaml):
        """Test creating dataset from YAML string."""
        manager = DatasetManager()

        dataset = manager.create_from_string(
            sample_dataset_yaml,
            format="yaml",
        )

        assert dataset.name == "test_dataset"
        assert dataset.test_case_count == 1

    def test_filter_by_category(self, sample_dataset):
        """Test filtering by category."""
        manager = DatasetManager()

        filtered = manager.filter_by_category(
            sample_dataset,
            "factual",
        )

        assert filtered.test_case_count == 1

    def test_filter_by_tags(self, sample_test_case):
        """Test filtering by tags."""
        manager = DatasetManager()

        dataset = TestDataset(
            name="test",
            version="1.0",
            description="Test",
            test_cases=[sample_test_case],
        )

        filtered = manager.filter_by_tags(
            dataset,
            ["geography"],
        )

        assert filtered.test_case_count == 1

    def test_sample(self, sample_dataset):
        """Test sampling dataset."""
        manager = DatasetManager()

        # Sample with n larger than dataset
        sampled = manager.sample(sample_dataset, n=5)

        assert sampled.test_case_count == 1


# ============================================================================
# Tests: Runner
# ============================================================================

class TestModelConfig:
    """Test ModelConfig."""

    def test_create_config(self):
        """Test creating model config."""
        config = ModelConfig(
            provider="openai",
            model="gpt-4o-mini",
        )

        assert config.provider == "openai"
        assert config.model == "gpt-4o-mini"


class TestEvaluationConfig:
    """Test EvaluationConfig."""

    def test_create_config(self):
        """Test creating evaluation config."""
        models = [
            ModelConfig(provider="openai", model="gpt-4o-mini"),
        ]

        config = EvaluationConfig(
            name="test_eval",
            dataset="test_dataset",
            models=models,
            metrics=["exact_match"],
        )

        assert config.name == "test_eval"
        assert len(config.models) == 1
        assert config.parallel == 5  # Default


class TestTestResult:
    """Test TestResult."""

    def test_create_result(self):
        """Test creating test result."""
        result = TestResult(
            test_id="test_001",
            model="gpt-4o-mini",
            provider="openai",
            prompt="Test",
            response="Response",
            expected="Expected",
        )

        assert result.test_id == "test_001"
        assert result.passed is True  # No failed metrics

    def test_passed_property(self):
        """Test passed property."""
        result = TestResult(
            test_id="test_001",
            model="gpt-4o-mini",
            provider="openai",
            prompt="Test",
            response="Response",
            expected="Expected",
            metrics={
                "exact_match": MetricResult(
                    name="exact_match",
                    value=1.0,
                    passed=True,
                )
            },
        )

        assert result.passed is True

    def test_overall_score(self):
        """Test overall score calculation."""
        result = TestResult(
            test_id="test_001",
            model="gpt-4o-mini",
            provider="openai",
            prompt="Test",
            response="Response",
            expected="Expected",
            metrics={
                "metric1": MetricResult(name="metric1", value=0.8, passed=True),
                "metric2": MetricResult(name="metric2", value=0.6, passed=True),
            },
        )

        assert result.overall_score == 0.7


# ============================================================================
# Tests: Reporting
# ============================================================================

class TestReportGenerator:
    """Test ReportGenerator."""

    def test_generate_markdown_report(self, sample_dataset, sample_llm_response):
        """Test generating markdown report."""
        # Create a mock evaluation result
        result = EvaluationResult(
            config=EvaluationConfig(
                name="test",
                dataset="test_dataset",
                models=[ModelConfig(provider="openai", model="gpt-4o-mini")],
                metrics=["exact_match"],
            ),
            results=[
                TestResult(
                    test_id="test_001",
                    model="gpt-4o-mini",
                    provider="openai",
                    prompt="What is 2+2?",
                    response="4",
                    expected="4",
                    metrics={
                        "exact_match": MetricResult(
                            name="exact_match",
                            value=1.0,
                            passed=True,
                        )
                    },
                    latency_ms=100,
                    cost_usd=0.0001,
                )
            ],
            summary={
                "total_tests": 1,
                "successful_tests": 1,
                "failed_tests": 0,
                "success_rate": 100.0,
                "total_cost_usd": 0.0001,
            },
            start_time=datetime.utcnow().isoformat(),
            end_time=datetime.utcnow().isoformat(),
            duration_seconds=10.0,
        )

        generator = ReportGenerator()
        markdown = generator.generate_markdown_report(result)

        assert "# Evaluation Report: test" in markdown
        assert "## Summary" in markdown
        assert "## Model Comparison" in markdown


# ============================================================================
# Integration Tests
# ============================================================================

class TestEvaluateMetrics:
    """Test evaluate_metrics function."""

    def test_evaluate_single_metric(self):
        """Test evaluating with single metric."""
        results = asyncio.run(evaluate_metrics(
            response="Paris",
            expected="Paris",
            metrics=[create_metric("exact_match")],
        ))

        assert results.overall_score == 1.0
        assert results.total_metrics == 1
        assert results.passed_metrics == 1

    def test_evaluate_multiple_metrics(self):
        """Test evaluating with multiple metrics."""
        results = asyncio.run(evaluate_metrics(
            response='{"name": "John"}',
            expected='{"name": "John"}',
            metrics=[
                create_metric("exact_match"),
                create_metric("format", expected_format="json"),
            ],
        ))

        assert results.total_metrics == 2


# ============================================================================
# Test Utilities
# ============================================================================

def test_metric_result_to_dict():
    """Test MetricResult serialization."""
    result = MetricResult(
        name="test_metric",
        value=0.85,
        passed=True,
        threshold=0.8,
    )

    data = result.to_dict()

    assert data["name"] == "test_metric"
    assert data["value"] == 0.85
    assert data["passed"] is True
    assert data["threshold"] == 0.8


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
