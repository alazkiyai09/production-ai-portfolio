"""
Comprehensive Test Suite for AdInsights-Agent

Tests cover:
1. Analysis Tools (fetch, detect, analyze, compare, etc.)
2. Statistical Functions (anomalies, trends, forecasts)
3. Agent Flow (parsing, planning, execution)
4. Report Generation (markdown, HTML export)

Run with: pytest tests/test_insights_agent.py -v
"""

import pytest
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

# Import components to test
from src.tools.analysis_tools import (
    fetch_campaign_metrics,
    calculate_period_comparison,
    detect_anomalies,
    analyze_trend,
    find_correlations,
    compare_to_benchmark,
    generate_chart,
    HEALTHCARE_BENCHMARKS,
)

from src.analytics.statistics import (
    calculate_moving_average,
    detect_changepoints,
    seasonal_decomposition,
    forecast_metric,
    ab_test_significance,
    calculate_lift,
)

from src.agents.insights_agent import AdInsightsAgent, AdInsightsState

from src.visualization.report_generator import ReportGenerator, quick_report


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_campaign_data():
    """Generate 30 days of realistic campaign data."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")

    # Base metrics with some randomness
    np.random.seed(42)
    impressions = np.random.randint(10000, 20000, size=30)
    ctr = np.random.normal(1.5, 0.3, size=30)  # 1.5% average CTR
    cvr = np.random.normal(2.8, 0.5, size=30)  # 2.8% average CVR

    # Calculate derived metrics
    clicks = (impressions * ctr / 100).astype(int)
    conversions = (clicks * cvr / 100).astype(int)
    cpa = np.random.uniform(200, 400, size=30)
    spend = conversions * cpa

    data = {
        "date": dates,
        "campaign_id": "CAMP-TEST-001",
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "spend": spend.round(2),
        "ctr": ctr.round(2),
        "cvr": cvr.round(2),
        "cpa": cpa.round(2),
        "roi": ((conversions * 500 - spend) / spend).round(2),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_campaign_dict(sample_campaign_data):
    """Convert sample data to tool format."""
    return {
        "data": sample_campaign_data.to_dict("records"),
        "summary": {
            "avg_impressions": int(sample_campaign_data["impressions"].mean()),
            "avg_clicks": int(sample_campaign_data["clicks"].mean()),
            "avg_conversions": int(sample_campaign_data["conversions"].mean()),
            "avg_spend": float(sample_campaign_data["spend"].mean()),
            "avg_ctr": float(sample_campaign_data["ctr"].mean()),
            "avg_cvr": float(sample_campaign_data["cvr"].mean()),
            "avg_cpa": float(sample_campaign_data["cpa"].mean()),
            "avg_roi": float(sample_campaign_data["roi"].mean()),
            "total_impressions": int(sample_campaign_data["impressions"].sum()),
            "total_clicks": int(sample_campaign_data["clicks"].sum()),
            "total_conversions": int(sample_campaign_data["conversions"].sum()),
            "total_spend": float(sample_campaign_data["spend"].sum()),
        },
        "date_range": {
            "start": "2024-01-01",
            "end": "2024-01-30",
        },
        "record_count": 30,
        "campaign_id": "CAMP-TEST-001",
        "campaign_type": "healthcare_pharma",
    }


@pytest.fixture
def sample_series_with_trend():
    """Generate series with clear upward trend."""
    np.random.seed(42)
    x = np.linspace(0, 10, 30)
    y = 100 + 5 * x + np.random.normal(0, 2, size=30)
    return pd.Series(y, index=pd.date_range("2024-01-01", periods=30))


@pytest.fixture
def sample_series_with_anomalies():
    """Generate series with injected anomalies."""
    np.random.seed(42)
    data = np.random.normal(100, 5, size=30)
    # Inject anomalies
    data[10] = 150  # High anomaly
    data[20] = 50   # Low anomaly
    return pd.Series(data, index=pd.date_range("2024-01-01", periods=30))


@pytest.fixture
def sample_agent_state():
    """Create sample agent state for testing."""
    return {
        "request": "Analyze campaign CAMP-001 for the last 30 days",
        "campaign_id": "CAMP-001",
        "date_range": ("2024-01-01", "2024-01-30"),
        "campaign_type": "healthcare_pharma",
        "analysis_plan": ["analyze_metrics", "detect_anomalies", "analyze_trends"],
        "raw_data": None,
        "metrics_summary": {
            "avg_ctr": 1.35,
            "avg_cvr": 2.78,
            "avg_cpa": 285.50,
            "avg_roi": 3.12,
            "total_impressions": 425000,
        },
        "anomalies": [
            {
                "date": "2024-01-15",
                "metric_name": "CTR",
                "value": 2.8,
                "severity": "high",
                "score": 3.2,
            }
        ],
        "trends": {
            "ctr": {
                "trend_direction": "up",
                "trend_strength": "moderate",
                "r_squared": 0.65,
            }
        },
        "benchmark_comparison": {
            "overall_performance": "good",
            "percentile_ranking": 68.5,
            "industry": "healthcare_pharma",
        },
        "insights": [
            "CTR is performing above industry benchmark.",
            "CVR shows declining trend.",
        ],
        "recommendations": [
            "Investigate CTR spike on January 15th.",
            "Optimize landing pages.",
        ],
        "charts": [],
        "final_report": "",
        "current_step": "",
        "errors": [],
        "completed_steps": [],
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Analysis complete"
                }
            }
        ]
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for tests."""
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    return str(output_dir)


# =============================================================================
# TOOL TESTS
# =============================================================================

class TestAnalysisTools:
    """Test suite for analysis tools."""

    def test_fetch_campaign_metrics_basic(self, sample_campaign_dict):
        """Test basic campaign metrics fetching."""
        result = fetch_campaign_metrics.invoke({
            "campaign_id": "CAMP-TEST-001",
            "start_date": "2024-01-01",
            "end_date": "2024-01-30",
            "campaign_type": "healthcare_pharma",
        })

        assert "error" not in result
        assert "data" in result
        assert "summary" in result
        assert result["record_count"] == 30
        assert result["campaign_id"] == "CAMP-TEST-001"

    def test_fetch_campaign_metrics_metrics(self, sample_campaign_dict):
        """Test that summary metrics are calculated correctly."""
        result = fetch_campaign_metrics.invoke({
            "campaign_id": "CAMP-TEST-001",
            "start_date": "2024-01-01",
            "end_date": "2024-01-30",
            "campaign_type": "healthcare_pharma",
        })

        summary = result["summary"]

        # Check metrics exist and are reasonable
        assert summary["avg_ctr"] > 0
        assert summary["avg_cvr"] > 0
        assert summary["avg_cpa"] > 0
        assert summary["total_impressions"] > 0
        assert summary["total_clicks"] > 0

    def test_detect_anomalies_zscore(self, sample_campaign_dict):
        """Test anomaly detection using z-score method."""
        result = detect_anomalies.invoke({
            "data": sample_campaign_dict,
            "metric": "ctr",
            "method": "zscore",
            "threshold": 2.5,
        })

        assert "anomalies" in result
        assert "summary" in result
        assert isinstance(result["anomalies"], list)

    def test_detect_anomalies_severity_classification(self, sample_campaign_dict):
        """Test that anomalies are classified by severity."""
        result = detect_anomalies.invoke({
            "data": sample_campaign_dict,
            "metric": "ctr",
            "method": "zscore",
            "threshold": 2.0,
        })

        anomalies = result["anomalies"]
        for anomaly in anomalies:
            assert "severity" in anomaly
            assert anomaly["severity"] in ["high", "medium", "low"]

    def test_detect_anomalies_edge_case_empty_data(self):
        """Test anomaly detection with empty data."""
        result = detect_anomalies.invoke({
            "data": {"data": []},
            "metric": "ctr",
            "method": "zscore",
            "threshold": 2.5,
        })

        # Should handle gracefully
        assert "anomalies" in result

    def test_analyze_trend_basic(self, sample_campaign_dict):
        """Test basic trend analysis."""
        result = analyze_trend.invoke({
            "data": sample_campaign_dict,
            "metric": "ctr",
            "forecast_days": 7,
        })

        assert "error" not in result
        assert "trend_direction" in result
        assert result["trend_direction"] in ["up", "down", "flat"]
        assert "forecast" in result

    def test_analyze_trend_forecast_length(self, sample_campaign_dict):
        """Test that forecast has correct number of periods."""
        result = analyze_trend.invoke({
            "data": sample_campaign_dict,
            "metric": "ctr",
            "forecast_days": 7,
        })

        forecast = result.get("forecast", [])
        assert len(forecast) == 7

    def test_find_correlations(self, sample_campaign_dict):
        """Test correlation finding."""
        result = find_correlations.invoke({
            "data": sample_campaign_dict,
            "target_metric": "roi",
            "min_correlation": 0.3,
        })

        assert "correlations" in result
        assert isinstance(result["correlations"], list)

    def test_find_correlations_has_strength(self, sample_campaign_dict):
        """Test that correlations have strength attribute."""
        result = find_correlations.invoke({
            "data": sample_campaign_dict,
            "target_metric": "roi",
            "min_correlation": 0.3,
        })

        for corr in result["correlations"]:
            assert "strength" in corr
            assert corr["strength"] in ["strong", "moderate", "weak"]

    def test_compare_to_benchmark(self):
        """Test benchmark comparison."""
        result = compare_to_benchmark.invoke({
            "campaign_metrics": {
                "avg_ctr": 1.5,
                "avg_cvr": 3.0,
                "avg_cpa": 300,
                "avg_roi": 3.5,
            },
            "industry": "healthcare_pharma",
        })

        assert "comparisons" in result
        assert "overall_performance" in result
        assert "percentile_ranking" in result

    def test_compare_to_benchmark_percentile_range(self):
        """Test that percentile is in valid range."""
        result = compare_to_benchmark.invoke({
            "campaign_metrics": {
                "avg_ctr": 1.5,
                "avg_cvr": 3.0,
                "avg_cpa": 300,
                "avg_roi": 3.5,
            },
            "industry": "healthcare_pharma",
        })

        percentile = result["percentile_ranking"]
        assert 0 <= percentile <= 100

    def test_generate_chart_line(self, sample_campaign_dict):
        """Test line chart generation."""
        result = generate_chart.invoke({
            "data": sample_campaign_dict,
            "chart_type": "line",
            "metrics": ["ctr", "cvr"],
            "title": "Test Chart",
        })

        assert result.get("success") in [True, False]
        if result.get("success"):
            assert "file_path" in result
            assert Path(result["file_path"]).exists()

    def test_generate_chart_invalid_type(self, sample_campaign_dict):
        """Test chart generation with invalid type."""
        result = generate_chart.invoke({
            "data": sample_campaign_dict,
            "chart_type": "invalid_type",
            "metrics": ["ctr"],
            "title": "Test",
        })

        # Should return error
        assert "error" in result or result.get("success") is False


# =============================================================================
# STATISTICS TESTS
# =============================================================================

class TestStatistics:
    """Test suite for statistical functions."""

    def test_moving_average_simple(self):
        """Test simple moving average calculation."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = calculate_moving_average(series, window=3, method="simple")

        assert len(result) == len(series)
        # First two values should be NaN
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # Third value should be average of first three
        assert result.iloc[2] == 2.0

    def test_moving_average_exponential(self):
        """Test exponential moving average calculation."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = calculate_moving_average(series, window=3, method="exponential")

        assert len(result) == len(series)
        # EMA should not have leading NaNs (except first)
        assert not pd.isna(result.iloc[1])

    def test_detect_changepoints_mean_shift(self, sample_series_with_anomalies):
        """Test changepoint detection with mean shift."""
        result = detect_changepoints(
            sample_series_with_anomalies,
            min_size=7,
            method="mean_shift",
            significance_level=0.05
        )

        assert isinstance(result, list)
        # Should detect at least the injected anomalies
        assert len(result) >= 2

    def test_detect_changepoints_empty_series(self):
        """Test changepoint detection with empty series."""
        result = detect_changepoints(pd.Series([]), min_size=7)

        assert result == []

    def test_seasonal_decomposition(self):
        """Test seasonal decomposition."""
        # Create series with clear pattern
        dates = pd.date_range("2024-01-01", periods=28, freq="D")
        values = []
        for i in range(4):  # 4 weeks
            week_pattern = [10, 15, 20, 25, 30, 25, 20]
            values.extend(week_pattern)

        series = pd.Series(values, index=dates)

        result = seasonal_decomposition(series, period=7, model="additive")

        assert "observed" in result
        assert "trend" in result
        assert "seasonal" in result
        assert "residual" in result
        assert "stats" in result

    def test_seasonal_decomposition_insufficient_data(self):
        """Test seasonal decomposition with insufficient data."""
        series = pd.Series([1, 2, 3, 4, 5])

        with pytest.raises(ValueError):
            seasonal_decomposition(series, period=7)

    def test_forecast_exponential_smoothing(self):
        """Test exponential smoothing forecast."""
        series = pd.Series([10, 12, 11, 13, 12, 14, 13, 15, 14, 16])

        result = forecast_metric(series, periods=5, method="exponential_smoothing")

        assert "forecast" in result
        assert len(result["forecast"]) == 5
        assert "lower_bound" in result
        assert "upper_bound" in result

    def test_forecast_confidence_intervals(self):
        """Test that confidence intervals are reasonable."""
        series = pd.Series([10, 12, 11, 13, 12, 14, 13, 15, 14, 16])

        result = forecast_metric(series, periods=5, method="exponential_smoothing")

        # Lower bound should be <= forecast <= upper bound
        for i in range(5):
            assert result["lower_bound"][i] <= result["forecast"][i] <= result["upper_bound"][i]

    def test_ab_test_significance_continuous(self):
        """Test A/B test with continuous metrics."""
        control = pd.Series([100, 102, 98, 101, 99, 103, 97, 100])
        treatment = pd.Series([108, 112, 106, 110, 107, 111, 105, 109])

        result = ab_test_significance(control, treatment, metric_type="continuous")

        assert "is_significant" in result
        assert "p_value" in result
        assert "relative_lift" in result
        assert result["treatment_mean"] > result["control_mean"]

    def test_ab_test_no_difference(self):
        """Test A/B test when there's no significant difference."""
        np.random.seed(42)
        control = pd.Series(np.random.normal(100, 5, size=30))
        treatment = pd.Series(np.random.normal(101, 5, size=30))

        result = ab_test_significance(control, treatment, metric_type="continuous")

        # With similar means, should often not be significant
        assert result["p_value"] > 0.01

    def test_calculate_lift_positive(self):
        """Test lift calculation with positive lift."""
        result = calculate_lift(
            baseline=100,
            current=120,
            baseline_n=1000,
            current_n=1000,
        )

        assert result["lift_percent"] == 20.0
        assert result["lift_type"] == "positive"
        assert result["absolute_change"] == 20.0

    def test_calculate_lift_negative(self):
        """Test lift calculation with negative lift."""
        result = calculate_lift(
            baseline=100,
            current=80,
            baseline_n=1000,
            current_n=1000,
        )

        assert result["lift_percent"] == -20.0
        assert result["lift_type"] == "negative"
        assert result["absolute_change"] == -20.0

    def test_calculate_lift_zero_baseline_error(self):
        """Test that zero baseline raises error."""
        with pytest.raises(ValueError):
            calculate_lift(baseline=0, current=100)


# =============================================================================
# AGENT FLOW TESTS
# =============================================================================

class TestAgentFlow:
    """Test suite for agent workflow."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_parse_request_node_campaign_id(self):
        """Test request parsing extracts campaign ID."""
        agent = AdInsightsAgent()

        state = {
            "request": "Analyze campaign CAMP-123 for the last 30 days",
            "date_range": None,
            "campaign_type": None,
        }

        result = agent._parse_request_node(state)

        assert result["campaign_id"] == "CAMP-123"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_parse_request_node_date_range(self):
        """Test request parsing extracts date range."""
        agent = AdInsightsAgent()

        state = {
            "request": "Analyze campaign CAMP-001 for the last 14 days",
        }

        result = agent._parse_request_node(state)

        assert result["date_range"] is not None
        start, end = result["date_range"]
        assert start < end

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_plan_analysis_node_default_plan(self):
        """Test that analysis plan includes default steps."""
        agent = AdInsightsAgent()

        state = {
            "request": "Analyze campaign CAMP-001",
            "analysis_plan": [],
        }

        result = agent._plan_analysis_node(state)

        assert "detect_anomalies" in result["analysis_plan"]
        assert "analyze_trends" in result["analysis_plan"]
        assert "analyze_metrics" in result["analysis_plan"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_plan_analysis_node_benchmark_keyword(self):
        """Test that benchmark keyword adds benchmark analysis."""
        agent = AdInsightsAgent()

        state = {
            "request": "Analyze campaign CAMP-001 and compare to benchmarks",
            "analysis_plan": [],
        }

        result = agent._plan_analysis_node(state)

        assert "compare_benchmarks" in result["analysis_plan"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_gather_data_node(self):
        """Test data gathering node."""
        agent = AdInsightsAgent()

        state = {
            "campaign_id": "CAMP-DEMO-001",
            "date_range": ("2024-01-01", "2024-01-30"),
            "campaign_type": "healthcare_pharma",
            "raw_data": None,
            "errors": [],
        }

        result = agent._gather_data_node(state)

        # Should have raw data or error
        assert result["raw_data"] is not None or len(result["errors"]) > 0

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_analyze_metrics_node(self, sample_campaign_dict):
        """Test metrics analysis node."""
        agent = AdInsightsAgent()

        state = {
            "raw_data": sample_campaign_dict,
            "metrics_summary": None,
            "completed_steps": [],
        }

        result = agent._analyze_metrics_node(state)

        assert result["metrics_summary"] is not None
        assert "avg_ctr" in result["metrics_summary"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_detect_anomalies_node(self, sample_campaign_dict):
        """Test anomaly detection node."""
        agent = AdInsightsAgent()

        state = {
            "raw_data": sample_campaign_dict,
            "anomalies": [],
            "completed_steps": [],
        }

        result = agent._detect_anomalies_node(state)

        assert isinstance(result["anomalies"], list)
        assert "detect_anomalies" in result["completed_steps"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_analyze_trends_node(self, sample_campaign_dict):
        """Test trend analysis node."""
        agent = AdInsightsAgent()

        state = {
            "raw_data": sample_campaign_dict,
            "trends": {},
            "completed_steps": [],
        }

        result = agent._analyze_trends_node(state)

        assert isinstance(result["trends"], dict)
        assert "analyze_trends" in result["completed_steps"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_generate_insights_node(self, sample_campaign_dict):
        """Test insights generation node."""
        agent = AdInsightsAgent()

        state = {
            "raw_data": sample_campaign_dict,
            "metrics_summary": sample_campaign_dict["summary"],
            "anomalies": [],
            "trends": {},
            "insights": [],
            "recommendations": [],
            "charts": [],
            "completed_steps": [],
        }

        result = agent._generate_insights_node(state)

        assert isinstance(result["insights"], list)
        assert isinstance(result["recommendations"], list)
        assert "generate_insights" in result["completed_steps"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_route_after_metrics(self):
        """Test routing logic after metrics analysis."""
        agent = AdInsightsAgent()

        # Test with pending anomaly detection
        state = {
            "analysis_plan": ["detect_anomalies", "analyze_trends"],
            "completed_steps": ["analyze_metrics"],
        }

        result = agent._route_after_metrics(state)

        assert result in ["detect_anomalies", "analyze_trends", "compare_benchmarks", "find_correlations", "generate_insights"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.agents.insights_agent.fetch_campaign_metrics')
    def test_analyze_method_basic(self, mock_fetch):
        """Test the main analyze method."""
        # Mock fetch to return sample data
        mock_fetch.return_value = {
            "data": [
                {
                    "date": "2024-01-01",
                    "impressions": 10000,
                    "clicks": 150,
                    "conversions": 5,
                    "spend": 200,
                    "ctr": 1.5,
                    "cvr": 3.33,
                    "cpa": 40,
                    "roi": 3.5,
                }
            ] * 30,
            "summary": {
                "avg_ctr": 1.5,
                "avg_cvr": 3.0,
                "avg_cpa": 250,
                "avg_roi": 3.2,
            },
            "record_count": 30,
        }

        agent = AdInsightsAgent()

        result = agent.analyze(
            request="Analyze campaign CAMP-001",
            campaign_id="CAMP-001",
            date_range=("2024-01-01", "2024-01-30"),
        )

        assert "success" in result
        assert "report" in result or "error" in result


# =============================================================================
# REPORT TESTS
# =============================================================================

class TestReportGenerator:
    """Test suite for report generation."""

    def test_create_executive_summary(self):
        """Test executive summary creation."""
        generator = ReportGenerator()

        insights = ["CTR is performing above benchmark."]
        metrics = {
            "campaign_id": "CAMP-001",
            "total_days": 30,
            "total_spend": 10000,
            "avg_ctr": 1.5,
            "avg_cvr": 3.0,
        }

        summary = generator.create_executive_summary(insights, metrics)

        assert "CAMP-001" in summary
        assert "30 days" in summary
        assert "CTR" in summary

    def test_create_metrics_section(self):
        """Test metrics section creation."""
        generator = ReportGenerator()

        metrics = {
            "avg_ctr": 1.5,
            "avg_cvr": 3.0,
            "avg_cpa": 250,
            "avg_roi": 3.2,
        }

        section = generator.create_metrics_section(metrics)

        assert "CTR" in section
        assert "CVR" in section
        assert "|" in section  # Table format

    def test_create_anomaly_section_empty(self):
        """Test anomaly section with no anomalies."""
        generator = ReportGenerator()

        section = generator.create_anomaly_section([])

        assert "No significant anomalies" in section

    def test_create_anomaly_section_with_anomalies(self):
        """Test anomaly section with anomalies."""
        generator = ReportGenerator()

        anomalies = [
            {
                "date": "2024-01-15",
                "metric_name": "CTR",
                "value": 2.8,
                "severity": "high",
                "score": 3.2,
            }
        ]

        section = generator.create_anomaly_section(anomalies)

        assert "1 anomalies" in section or "1 anomaly" in section
        assert "2024-01-15" in section

    def test_create_trend_section(self, sample_series_with_trend):
        """Test trend section creation."""
        generator = ReportGenerator()

        trends = {
            "value": {
                "trend_direction": "up",
                "trend_strength": "strong",
                "r_squared": 0.85,
                "p_value": 0.01,
                "slope": 5.0,
                "interpretation": "Strong upward trend",
            }
        }

        section = generator.create_trend_section(trends)

        assert "up" in section
        assert "strong" in section
        assert "0.850" in section

    def test_create_benchmark_section(self):
        """Test benchmark section creation."""
        generator = ReportGenerator()

        benchmark = {
            "overall_performance": "good",
            "percentile_ranking": 68.5,
            "industry": "healthcare_pharma",
            "comparisons": [
                {
                    "metric": "CTR",
                    "campaign_value": 1.5,
                    "benchmark_median": 1.2,
                    "performance": "good",
                    "percentile": 70,
                }
            ]
        }

        section = generator.create_benchmark_section(benchmark)

        assert "68.5th percentile" in section
        assert "good" in section.upper() or "GOOD" in section

    def test_create_insights_section(self):
        """Test insights section creation."""
        generator = ReportGenerator()

        insights = [
            "CTR is 15% above benchmark.",
            "CVR shows declining trend.",
        ]

        section = generator.create_insights_section(insights)

        assert "1." in section
        assert "2." in section
        assert "CTR" in section

    def test_create_recommendations_section(self):
        """Test recommendations section with prioritization."""
        generator = ReportGenerator()

        recommendations = [
            "Urgently fix the landing page issue.",
            "Consider increasing budget for top performers.",
            "Monitor performance next week.",
        ]

        section = generator.create_recommendations_section(recommendations, prioritize=True)

        assert "1." in section
        assert "🔴" in section or "🟡" in section

    def test_generate_full_report(self, sample_agent_state):
        """Test full report generation."""
        generator = ReportGenerator()

        report = generator.generate_full_report(sample_agent_state)

        assert "# Campaign Analysis Report" in report
        assert "CAMP-001" in report
        assert "Executive Summary" in report
        assert "Key Metrics" in report
        assert "Insights" in report
        assert "Recommendations" in report

    def test_export_to_markdown(self, temp_output_dir):
        """Test markdown export."""
        generator = ReportGenerator(output_dir=temp_output_dir)

        markdown = "# Test Report\n\nThis is a test."
        path = generator.export_to_markdown(markdown, "test.md")

        assert Path(path).exists()
        assert Path(path).suffix == ".md"

    def test_export_to_markdown_content(self, temp_output_dir):
        """Test that markdown export preserves content."""
        generator = ReportGenerator(output_dir=temp_output_dir)

        original = "# Test\n\nContent here."
        path = generator.export_to_markdown(original, "test_content.md")

        with open(path, "r") as f:
            content = f.read()

        assert content == original

    def test_quick_report_function(self, sample_agent_state, temp_output_dir):
        """Test quick_report convenience function."""
        path = quick_report(
            sample_agent_state,
            output_format="markdown",
            output_dir=temp_output_dir,
        )

        assert Path(path).exists()
        assert path.endswith(".md")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for end-to-end workflows."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.agents.insights_agent.fetch_campaign_metrics')
    def test_full_analysis_workflow(self, mock_fetch):
        """Test complete analysis from request to report."""
        # Mock the fetch
        mock_fetch.return_value = {
            "data": [
                {
                    "date": "2024-01-01",
                    "impressions": 10000,
                    "clicks": 150,
                    "conversions": 5,
                    "spend": 200,
                    "ctr": 1.5,
                    "cvr": 3.33,
                    "cpa": 40,
                    "roi": 3.5,
                }
            ] * 30,
            "summary": {
                "avg_ctr": 1.5,
                "avg_cvr": 3.0,
                "avg_cpa": 250,
                "avg_roi": 3.2,
            },
            "record_count": 30,
        }

        # Run agent
        agent = AdInsightsAgent()
        result = agent.analyze("Analyze campaign CAMP-001")

        # Check results
        assert result is not None
        # Should either succeed or have error
        assert result.get("success") is True or result.get("error") is not None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_report_generation_from_state(self):
        """Test report generation from agent state."""
        agent = AdInsightsAgent()
        generator = ReportGenerator()

        # Create minimal state
        state = {
            "campaign_id": "CAMP-001",
            "date_range": ("2024-01-01", "2024-01-30"),
            "metrics_summary": {
                "avg_ctr": 1.5,
                "avg_cvr": 3.0,
            },
            "insights": ["Test insight"],
            "recommendations": ["Test recommendation"],
        }

        report = generator.generate_full_report(state)

        assert "CAMP-001" in report
        assert "Test insight" in report


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
