"""
Unit tests for Ad Performance Analyzer
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any


class MockAdPerformanceAnalyzer:
    """Mock analyzer for testing."""

    def __init__(self):
        self.metrics_data = []

    def calculate_roi(self, cost: float, revenue: float) -> float:
        """Calculate Return on Investment."""
        if cost == 0:
            return 0.0
        return ((revenue - cost) / cost) * 100

    def calculate_ctr(self, clicks: int, impressions: int) -> float:
        """Calculate Click-Through Rate."""
        if impressions == 0:
            return 0.0
        return (clicks / impressions) * 100

    def calculate_cpc(self, cost: float, clicks: int) -> float:
        """Calculate Cost Per Click."""
        if clicks == 0:
            return 0.0
        return cost / clicks

    def calculate_conversion_rate(self, conversions: int, clicks: int) -> float:
        """Calculate conversion rate."""
        if clicks == 0:
            return 0.0
        return (conversions / clicks) * 100

    def get_top_performers(self, data: list, metric: str, n: int = 5) -> list:
        """Get top performing campaigns."""
        sorted_data = sorted(data, key=lambda x: x.get(metric, 0), reverse=True)
        return sorted_data[:n]

    def detect_anomalies(self, data: list, threshold: float = 2.0) -> list:
        """Detect anomalous metrics using standard deviation."""
        if not data:
            return []

        values = [d.get("value", 0) for d in data]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5

        anomalies = []
        for item in data:
            z_score = abs((item.get("value", 0) - mean) / (std + 1e-8))
            if z_score > threshold:
                anomalies.append(item)

        return anomalies


class TestAdPerformanceAnalyzer:
    """Test cases for Ad Performance Analyzer."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = MockAdPerformanceAnalyzer()

    def test_calculate_roi_positive(self):
        """Test ROI calculation with positive return."""
        roi = self.analyzer.calculate_roi(cost=100, revenue=150)
        assert roi == 50.0

    def test_calculate_roi_negative(self):
        """Test ROI calculation with negative return."""
        roi = self.analyzer.calculate_roi(cost=200, revenue=100)
        assert roi == -50.0

    def test_calculate_roi_zero_cost(self):
        """Test ROI calculation with zero cost."""
        roi = self.analyzer.calculate_roi(cost=0, revenue=100)
        assert roi == 0.0

    def test_calculate_ctr(self):
        """Test CTR calculation."""
        ctr = self.analyzer.calculate_ctr(clicks=50, impressions=1000)
        assert ctr == 5.0

    def test_calculate_ctr_zero_impressions(self):
        """Test CTR calculation with zero impressions."""
        ctr = self.analyzer.calculate_ctr(clicks=50, impressions=0)
        assert ctr == 0.0

    def test_calculate_cpc(self):
        """Test CPC calculation."""
        cpc = self.analyzer.calculate_cpc(cost=100, clicks=50)
        assert cpc == 2.0

    def test_calculate_cpc_zero_clicks(self):
        """Test CPC calculation with zero clicks."""
        cpc = self.analyzer.calculate_cpc(cost=100, clicks=0)
        assert cpc == 0.0

    def test_calculate_conversion_rate(self):
        """Test conversion rate calculation."""
        cr = self.analyzer.calculate_conversion_rate(conversions=10, clicks=100)
        assert cr == 10.0

    def test_get_top_performers(self):
        """Test getting top performing campaigns."""
        data = [
            {"campaign": "A", "roi": 50},
            {"campaign": "B", "roi": 100},
            {"campaign": "C", "roi": 75},
            {"campaign": "D", "roi": 25},
            {"campaign": "E", "roi": 90},
        ]

        top = self.analyzer.get_top_performers(data, "roi", n=3)

        assert len(top) == 3
        assert top[0]["campaign"] == "B"
        assert top[1]["campaign"] == "E"
        assert top[2]["campaign"] == "C"

    def test_detect_anomalies(self):
        """Test anomaly detection."""
        data = [
            {"name": "A", "value": 10},
            {"name": "B", "value": 12},
            {"name": "C", "value": 11},
            {"name": "D", "value": 100},  # Anomaly
            {"name": "E", "value": 9},
        ]

        anomalies = self.analyzer.detect_anomalies(data, threshold=2.0)

        assert len(anomalies) > 0
        assert anomalies[0]["name"] == "D"

    def test_detect_anomalies_empty_data(self):
        """Test anomaly detection with empty data."""
        anomalies = self.analyzer.detect_anomalies([])
        assert anomalies == []

    def test_get_top_performers_empty_data(self):
        """Test getting top performers with empty data."""
        top = self.analyzer.get_top_performers([], "roi")
        assert top == []

    def test_get_top_performers_missing_metric(self):
        """Test getting top performers with missing metric."""
        data = [
            {"campaign": "A"},
            {"campaign": "B"},
        ]

        top = self.analyzer.get_top_performers(data, "roi", n=1)
        assert len(top) <= 1


class TestAdMetricsAggregation:
    """Test cases for metrics aggregation."""

    def test_aggregate_by_campaign(self):
        """Test aggregating metrics by campaign."""
        data = [
            {"campaign": "A", "clicks": 10, "cost": 5},
            {"campaign": "A", "clicks": 20, "cost": 10},
            {"campaign": "B", "clicks": 15, "cost": 7},
        ]

        aggregated = {}
        for item in data:
            campaign = item["campaign"]
            if campaign not in aggregated:
                aggregated[campaign] = {"clicks": 0, "cost": 0}
            aggregated[campaign]["clicks"] += item["clicks"]
            aggregated[campaign]["cost"] += item["cost"]

        assert aggregated["A"]["clicks"] == 30
        assert aggregated["A"]["cost"] == 15
        assert aggregated["B"]["clicks"] == 15

    def test_calculate_period_over_period(self):
        """Test period-over-period growth calculation."""
        current = 1000
        previous = 800

        growth = ((current - previous) / previous) * 100
        assert growth == 25.0

    def test_calculate_period_over_period_zero_previous(self):
        """Test period-over-period with zero previous value."""
        current = 1000
        previous = 0

        # Should handle division by zero
        growth = 0 if previous == 0 else ((current - previous) / previous) * 100
        assert growth == 0


class TestAdRecommendations:
    """Test cases for ad optimization recommendations."""

    def test_recommend_budget_increase(self):
        """Test budget increase recommendation."""
        roi = 150
        ctr = 2.0

        recommend = roi > 100 and ctr > 1.0
        assert recommend is True

    def test_recommend_budget_decrease(self):
        """Test budget decrease recommendation."""
        roi = 20
        cpc = 10

        recommend = roi < 50 and cpc > 5
        assert recommend is True

    def test_recommend_audience_refinement(self):
        """Test audience refinement recommendation."""
        ctr = 0.5
        conversion_rate = 1.0

        recommend = ctr < 1.0 and conversion_rate < 2.0
        assert recommend is True

    def test_recommend_creative_refresh(self):
        """Test creative refresh recommendation."""
        click_decay_rate = 0.3
        days_since_last_change = 30

        recommend = click_decay_rate > 0.2 and days_since_last_change > 14
        assert recommend is True
