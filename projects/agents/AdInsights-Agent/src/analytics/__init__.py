"""
Statistical analysis functions.
"""

# Import from statistics module
from src.analytics.statistics import (
    calculate_moving_average,
    calculate_ema,
    detect_changepoints,
    seasonal_decomposition,
    forecast_metric,
    ab_test_significance,
    calculate_lift,
    calculate_metrics_summary,
)

# Import from time_series module
from src.analytics.time_series import (
    TimeSeriesAnalyzer,
    TrendDirection,
    TrendResult,
    AnomalyType,
    AnomalyResult,
    ForecastResult,
    analyze_trend,
    detect_anomalies,
    forecast_metrics,
)

# Import from cohort module
from src.analytics.cohort import (
    CohortAnalyzer,
    CohortMetrics,
    CohortComparison,
    RetentionMetrics,
    analyze_time_cohorts,
    compare_cohorts,
    analyze_retention,
)

__all__ = [
    # Statistics module functions
    "calculate_moving_average",
    "calculate_ema",
    "detect_changepoints",
    "seasonal_decomposition",
    "forecast_metric",
    "ab_test_significance",
    "calculate_lift",
    "calculate_metrics_summary",
    # Time series analytics
    "TimeSeriesAnalyzer",
    "TrendDirection",
    "TrendResult",
    "AnomalyType",
    "AnomalyResult",
    "ForecastResult",
    "analyze_trend",
    "detect_anomalies",
    "forecast_metrics",
    # Cohort analytics
    "CohortAnalyzer",
    "CohortMetrics",
    "CohortComparison",
    "RetentionMetrics",
    "analyze_time_cohorts",
    "compare_cohorts",
    "analyze_retention",
]
