"""
Cohort Analytics Module for AdInsights-Agent

Provides cohort analysis for campaign performance, user segmentation,
and behavioral analysis over time.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import warnings

import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =============================================================================
# RESULT CLASSES
# =============================================================================

class CohortMetrics:
    """Metrics for a single cohort."""

    def __init__(
        self,
        cohort_name: str,
        size: int,
        avg_metric: float,
        std_metric: float,
        median_metric: float,
        total_value: float,
        pct_change: Optional[float] = None
    ):
        self.cohort_name = cohort_name
        self.size = size
        self.avg_metric = avg_metric
        self.std_metric = std_metric
        self.median_metric = median_metric
        self.total_value = total_value
        self.pct_change = pct_change  # Percentage change from previous period

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cohort_name": self.cohort_name,
            "size": self.size,
            "avg_metric": self.avg_metric,
            "std_metric": self.std_metric,
            "median_metric": self.median_metric,
            "total_value": self.total_value,
            "pct_change": self.pct_change
        }


class CohortComparison:
    """Comparison results between cohorts."""

    def __init__(
        self,
        metric: str,
        cohorts: List[CohortMetrics],
        statistical_significance: float,
        winner: str,
        uplift: float,
        confidence_interval: Tuple[float, float]
    ):
        self.metric = metric
        self.cohorts = cohorts
        self.statistical_significance = statistical_significance  # p-value
        self.winner = winner
        self.uplift = uplift  # Relative improvement
        self.confidence_interval = confidence_interval

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric,
            "cohorts": [c.to_dict() for c in self.cohorts],
            "statistical_significance": self.statistical_significance,
            "winner": self.winner,
            "uplift": self.uplift,
            "confidence_interval": self.confidence_interval
        }


class RetentionMetrics:
    """Retention analysis results."""

    def __init__(
        self,
        cohort_size: int,
        retention_rates: List[float],  # Retention rate per period
        period_labels: List[str],
        churn_rate: float
    ):
        self.cohort_size = cohort_size
        self.retention_rates = retention_rates
        self.period_labels = period_labels
        self.churn_rate = churn_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cohort_size": self.cohort_size,
            "retention_rates": self.retention_rates,
            "period_labels": self.period_labels,
            "churn_rate": self.churn_rate
        }


# =============================================================================
# COHORT ANALYZER
# =============================================================================

class CohortAnalyzer:
    """
    Cohort analysis for ad campaign performance.

    Provides:
    - Time-based cohorting (by acquisition date, campaign, etc.)
    - Behavioral cohorting (by user segments)
    - Cohort comparison with statistical significance testing
    - Retention analysis over time
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        min_cohort_size: int = 30
    ):
        """
        Initialize the cohort analyzer.

        Args:
            confidence_level: Confidence level for statistical tests
            min_cohort_size: Minimum cohort size for analysis
        """
        self.confidence_level = confidence_level
        self.min_cohort_size = min_cohort_size

    def analyze_by_time_cohort(
        self,
        data: pd.DataFrame,
        cohort_column: str,
        value_column: str,
        date_column: str = "date"
    ) -> List[CohortMetrics]:
        """
        Analyze metrics by time-based cohorts.

        Args:
            data: DataFrame with campaign data
            cohort_column: Column to cohort by (e.g., "acquisition_date", "campaign_name")
            value_column: Metric to analyze (e.g., "revenue", "conversions")
            date_column: Date column (if not using cohort_column)

        Returns:
            List of CohortMetrics sorted by cohort

        Raises:
            ValueError: If insufficient data or invalid columns
        """
        df = data.copy()

        # Validate columns
        required_cols = [cohort_column, value_column]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Group by cohort
        grouped = df.groupby(cohort_column)[value_column].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('median', 'median'),
            ('sum', 'sum'),
            ('count', 'count')
        ]).reset_index()

        # Flatten column names
        grouped.columns = [cohort_column, 'mean', 'std', 'median', 'sum', 'count']

        # Filter by minimum size
        grouped = grouped[grouped['count'] >= self.min_cohort_size]

        # Calculate percentage change from previous cohort
        grouped['pct_change'] = grouped['mean'].pct_change() * 100

        # Create CohortMetrics objects
        cohorts = []
        for _, row in grouped.iterrows():
            cohorts.append(CohortMetrics(
                cohort_name=str(row[cohort_column]),
                size=int(row['count']),
                avg_metric=float(row['mean']),
                std_metric=float(row['std']) if not pd.isna(row['std']) else 0.0,
                median_metric=float(row['median']),
                total_value=float(row['sum']),
                pct_change=float(row['pct_change']) if not pd.isna(row['pct_change']) else None
            ))

        logger.info(f"Analyzed {len(cohorts)} cohorts by {cohort_column}")

        return cohorts

    def compare_cohorts(
        self,
        data: pd.DataFrame,
        cohort_column: str,
        value_column: str,
        metric_name: Optional[str] = None
    ) -> CohortComparison:
        """
        Compare cohorts with statistical significance testing.

        Performs t-test between cohorts to determine if differences
        are statistically significant.

        Args:
            data: DataFrame with campaign data
            cohort_column: Column defining cohorts
            value_column: Metric to compare
            metric_name: Human-readable metric name

        Returns:
            CohortComparison with statistical analysis

        Raises:
            ValueError: If fewer than 2 cohorts or insufficient data
        """
        df = data.copy()

        # Validate
        if cohort_column not in df.columns or value_column not in df.columns:
            raise ValueError(f"Missing required columns")

        # Get unique cohorts
        cohorts = df[cohort_column].unique()

        if len(cohorts) < 2:
            raise ValueError("Need at least 2 cohorts for comparison")

        # Analyze each cohort
        cohort_metrics = self.analyze_by_time_cohort(
            df, cohort_column, value_column
        )

        if len(cohort_metrics) < 2:
            raise ValueError(
                f"Need at least 2 valid cohorts (min size {self.min_cohort_size})"
            )

        # Find best and worst performing cohorts
        sorted_cohorts = sorted(cohort_metrics, key=lambda c: c.avg_metric, reverse=True)
        best = sorted_cohorts[0]
        worst = sorted_cohorts[-1]

        # Perform statistical test
        best_data = df[df[cohort_column] == best.cohort_name][value_column].dropna()
        worst_data = df[df[cohort_column] == worst.cohort_name][value_column].dropna()

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(best_data, worst_data)

        # Calculate uplift (relative improvement)
        uplift = ((best.avg_metric - worst.avg_metric) / worst.avg_metric * 100) if worst.avg_metric != 0 else 0

        # Calculate confidence interval for difference
        se_diff = np.sqrt(best.std_metric**2/best.size + worst.std_metric**2/worst.size)
        ci_width = 1.96 * se_diff  # 95% CI
        diff_mean = best.avg_metric - worst.avg_metric
        ci_lower = diff_mean - ci_width
        ci_upper = diff_mean + ci_width

        logger.info(
            f"Cohort comparison: {best.cohort_name} vs {worst.cohort_name}, "
            f"p={p_value:.4f}, uplift={uplift:.1f}%"
        )

        return CohortComparison(
            metric=metric_name or value_column,
            cohorts=[best, worst],
            statistical_significance=p_value,
            winner=best.cohort_name,
            uplift=uplift,
            confidence_interval=(ci_lower, ci_upper)
        )

    def analyze_retention(
        self,
        data: pd.DataFrame,
        user_column: str,
        date_column: str,
        activity_column: Optional[str] = None
    ) -> RetentionMetrics:
        """
        Analyze user retention over time.

        Args:
            data: DataFrame with user activity data
            user_column: User identifier column
            date_column: Date column for activity
            activity_column: Optional activity count column

        Returns:
            RetentionMetrics with retention rates by period

        Raises:
            ValueError: If insufficient data
        """
        df = data.copy()

        # Convert date column
        df[date_column] = pd.to_datetime(df[date_column])

        # Find first activity per user
        user_first = df.groupby(user_column)[date_column].min().reset_index()
        user_first.columns = [user_column, 'first_seen']

        # Merge back to get period offsets
        df = df.merge(user_first, on=user_column)
        df['period_offset'] = ((df[date_column] - df['first_seen']).dt.days // 7) + 1  # Weekly periods

        # Calculate retention rate per period
        retention = []
        period_labels = []

        unique_users = df[user_column].nunique()

        for period in range(1, 9):  # 8 periods
            users_active = df[df['period_offset'] == period][user_column].nunique()
            retention_rate = users_active / unique_users if unique_users > 0 else 0
            retention.append(retention_rate)
            period_labels.append(f"Period {period}")

        # Calculate churn rate
        churn_rate = 1 - retention[0]  # Churn after first period

        logger.info(f"Retention analysis: {unique_users} users, churn rate {churn_rate:.2%}")

        return RetentionMetrics(
            cohort_size=unique_users,
            retention_rates=retention,
            period_labels=period_labels,
            churn_rate=churn_rate
        )

    def segment_by_behavior(
        self,
        data: pd.DataFrame,
        value_column: str,
        n_segments: int = 4,
        segment_labels: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Segment users into behavioral cohorts based on metric value.

        Uses quantile-based segmentation for robust grouping.

        Args:
            data: DataFrame with user metrics
            value_column: Metric to segment by
            n_segments: Number of segments to create
            segment_labels: Optional custom labels for segments

        Returns:
            DataFrame with segment assignments

        Raises:
            ValueError: If invalid parameters
        """
        df = data.copy()

        if value_column not in df.columns:
            raise ValueError(f"Missing column: {value_column}")

        if n_segments < 2 or n_segments > 10:
            raise ValueError("n_segments must be between 2 and 10")

        # Calculate quantiles
        quantiles = [i / n_segments for i in range(n_segments + 1)]

        # Create segment labels
        if segment_labels is None:
            segment_labels = [f"Segment {i+1} ({quantiles[i]*100:.0f}-{quantiles[i+1]*100:.0f}%)"
                            for i in range(n_segments)]
        else:
            if len(segment_labels) != n_segments:
                raise ValueError(f"Need {n_segments} segment labels, got {len(segment_labels)}")

        # Assign segments
        df['_segment'] = pd.qcut(
            df[value_column],
            q=quantiles,
            labels=segment_labels,
            duplicates='drop'
        )

        logger.info(f"Created {n_segments} behavioral segments by {value_column}")

        return df

    def analyze_lifecycle(
        self,
        data: pd.DataFrame,
        date_column: str,
        value_column: str,
        lifecycle_stages: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> Dict[str, CohortMetrics]:
        """
        Analyze metrics by customer lifecycle stage.

        Args:
            data: DataFrame with time series data
            date_column: Date column
            value_column: Metric column
            lifecycle_stages: Dict mapping stage name to (start_day, end_day)

        Returns:
            Dictionary mapping lifecycle stage to CohortMetrics

        Raises:
            ValueError: If insufficient data
        """
        df = data.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Default lifecycle stages (days since first purchase)
        if lifecycle_stages is None:
            lifecycle_stages = {
                "new": (0, 7),        # Week 1
                "active": (8, 30),    # Weeks 2-4
                "mature": (31, 90),   # Months 2-3
                "churned": (91, 365)  # Months 4-12
            }

        # Calculate days since start
        min_date = df[date_column].min()
        df['days_since_start'] = (df[date_column] - min_date).dt.days

        # Assign lifecycle stages
        def get_lifecycle(days: int) -> str:
            for stage, (start, end) in lifecycle_stages.items():
                if start <= days <= end:
                    return stage
            return "unknown"

        df['lifecycle_stage'] = df['days_since_start'].apply(get_lifecycle)

        # Analyze by lifecycle stage
        results = {}
        for stage in lifecycle_stages.keys():
            stage_data = df[df['lifecycle_stage'] == stage]

            if len(stage_data) > 0:
                values = stage_data[value_column].dropna()

                if len(values) >= self.min_cohort_size:
                    results[stage] = CohortMetrics(
                        cohort_name=stage,
                        size=len(values),
                        avg_metric=values.mean(),
                        std_metric=values.std(),
                        median_metric=values.median(),
                        total_value=values.sum()
                    )

        logger.info(f"Lifecycle analysis: {len(results)} stages analyzed")

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_time_cohorts(
    data: pd.DataFrame,
    cohort_column: str,
    value_column: str
) -> List[CohortMetrics]:
    """
    Convenience function for time-based cohort analysis.

    Args:
        data: DataFrame with campaign data
        cohort_column: Column to cohort by
        value_column: Metric to analyze

    Returns:
        List of CohortMetrics
    """
    analyzer = CohortAnalyzer()
    return analyzer.analyze_by_time_cohort(data, cohort_column, value_column)


def compare_cohorts(
    data: pd.DataFrame,
    cohort_column: str,
    value_column: str,
    metric_name: Optional[str] = None
) -> CohortComparison:
    """
    Convenience function for cohort comparison.

    Args:
        data: DataFrame with campaign data
        cohort_column: Column defining cohorts
        value_column: Metric to compare
        metric_name: Human-readable metric name

    Returns:
        CohortComparison with statistical analysis
    """
    analyzer = CohortAnalyzer()
    return analyzer.compare_cohorts(data, cohort_column, value_column, metric_name)


def analyze_retention(
    data: pd.DataFrame,
    user_column: str,
    date_column: str
) -> RetentionMetrics:
    """
    Convenience function for retention analysis.

    Args:
        data: DataFrame with user activity data
        user_column: User identifier column
        date_column: Date column for activity

    Returns:
        RetentionMetrics
    """
    analyzer = CohortAnalyzer()
    return analyzer.analyze_retention(data, user_column, date_column)
