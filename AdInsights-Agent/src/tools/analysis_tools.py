"""
Analysis Tools for AdInsights-Agent

LangChain tools for autonomous analysis of AdTech campaign data.
Each tool is decorated with @tool for LangGraph agent integration.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

# Configure matplotlib
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


# =============================================================================
# MOCK DATA GENERATORS
# =============================================================================

HEALTHCARE_BENCHMARKS = {
    "healthcare_pharma": {
        "ctr": {"median": 1.2, "p25": 0.8, "p75": 1.8},
        "cvr": {"median": 2.5, "p25": 1.5, "p75": 4.0},
        "cpa": {"median": 350, "p25": 150, "p75": 600},  # Lower is better
        "roi": {"median": 2.8, "p25": 1.8, "p75": 4.2},
    },
    "healthcare_hospitals": {
        "ctr": {"median": 1.8, "p25": 1.2, "p75": 2.5},
        "cvr": {"median": 3.5, "p25": 2.0, "p75": 5.5},
        "cpa": {"median": 200, "p25": 100, "p75": 400},
        "roi": {"median": 3.5, "p25": 2.2, "p75": 5.0},
    },
    "healthcare_telehealth": {
        "ctr": {"median": 2.2, "p25": 1.5, "p75": 3.0},
        "cvr": {"median": 4.0, "p25": 2.5, "p75": 6.0},
        "cpa": {"median": 120, "p25": 60, "p75": 200},
        "roi": {"median": 4.2, "p25": 2.8, "p75": 6.0},
    },
    "healthcare_insurance": {
        "ctr": {"median": 1.0, "p25": 0.6, "p75": 1.5},
        "cvr": {"median": 1.8, "p25": 1.0, "p75": 3.0},
        "cpa": {"median": 450, "p25": 250, "p75": 800},
        "roi": {"median": 2.2, "p25": 1.5, "p75": 3.5},
    },
}


def _generate_mock_campaign_data(
    campaign_id: str,
    start_date: str,
    end_date: str,
    campaign_type: str = "healthcare_pharma",
    base_ctr: float = 1.2,
    base_cvr: float = 2.5,
    base_cpa: float = 350,
    trend_factor: float = 0.0,
    volatility: float = 0.15,
    include_anomalies: bool = True,
) -> pd.DataFrame:
    """
    Generate realistic mock campaign data with trends and anomalies.

    Args:
        campaign_id: Campaign identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        campaign_type: Type of healthcare campaign
        base_ctr: Base click-through rate (%)
        base_cvr: Base conversion rate (%)
        base_cpa: Base cost per acquisition ($)
        trend_factor: Daily trend change (+/-)
        volatility: Daily volatility factor
        include_anomalies: Whether to include anomalies

    Returns:
        DataFrame with daily campaign metrics
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, freq="D")

    data = []
    impressions_base = np.random.randint(5000, 20000, size=len(dates))

    for i, date in enumerate(dates):
        # Apply trend
        trend_multiplier = 1 + (trend_factor * i / len(dates))

        # Add day-of-week seasonality (weekends lower)
        dow_factor = 1.0
        if date.weekday() >= 5:  # Weekend
            dow_factor = 0.75

        # Add random noise
        noise = np.random.normal(1, volatility)

        # Calculate metrics with realistic relationships
        impressions = int(impressions_base[i] * trend_multiplier * dow_factor * noise)
        impressions = max(impressions, 1000)  # Minimum floor

        # CTR with noise
        ctr = base_ctr * trend_multiplier * dow_factor * noise
        ctr = max(ctr, 0.2)  # Minimum floor

        clicks = int(impressions * ctr / 100)

        # CVR with slight inverse correlation to CTR (realistic)
        cvr = base_cvr * (1 / (1 + 0.1 * (ctr - base_ctr))) * noise
        cvr = max(cvr, 0.5)  # Minimum floor

        conversions = int(clicks * cvr / 100)

        # CPA with some volatility
        cpa = base_cpa * (1 + np.random.normal(0, 0.2))

        # Calculate derived metrics
        spend = conversions * cpa if conversions > 0 else np.random.randint(50, 200)

        # Add anomalies for specific dates
        if include_anomalies and np.random.random() < 0.05:  # 5% chance
            anomaly_type = np.random.choice(["spike", "drop", "normal"])
            if anomaly_type == "spike":
                clicks = int(clicks * 2.5)
                conversions = int(conversions * 2.0)
                spend = int(spend * 2.0)
            elif anomaly_type == "drop":
                clicks = int(clicks * 0.3)
                conversions = int(conversions * 0.4)
                spend = int(spend * 0.5)

        # Recalculate derived metrics
        ctr = (clicks / impressions * 100) if impressions > 0 else 0
        cvr = (conversions / clicks * 100) if clicks > 0 else 0
        cpa = (spend / conversions) if conversions > 0 else spend

        data.append({
            "date": date,
            "campaign_id": campaign_id,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "spend": round(spend, 2),
            "ctr": round(ctr, 2),
            "cvr": round(cvr, 2),
            "cpa": round(cpa, 2),
            "roi": round((conversions * 500 - spend) / spend, 2) if spend > 0 else 0,  # Assuming $500 value
        })

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# =============================================================================
# ANALYSIS TOOLS
# =============================================================================

@tool
def fetch_campaign_metrics(
    campaign_id: str,
    start_date: str,
    end_date: str,
    campaign_type: str = "healthcare_pharma",
) -> Dict[str, Any]:
    """
    Fetch daily campaign metrics for the specified date range.

    Args:
        campaign_id: Unique campaign identifier
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        campaign_type: Type of healthcare campaign (pharma, hospitals, telehealth, insurance)

    Returns:
        Dictionary containing:
        - data: List of daily metric records
        - summary: Statistical summary of metrics
        - date_range: Actual date range covered
        - record_count: Number of days

    Example:
        >>> result = fetch_campaign_metrics("CAMP-001", "2024-01-01", "2024-01-31")
        >>> print(result["summary"]["avg_ctr"])
    """
    try:
        # Generate mock data
        df = _generate_mock_campaign_data(
            campaign_id=campaign_id,
            start_date=start_date,
            end_date=end_date,
            campaign_type=campaign_type,
        )

        # Calculate summary statistics
        summary = {
            "avg_impressions": int(df["impressions"].mean()),
            "avg_clicks": int(df["clicks"].mean()),
            "avg_conversions": int(df["conversions"].mean()),
            "avg_spend": round(df["spend"].mean(), 2),
            "avg_ctr": round(df["ctr"].mean(), 2),
            "avg_cvr": round(df["cvr"].mean(), 2),
            "avg_cpa": round(df["cpa"].mean(), 2),
            "avg_roi": round(df["roi"].mean(), 2),
            "total_impressions": int(df["impressions"].sum()),
            "total_clicks": int(df["clicks"].sum()),
            "total_conversions": int(df["conversions"].sum()),
            "total_spend": round(df["spend"].sum(), 2),
        }

        return {
            "data": df.to_dict("records"),
            "summary": summary,
            "date_range": {
                "start": start_date,
                "end": end_date,
            },
            "record_count": len(df),
            "campaign_id": campaign_id,
            "campaign_type": campaign_type,
        }

    except Exception as e:
        return {
            "error": f"Failed to fetch campaign metrics: {str(e)}",
            "campaign_id": campaign_id,
        }


@tool
def calculate_period_comparison(
    campaign_id: str,
    metric: str,
    period1: Tuple[str, str],
    period2: Tuple[str, str],
    campaign_type: str = "healthcare_pharma",
) -> Dict[str, Any]:
    """
    Compare a specific metric between two time periods.

    Calculates the absolute and relative change between two periods,
    performs statistical significance testing, and provides interpretation.

    Args:
        campaign_id: Campaign identifier
        metric: Metric to compare (ctr, cvr, cpa, roi, spend, conversions, etc.)
        period1: First period as tuple (start_date, end_date)
        period2: Second period as tuple (start_date, end_date)
        campaign_type: Type of healthcare campaign

    Returns:
        Dictionary containing:
        - period1_avg: Average metric value in period 1
        - period2_avg: Average metric value in period 2
        - period1_std: Standard deviation in period 1
        - period2_std: Standard deviation in period 2
        - absolute_change: period2 - period1
        - percent_change: Percentage change
        - is_significant: Whether difference is statistically significant
        - p_value: P-value from t-test
        - confidence_interval: 95% CI for the difference
        - interpretation: Natural language summary

    Example:
        >>> result = calculate_period_comparison(
        ...     "CAMP-001", "ctr",
        ...     ("2024-01-01", "2024-01-15"),
        ...     ("2024-01-16", "2024-01-31")
        ... )
    """
    try:
        # Fetch data for both periods
        df1 = _generate_mock_campaign_data(
            campaign_id=campaign_id,
            start_date=period1[0],
            end_date=period1[1],
            campaign_type=campaign_type,
        )
        df2 = _generate_mock_campaign_data(
            campaign_id=campaign_id,
            start_date=period2[0],
            end_date=period2[1],
            campaign_type=campaign_type,
        )

        # Extract metric values
        values1 = df1[metric].values
        values2 = df2[metric].values

        # Calculate statistics
        mean1 = np.mean(values1)
        mean2 = np.mean(values2)
        std1 = np.std(values1, ddof=1)
        std2 = np.std(values2, ddof=1)

        # Calculate changes
        absolute_change = mean2 - mean1
        percent_change = (absolute_change / mean1 * 100) if mean1 != 0 else 0

        # Perform t-test for significance
        t_stat, p_value = stats.ttest_ind(values2, values1)
        is_significant = p_value < 0.05

        # Calculate confidence interval for difference
        se_diff = np.sqrt(std1**2 / len(values1) + std2**2 / len(values2))
        ci_lower = absolute_change - 1.96 * se_diff
        ci_upper = absolute_change + 1.96 * se_diff

        # Generate interpretation
        direction = "increased" if absolute_change > 0 else "decreased"
        significance = "significantly" if is_significant else "not significantly"

        if metric in ["ctr", "cvr", "roi"]:
            change_desc = f"{direction} by {abs(percent_change):.1f}%"
        elif metric in ["cpa"]:
            change_desc = f"{'increased' if absolute_change > 0 else 'decreased'} by ${abs(absolute_change):.2f}"
        else:
            change_desc = f"{direction} by {abs(percent_change):.1f}%"

        interpretation = (
            f"The {metric} {significance} {direction} from {mean1:.2f} in Period 1 "
            f"to {mean2:.2f} in Period 2 ({change_desc}). "
            f"{'The difference is statistically significant (p={p_value:.4f}).' if is_significant else f'The difference is not statistically significant (p={p_value:.4f}).'}"
        )

        return {
            "metric": metric,
            "period1": {
                "start": period1[0],
                "end": period1[1],
                "avg": round(mean1, 4),
                "std": round(std1, 4),
                "n": len(values1),
            },
            "period2": {
                "start": period2[0],
                "end": period2[1],
                "avg": round(mean2, 4),
                "std": round(std2, 4),
                "n": len(values2),
            },
            "absolute_change": round(absolute_change, 4),
            "percent_change": round(percent_change, 2),
            "is_significant": is_significant,
            "p_value": round(p_value, 4),
            "confidence_interval": {
                "lower": round(ci_lower, 4),
                "upper": round(ci_upper, 4),
            },
            "interpretation": interpretation,
        }

    except Exception as e:
        return {
            "error": f"Failed to calculate period comparison: {str(e)}",
            "metric": metric,
        }


@tool
def detect_anomalies(
    data: Dict[str, Any],
    metric: str,
    method: str = "zscore",
    threshold: float = 2.5,
) -> Dict[str, Any]:
    """
    Detect anomalies in campaign metrics using statistical methods.

    Supports multiple detection algorithms:
    - zscore: Standard score method (default)
    - iqr: Interquartile range method
    - isolation_forest: Isolation Forest algorithm

    Args:
        data: Campaign data dictionary from fetch_campaign_metrics
        metric: Metric to analyze (ctr, cvr, cpa, etc.)
        method: Detection method ('zscore', 'iqr', 'isolation_forest')
        threshold: Threshold for anomaly detection
                 (zscore: std deviations, iqr: multiplier, isolation_forest: contamination)

    Returns:
        Dictionary containing:
        - anomalies: List of anomaly records with date, value, severity, score
        - summary: Count of anomalies by severity
        - statistics: Data statistics (mean, std, min, max)
        - method: Detection method used
        - threshold: Threshold applied

    Example:
        >>> data = fetch_campaign_metrics("CAMP-001", "2024-01-01", "2024-01-31")
        >>> anomalies = detect_anomalies(data, "ctr", method="zscore", threshold=2.5)
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data["data"])

        if metric not in df.columns:
            return {"error": f"Metric '{metric}' not found in data"}

        values = df[metric].values
        dates = df["date"].values

        # Detect anomalies based on method
        anomaly_indices = []
        anomaly_scores = []

        if method == "zscore":
            # Z-score method
            z_scores = np.abs(stats.zscore(values))
            anomaly_indices = np.where(z_scores > threshold)[0]
            anomaly_scores = z_scores[anomaly_indices]

        elif method == "iqr":
            # Interquartile Range method
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            anomaly_mask = (values < lower_bound) | (values > upper_bound)
            anomaly_indices = np.where(anomaly_mask)[0]
            # Score based on distance from bounds
            anomaly_scores = []
            for idx in anomaly_indices:
                dist = min(abs(values[idx] - lower_bound), abs(values[idx] - upper_bound))
                anomaly_scores.append(dist / IQR)

        elif method == "isolation_forest":
            # Isolation Forest method
            from sklearn.ensemble import IsolationForest

            clf = IsolationForest(contamination=threshold / 10, random_state=42)
            outlier_labels = clf.fit_predict(values.reshape(-1, 1))
            anomaly_indices = np.where(outlier_labels == -1)[0]
            anomaly_scores = np.abs(clf.score_samples(values.reshape(-1, 1))[anomaly_indices])

        # Build anomaly records
        anomalies = []
        for idx, score in zip(anomaly_indices, anomaly_scores):
            value = values[idx]
            date = str(dates[idx])

            # Determine severity
            if method == "zscore":
                severity = "high" if score > 4 else ("medium" if score > 3 else "low")
            else:
                severity = "high" if score > np.percentile(anomaly_scores, 75) else "medium"

            anomalies.append({
                "date": date,
                "value": round(value, 4),
                "severity": severity,
                "score": round(float(score), 4),
                "metric": metric,
            })

        # Count by severity
        severity_counts = {}
        for anomaly in anomalies:
            sev = anomaly["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # Calculate statistics
        statistics = {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
            "median": round(float(np.median(values)), 4),
            "q25": round(float(np.percentile(values, 25)), 4),
            "q75": round(float(np.percentile(values, 75)), 4),
        }

        return {
            "anomalies": anomalies,
            "summary": {
                "total_anomalies": len(anomalies),
                "severity_counts": severity_counts,
                "anomaly_rate": round(len(anomalies) / len(values) * 100, 2),
            },
            "statistics": statistics,
            "method": method,
            "threshold": threshold,
            "metric": metric,
        }

    except Exception as e:
        return {
            "error": f"Failed to detect anomalies: {str(e)}",
            "metric": metric,
            "method": method,
        }


@tool
def analyze_trend(
    data: Dict[str, Any],
    metric: str,
    forecast_days: int = 7,
) -> Dict[str, Any]:
    """
    Analyze trends in campaign metrics and forecast future values.

    Performs linear regression to detect trends, calculates goodness of fit,
    and provides simple forecasts for the next N days.

    Args:
        data: Campaign data dictionary from fetch_campaign_metrics
        metric: Metric to analyze (ctr, cvr, cpa, etc.)
        forecast_days: Number of days to forecast (default: 7)

    Returns:
        Dictionary containing:
        - trend_direction: 'up', 'down', or 'flat'
        - slope: Daily change in metric
        - r_squared: Goodness of fit (0-1)
        - p_value: Statistical significance of trend
        - intercept: Y-intercept of trend line
        - forecast: Predicted values for next N days
        - trend_strength: 'strong', 'moderate', or 'weak'
        - interpretation: Natural language summary
        - trend_line_data: Points for plotting trend line

    Example:
        >>> data = fetch_campaign_metrics("CAMP-001", "2024-01-01", "2024-01-31")
        >>> trend = analyze_trend(data, "ctr", forecast_days=7)
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data["data"]).sort_values("date")
        df = df.reset_index(drop=True)

        if metric not in df.columns:
            return {"error": f"Metric '{metric}' not found in data"}

        # Prepare data for regression
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[metric].values

        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Get predictions and calculate R²
        y_pred = model.predict(X)
        r_squared = model.score(X, y)

        # Calculate p-value for trend significance
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            np.arange(len(df)), y
        )

        # Determine trend direction
        if abs(slope) < std_err:
            direction = "flat"
        elif slope > 0:
            direction = "up"
        else:
            direction = "down"

        # Determine trend strength
        if r_squared > 0.7:
            strength = "strong"
        elif r_squared > 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        # Generate forecast
        forecast_X = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
        forecast_y = model.predict(forecast_X)
        forecast_dates = pd.date_range(
            start=df["date"].iloc[-1] + timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )

        forecast = [
            {
                "date": str(date),
                "predicted_value": round(float(value), 4),
                "day_ahead": i + 1,
            }
            for i, (date, value) in enumerate(zip(forecast_dates, forecast_y))
        ]

        # Trend line data for plotting
        trend_line = [
            {"date": str(df["date"].iloc[i]), "trend_value": round(float(val), 4)}
            for i, val in enumerate(y_pred)
        ]

        # Generate interpretation
        direction_desc = f"{'upward' if direction == 'up' else 'downward' if direction == 'down' else 'flat'}"
        significance = "statistically significant" if p_value < 0.05 else "not statistically significant"

        daily_change = abs(slope)
        if metric in ["ctr", "cvr"]:
            change_desc = f"{daily_change:.3f} percentage points"
        elif metric in ["cpa", "spend"]:
            change_desc = f"${daily_change:.2f}"
        else:
            change_desc = f"{daily_change:.2f} units"

        interpretation = (
            f"The {metric} shows a {strength} {direction_desc} trend, "
            f"changing by {change_desc} per day. "
            f"The trend is {significance} (p={p_value:.4f}, R²={r_squared:.3f}). "
            f"Forecast for the next {forecast_days} days shows a projected "
            f"final value of {forecast[-1]['predicted_value']:.2f}."
        )

        return {
            "metric": metric,
            "trend_direction": direction,
            "slope": round(float(slope), 6),
            "slope_std_error": round(float(std_err), 6),
            "r_squared": round(float(r_squared), 4),
            "p_value": round(float(p_value), 4),
            "intercept": round(float(intercept), 4),
            "trend_strength": strength,
            "forecast": forecast,
            "trend_line_data": trend_line,
            "interpretation": interpretation,
            "data_points": len(df),
        }

    except Exception as e:
        return {
            "error": f"Failed to analyze trend: {str(e)}",
            "metric": metric,
        }


@tool
def find_correlations(
    data: Dict[str, Any],
    target_metric: str,
    min_correlation: float = 0.3,
) -> Dict[str, Any]:
    """
    Find correlations between target metric and other campaign metrics.

    Calculates Pearson and Spearman correlation coefficients with p-values.
    Returns sorted list of metrics most correlated with the target.

    Args:
        data: Campaign data dictionary from fetch_campaign_metrics
        target_metric: Metric to find correlations for
        min_correlation: Minimum absolute correlation to include (default: 0.3)

    Returns:
        Dictionary containing:
        - correlations: List of correlation records sorted by strength
        - strongest_positive: Metric with highest positive correlation
        - strongest_negative: Metric with highest negative correlation
        - summary: Quick stats about correlations found
        - method: Correlation method used (pearson, spearman)

    Example:
        >>> data = fetch_campaign_metrics("CAMP-001", "2024-01-01", "2024-01-31")
        >>> corr = find_correlations(data, "roi", min_correlation=0.3)
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data["data"])

        if target_metric not in df.columns:
            return {"error": f"Metric '{target_metric}' not found in data"}

        # Numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_metric not in numeric_cols:
            return {"error": f"Target metric '{target_metric}' is not numeric"}

        # Remove target from predictor list
        predictor_cols = [col for col in numeric_cols if col != target_metric and col != "date"]

        correlations = []
        for col in predictor_cols:
            # Remove NaN values
            mask = ~(df[col].isna() | df[target_metric].isna())
            x = df.loc[mask, col].values
            y = df.loc[mask, target_metric].values

            if len(x) < 3:  # Need at least 3 points
                continue

            # Calculate Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(x, y)

            # Calculate Spearman correlation
            spearman_r, spearman_p = stats.spearmanr(x, y)

            # Only include if meets threshold
            if abs(pearson_r) >= min_correlation:
                correlations.append({
                    "metric": col,
                    "pearson_r": round(float(pearson_r), 4),
                    "pearson_p": round(float(pearson_p), 4),
                    "spearman_r": round(float(spearman_r), 4),
                    "spearman_p": round(float(spearman_p), 4),
                    "is_significant": pearson_p < 0.05,
                    "direction": "positive" if pearson_r > 0 else "negative",
                    "strength": (
                        "strong" if abs(pearson_r) > 0.7 else
                        "moderate" if abs(pearson_r) > 0.5 else
                        "weak"
                    ),
                })

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

        # Find strongest positive/negative
        positive_corrs = [c for c in correlations if c["direction"] == "positive"]
        negative_corrs = [c for c in correlations if c["direction"] == "negative"]

        strongest_positive = positive_corrs[0] if positive_corrs else None
        strongest_negative = negative_corrs[0] if negative_corrs else None

        return {
            "target_metric": target_metric,
            "correlations": correlations,
            "strongest_positive": strongest_positive,
            "strongest_negative": strongest_negative,
            "summary": {
                "total_correlations": len(correlations),
                "significant_correlations": sum(1 for c in correlations if c["is_significant"]),
                "positive_correlations": len(positive_corrs),
                "negative_correlations": len(negative_corrs),
            },
            "min_correlation_threshold": min_correlation,
            "method": "pearson_and_spearman",
        }

    except Exception as e:
        return {
            "error": f"Failed to find correlations: {str(e)}",
            "target_metric": target_metric,
        }


@tool
def compare_to_benchmark(
    campaign_metrics: Dict[str, Any],
    industry: str = "healthcare_pharma",
) -> Dict[str, Any]:
    """
    Compare campaign performance to industry benchmarks.

    Uses healthcare AdTech industry benchmarks for different campaign types.
    Calculates percentile ranking and performance categorization.

    Args:
        campaign_metrics: Dictionary with avg_ctr, avg_cvr, avg_cpa, avg_roi
        industry: Industry segment (healthcare_pharma, healthcare_hospitals,
                  healthcare_telehealth, healthcare_insurance)

    Returns:
        Dictionary containing:
        - comparisons: List of metric comparisons to benchmarks
        - overall_performance: Summary of overall performance
        - percentile_ranking: Average percentile across all metrics
        - recommendations: Suggestions for improvement

    Example:
        >>> metrics = {"avg_ctr": 1.8, "avg_cvr": 3.2, "avg_cpa": 280, "avg_roi": 3.5}
        >>> comparison = compare_to_benchmark(metrics, "healthcare_pharma")
    """
    try:
        # Get benchmarks for industry
        if industry not in HEALTHCARE_BENCHMARKS:
            available = ", ".join(HEALTHCARE_BENCHMARKS.keys())
            return {
                "error": f"Unknown industry '{industry}'. Available: {available}"
            }

        benchmarks = HEALTHCARE_BENCHMARKS[industry]

        comparisons = []
        total_percentile = 0
        metric_count = 0

        for metric_name, campaign_value in campaign_metrics.items():
            if metric_name not in benchmarks:
                continue

            bench = benchmarks[metric_name]
            metric_count += 1

            # Determine performance
            if metric_name == "cpa":
                # For CPA, lower is better
                if campaign_value < bench["p25"]:
                    performance = "excellent"
                    percentile = 85
                elif campaign_value < bench["median"]:
                    performance = "good"
                    percentile = 65
                elif campaign_value < bench["p75"]:
                    performance = "average"
                    percentile = 40
                else:
                    performance = "below_average"
                    percentile = 15
            else:
                # For other metrics, higher is better
                if campaign_value > bench["p75"]:
                    performance = "excellent"
                    percentile = 85
                elif campaign_value > bench["median"]:
                    performance = "good"
                    percentile = 65
                elif campaign_value > bench["p25"]:
                    performance = "average"
                    percentile = 40
                else:
                    performance = "below_average"
                    percentile = 15

            total_percentile += percentile

            # Calculate difference from median
            diff = campaign_value - bench["median"]
            diff_pct = (diff / bench["median"] * 100) if bench["median"] != 0 else 0

            comparisons.append({
                "metric": metric_name.upper(),
                "campaign_value": round(campaign_value, 2),
                "benchmark_median": round(bench["median"], 2),
                "benchmark_p25": round(bench["p25"], 2),
                "benchmark_p75": round(bench["p75"], 2),
                "performance": performance,
                "percentile": percentile,
                "difference_from_median": round(diff, 2),
                "difference_percent": round(diff_pct, 1),
            })

        # Calculate overall
        avg_percentile = total_percentile / metric_count if metric_count > 0 else 0

        if avg_percentile >= 75:
            overall = "excellent"
        elif avg_percentile >= 55:
            overall = "good"
        elif avg_percentile >= 35:
            overall = "average"
        else:
            overall = "below_average"

        # Generate recommendations
        recommendations = []
        for comp in comparisons:
            if comp["performance"] in ["below_average", "average"]:
                if comp["metric"] == "CTR":
                    recommendations.append(
                        "Improve ad creative and targeting to increase click-through rate. "
                        "Consider A/B testing different headlines and images."
                    )
                elif comp["metric"] == "CVR":
                    recommendations.append(
                        "Optimize landing pages and conversion funnels. "
                        "Ensure clear call-to-actions and reduce friction points."
                    )
                elif comp["metric"] == "CPA":
                    recommendations.append(
                        "Refine audience targeting and bid strategies to reduce cost per acquisition. "
                        "Focus on high-performing segments."
                    )
                elif comp["metric"] == "ROI":
                    recommendations.append(
                        "Reallocate budget to top-performing campaigns and channels. "
                        "Pause underperforming creatives and audiences."
                    )

        return {
            "industry": industry,
            "comparisons": comparisons,
            "overall_performance": overall,
            "percentile_ranking": round(avg_percentile, 1),
            "summary": {
                "total_metrics_compared": metric_count,
                "excellent_metrics": sum(1 for c in comparisons if c["performance"] == "excellent"),
                "good_metrics": sum(1 for c in comparisons if c["performance"] == "good"),
                "average_metrics": sum(1 for c in comparisons if c["performance"] == "average"),
                "below_average_metrics": sum(1 for c in comparisons if c["performance"] == "below_average"),
            },
            "recommendations": recommendations[:3],  # Top 3
        }

    except Exception as e:
        return {
            "error": f"Failed to compare to benchmark: {str(e)}",
            "industry": industry,
        }


@tool
def generate_chart(
    data: Dict[str, Any],
    chart_type: str,
    metrics: List[str],
    title: str,
    output_dir: str = "./data/charts",
) -> Dict[str, Any]:
    """
    Generate a chart from campaign data and save to file.

    Supports multiple chart types with customizable styling.
    Charts are saved as high-resolution PNG files.

    Args:
        data: Campaign data dictionary from fetch_campaign_metrics
        chart_type: Type of chart ('line', 'bar', 'scatter', 'heatmap', 'box')
        metrics: List of metric names to include in chart
        title: Chart title
        output_dir: Directory to save chart (default: ./data/charts)

    Returns:
        Dictionary containing:
        - file_path: Full path to saved chart
        - file_name: Name of saved file
        - chart_type: Type of chart generated
        - metrics: Metrics included in chart
        - file_size: Size of file in bytes
        - success: True if chart was generated successfully

    Example:
        >>> data = fetch_campaign_metrics("CAMP-001", "2024-01-01", "2024-01-31")
        >>> chart = generate_chart(data, "line", ["ctr", "cvr"], "Campaign Performance")
        >>> print(f"Chart saved to: {chart['file_path']}")
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data["data"])

        # Validate metrics
        valid_metrics = [m for m in metrics if m in df.columns]
        if not valid_metrics:
            return {
                "error": f"No valid metrics found. Available: {df.columns.tolist()}",
                "requested_metrics": metrics,
            }

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        chart_id = uuid.uuid4().hex[:8]
        file_name = f"chart_{chart_type}_{chart_id}.png"
        file_path = output_path / file_name

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        if chart_type == "line":
            # Line chart for time series
            df = df.sort_values("date")
            x = pd.to_datetime(df["date"])

            for i, metric in enumerate(valid_metrics):
                ax.plot(x, df[metric], marker="o", linewidth=2, markersize=4,
                       label=metric.upper(), alpha=0.8)

            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.legend(loc="best")
            plt.xticks(rotation=45)

        elif chart_type == "bar":
            # Bar chart for comparison
            x = np.arange(len(df))
            width = 0.8 / len(valid_metrics)

            for i, metric in enumerate(valid_metrics):
                offset = (i - len(valid_metrics) / 2) * width
                ax.bar(x + offset, df[metric], width, label=metric.upper(), alpha=0.8)

            ax.set_xlabel("Day")
            ax.set_ylabel("Value")
            ax.legend(loc="best")

        elif chart_type == "scatter":
            # Scatter plot for correlations
            if len(valid_metrics) < 2:
                return {"error": "Scatter plot requires at least 2 metrics"}

            x_metric = valid_metrics[0]
            y_metric = valid_metrics[1]

            scatter = ax.scatter(df[x_metric], df[y_metric], c=range(len(df)),
                               cmap="viridis", alpha=0.6, s=50, edgecolors="black")

            ax.set_xlabel(x_metric.upper())
            ax.set_ylabel(y_metric.upper())
            ax.set_title(f"{y_metric.upper()} vs {x_metric.upper()}")
            plt.colorbar(scatter, ax=ax, label="Day")

        elif chart_type == "heatmap":
            # Heatmap for correlation matrix
            corr_df = df[valid_metrics].corr()
            im = ax.imshow(corr_df, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)

            ax.set_xticks(np.arange(len(valid_metrics)))
            ax.set_yticks(np.arange(len(valid_metrics)))
            ax.set_xticklabels([m.upper() for m in valid_metrics])
            ax.set_yticklabels([m.upper() for m in valid_metrics])

            # Add correlation values
            for i in range(len(valid_metrics)):
                for j in range(len(valid_metrics)):
                    text = ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}",
                                 ha="center", va="center", color="black", fontsize=10)

            plt.colorbar(im, ax=ax, label="Correlation")
            title = "Correlation Matrix"

        elif chart_type == "box":
            # Box plot for distributions
            box_data = [df[metric].values for metric in valid_metrics]
            bp = ax.boxplot(box_data, labels=[m.upper() for m in valid_metrics],
                           patch_artist=True, showmeans=True)

            # Color boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(valid_metrics)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)

            ax.set_ylabel("Value")
            title = "Distribution by Metric"

        else:
            return {
                "error": f"Unknown chart type '{chart_type}'. "
                        f"Available: line, bar, scatter, heatmap, box"
            }

        # Styling
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save chart
        fig.savefig(str(file_path), dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Get file size
        file_size = file_path.stat().st_size

        return {
            "file_path": str(file_path.absolute()),
            "file_name": file_name,
            "chart_type": chart_type,
            "metrics": valid_metrics,
            "title": title,
            "file_size": file_size,
            "success": True,
        }

    except Exception as e:
        return {
            "error": f"Failed to generate chart: {str(e)}",
            "chart_type": chart_type,
            "metrics": metrics,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_available_tools() -> List[Dict[str, str]]:
    """
    Get list of all available analysis tools with descriptions.

    Returns:
        List of tool dictionaries with name and description
    """
    tools_info = [
        {
            "name": "fetch_campaign_metrics",
            "description": "Fetch daily campaign metrics for a date range",
        },
        {
            "name": "calculate_period_comparison",
            "description": "Compare metrics between two time periods",
        },
        {
            "name": "detect_anomalies",
            "description": "Detect anomalies using statistical methods",
        },
        {
            "name": "analyze_trend",
            "description": "Analyze trends and forecast future values",
        },
        {
            "name": "find_correlations",
            "description": "Find correlations between metrics",
        },
        {
            "name": "compare_to_benchmark",
            "description": "Compare performance to industry benchmarks",
        },
        {
            "name": "generate_chart",
            "description": "Generate and save visualization charts",
        },
    ]

    return tools_info


if __name__ == "__main__":
    # Example usage
    print("AdInsights-Agent Analysis Tools")
    print("=" * 50)

    # Fetch campaign metrics
    result = fetch_campaign_metrics.invoke({
        "campaign_id": "CAMP-DEMO-001",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "campaign_type": "healthcare_pharma",
    })

    print(f"\nFetched {result['record_count']} days of data")
    print(f"Summary: {result['summary']}")

    # Detect anomalies
    anomalies = detect_anomalies.invoke({
        "data": result,
        "metric": "ctr",
        "method": "zscore",
        "threshold": 2.5,
    })

    print(f"\nDetected {anomalies['summary']['total_anomalies']} anomalies")

    # Analyze trend
    trend = analyze_trend.invoke({
        "data": result,
        "metric": "ctr",
        "forecast_days": 7,
    })

    print(f"\nTrend: {trend['trend_direction']} ({trend['trend_strength']})")
    print(f"Interpretation: {trend['interpretation']}")
