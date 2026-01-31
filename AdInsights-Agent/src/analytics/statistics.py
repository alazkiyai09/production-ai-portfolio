"""
Statistical Analytics Module for AdInsights-Agent

Provides statistical functions for time series analysis, forecasting,
and A/B testing in the context of AdTech campaign analytics.
"""

import warnings
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# =============================================================================
# MOVING AVERAGE
# =============================================================================

def calculate_moving_average(
    series: pd.Series,
    window: int,
    method: str = "simple",
    center: bool = False,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Calculate moving average of a time series.

    Formula (Simple Moving Average):
    SMA_t = (1/n) * Σ(x_i) for i from t-n+1 to t

    Formula (Exponential Moving Average):
    EMA_t = α * x_t + (1-α) * EMA_{t-1}
    where α = 2 / (window + 1)

    Args:
        series: Input time series
        window: Window size for moving average
        method: 'simple' or 'exponential'
        center: Whether to center the window (for simple MA only)
        min_periods: Minimum periods required (defaults to window)

    Returns:
        Series with moving average values (same length as input)

    Raises:
        ValueError: If window size is invalid or series is too short

    Examples:
        >>> series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> ma = calculate_moving_average(series, window=3, method="simple")
        >>> print(ma)
        0    NaN
        1    NaN
        2    2.0
        3    3.0
        ...
    """
    # Validate inputs
    if not isinstance(series, (pd.Series, list, np.ndarray)):
        raise TypeError("series must be a pandas Series, list, or numpy array")

    if isinstance(series, (list, np.ndarray)):
        series = pd.Series(series)

    if window <= 0:
        raise ValueError("window must be positive")

    if window > len(series):
        raise ValueError(f"window ({window}) cannot be larger than series length ({len(series)})")

    if len(series) == 0:
        return pd.Series([], dtype=float)

    # Handle missing values
    clean_series = series.copy()

    if min_periods is None:
        min_periods = window if method == "simple" else 1

    # Calculate moving average based on method
    if method == "simple":
        result = clean_series.rolling(
            window=window,
            center=center,
            min_periods=min_periods
        ).mean()

    elif method == "exponential":
        result = clean_series.ewm(
            span=window,
            min_periods=min_periods
        ).mean()

    elif method == "weighted":
        # Linearly weighted moving average
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()

        result = clean_series.rolling(
            window=window,
            min_periods=min_periods
        ).apply(
            lambda x: np.dot(x, weights[-len(x):]) if len(x) > 0 else np.nan,
            raw=True
        )

    else:
        raise ValueError(f"Unknown method: {method}. Use 'simple', 'exponential', or 'weighted'")

    return result


# =============================================================================
# CHANGEPOINT DETECTION
# =============================================================================

def detect_changepoints(
    series: pd.Series,
    min_size: int = 7,
    method: str = "mean_shift",
    significance_level: float = 0.05,
    min_change_pct: float = 0.1,
) -> List[Dict[str, any]]:
    """
    Detect significant changepoints in a time series.

    Uses statistical tests to identify points where the time series
    exhibits significant changes in mean, variance, or trend.

    Methods:
    - mean_shift: Detects shifts in mean using sliding window t-tests
    - variance_change: Detects changes in variance using F-tests
    - trend_change: Detects changes in linear trend

    Args:
        series: Input time series
        min_size: Minimum segment size between changepoints
        method: Detection method ('mean_shift', 'variance_change', 'trend_change')
        significance_level: Alpha level for statistical tests
        min_change_pct: Minimum percent change to be considered

    Returns:
        List of changepoint dictionaries with keys:
        - index: Index of changepoint in series
        - date: Date value (if series has datetime index)
        - direction: 'up' or 'down'
        - magnitude: Size of the change
        - p_value: Statistical significance
        - confidence: Confidence level (1 - p_value)

    Examples:
        >>> series = pd.Series([10, 11, 10, 12, 20, 21, 19, 22, 10, 11, 10])
        >>> changepoints = detect_changepoints(series, min_size=3)
        >>> print(changepoints)
        [{'index': 4, 'direction': 'up', 'magnitude': 10.0, ...}]
    """
    # Validate inputs
    if not isinstance(series, (pd.Series, list, np.ndarray)):
        raise TypeError("series must be a pandas Series, list, or numpy array")

    if isinstance(series, (list, np.ndarray)):
        series = pd.Series(series)

    if len(series) < min_size * 2:
        return []  # Not enough data for changepoint detection

    if series.isna().all():
        return []

    # Remove NaN values for analysis
    valid_mask = series.notna()
    clean_series = series[valid_mask]
    original_indices = np.where(valid_mask)[0]

    if len(clean_series) < min_size * 2:
        return []

    changepoints = []

    if method == "mean_shift":
        # Sliding window approach to detect mean shifts
        for i in range(min_size, len(clean_series) - min_size + 1):
            # Get before and after windows
            before = clean_series.iloc[i - min_size:i].values
            after = clean_series.iloc[i:i + min_size].values

            # Skip if either window is constant
            if np.std(before) < 1e-10 or np.std(after) < 1e-10:
                continue

            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(before, after)

            # Calculate magnitude
            mean_before = np.mean(before)
            mean_after = np.mean(after)
            change = mean_after - mean_before

            # Check if change meets threshold and is significant
            change_pct = abs(change) / (abs(mean_before) + 1e-10)
            if p_value < significance_level and change_pct >= min_change_pct:
                original_idx = original_indices[i]

                changepoint = {
                    "index": int(original_idx),
                    "date": _get_date_value(series, original_idx),
                    "direction": "up" if change > 0 else "down",
                    "magnitude": round(float(change), 4),
                    "magnitude_pct": round(float(change_pct * 100), 2),
                    "p_value": round(float(p_value), 4),
                    "confidence": round(1 - p_value, 4),
                    "before_mean": round(float(mean_before), 4),
                    "after_mean": round(float(mean_after), 4),
                }
                changepoints.append(changepoint)

    elif method == "variance_change":
        # Detect changes in variance using F-test
        for i in range(min_size, len(clean_series) - min_size + 1):
            before = clean_series.iloc[i - min_size:i].values
            after = clean_series.iloc[i:i + min_size].values

            # Calculate variances
            var_before = np.var(before, ddof=1)
            var_after = np.var(after, ddof=1)

            # Skip if either variance is near zero
            if var_before < 1e-10 or var_after < 1e-10:
                continue

            # F-test for equality of variances
            f_stat = var_after / var_before if var_after >= var_before else var_before / var_after
            df1 = len(before) - 1
            df2 = len(after) - 1
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)

            if p_value < significance_level:
                original_idx = original_indices[i]
                change_pct = abs(var_after - var_before) / (var_before + 1e-10)

                changepoint = {
                    "index": int(original_idx),
                    "date": _get_date_value(series, original_idx),
                    "direction": "increase" if var_after > var_before else "decrease",
                    "magnitude": round(float(var_after - var_before), 4),
                    "magnitude_pct": round(float(change_pct * 100), 2),
                    "p_value": round(float(p_value), 4),
                    "confidence": round(1 - p_value, 4),
                    "before_variance": round(float(var_before), 4),
                    "after_variance": round(float(var_after), 4),
                }
                changepoints.append(changepoint)

    elif method == "trend_change":
        # Detect changes in linear trend
        for i in range(min_size, len(clean_series) - min_size + 1):
            before_x = np.arange(len(clean_series.iloc[i - min_size:i]))
            before_y = clean_series.iloc[i - min_size:i].values
            after_x = np.arange(len(clean_series.iloc[i:i + min_size]))
            after_y = clean_series.iloc[i:i + min_size].values

            # Fit linear models
            try:
                slope_before, _, _, _, _ = stats.linregress(before_x, before_y)
                slope_after, _, _, _, _ = stats.linregress(after_x, after_y)

                # Test if slopes are significantly different
                # Using Fisher's z transformation
                n1, n2 = len(before_y), len(after_y)
                se1 = np.std(before_y) / np.sqrt(len(before_y))
                se2 = np.std(after_y) / np.sqrt(len(after_y))

                if se1 > 1e-10 and se2 > 1e-10:
                    z_score = (slope_after - slope_before) / np.sqrt(se1**2 + se2**2)
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

                    if p_value < significance_level:
                        original_idx = original_indices[i]
                        change = slope_after - slope_before
                        change_pct = abs(change) / (abs(slope_before) + 1e-10)

                        if change_pct >= min_change_pct:
                            changepoint = {
                                "index": int(original_idx),
                                "date": _get_date_value(series, original_idx),
                                "direction": "up" if change > 0 else "down",
                                "magnitude": round(float(change), 4),
                                "magnitude_pct": round(float(change_pct * 100), 2),
                                "p_value": round(float(p_value), 4),
                                "confidence": round(1 - p_value, 4),
                                "before_slope": round(float(slope_before), 4),
                                "after_slope": round(float(slope_after), 4),
                            }
                            changepoints.append(changepoint)
            except Exception:
                continue

    # Sort by index and remove duplicates (within min_size)
    changepoints.sort(key=lambda x: x["index"])

    # Filter out changepoints that are too close
    filtered = []
    for cp in changepoints:
        if not filtered or cp["index"] - filtered[-1]["index"] >= min_size:
            filtered.append(cp)

    return filtered


def _get_date_value(series: pd.Series, index: int) -> Optional[str]:
    """Extract date value from series at index if available."""
    try:
        if isinstance(series.index, pd.DatetimeIndex):
            return str(series.index[index])
        elif hasattr(series.index[index], 'isoformat'):
            return series.index[index].isoformat()
        else:
            return None
    except Exception:
        return None


# =============================================================================
# SEASONAL DECOMPOSITION
# =============================================================================

def seasonal_decomposition(
    series: pd.Series,
    period: int = 7,
    model: str = "additive",
    extrapolate_trend: str = "freq",
) -> Dict[str, Union[pd.Series, Dict[str, float]]]:
    """
    Decompose time series into trend, seasonal, and residual components.

    Additive Model: Y_t = Trend_t + Seasonal_t + Residual_t
    Multiplicative Model: Y_t = Trend_t × Seasonal_t × Residual_t

    Uses statsmodels seasonal_decompose with moving averages.

    Args:
        series: Input time series (must have >= 2 * period observations)
        period: Period for seasonal decomposition (e.g., 7 for weekly)
        model: 'additive' or 'multiplicative'
        extrapolate_trend: How to handle trend at boundaries ('freq' or None)

    Returns:
        Dictionary with keys:
        - observed: Original series
        - trend: Long-term trend component
        - seasonal: Repeating seasonal component
        - residual: Random noise/residuals
        - stats: Dictionary with strength of trend and seasonality

    Raises:
        ValueError: If series is too short or contains invalid values

    Examples:
        >>> series = pd.Series([10, 20, 15, 25, 12, 22, 17, 27, 11, 21])
        >>> result = seasonal_decomposition(series, period=5)
        >>> print(result['trend'])
        >>> print(result['stats']['trend_strength'])
    """
    # Validate inputs
    if not isinstance(series, (pd.Series, list, np.ndarray)):
        raise TypeError("series must be a pandas Series, list, or numpy array")

    if isinstance(series, (list, np.ndarray)):
        series = pd.Series(series)

    if len(series) < 2 * period:
        raise ValueError(
            f"Series length ({len(series)}) must be at least 2 * period ({2 * period})"
        )

    if period < 2:
        raise ValueError("period must be at least 2")

    # Handle missing values
    clean_series = series.copy()

    if clean_series.isna().all():
        raise ValueError("Series contains only NaN values")

    # Forward fill NaN values (required for decomposition)
    clean_series = clean_series.ffill().bfill()

    if clean_series.isna().any() or (clean_series == 0).all():
        # If still have issues, add small constant
        if model == "multiplicative":
            clean_series = clean_series.replace(0, np.nan).ffill()
            clean_series = clean_series + clean_series.mean() * 0.01

    try:
        # Perform decomposition
        decomposition = seasonal_decompose(
            clean_series,
            model=model,
            period=period,
            extrapolate_trend=extrapolate_trend
        )

        # Calculate strength of components
        # Strength of trend: Var(Trend) / Var(Observed)
        # Strength of seasonality: Var(Seasonal) / Var(Residual + Seasonal)

        obs_var = clean_series.var()
        trend_var = decomposition.trend.var()
        resid_var = decomposition.resid.var()
        seas_var = decomposition.seasonal.var()

        # Handle edge cases
        if obs_var is None or obs_var == 0 or pd.isna(obs_var):
            trend_strength = 0.0
        else:
            trend_strength = max(0, min(1, trend_var / obs_var))

        seas_plus_resid_var = seas_var + resid_var
        if seas_plus_resid_var is None or seas_plus_resid_var == 0 or pd.isna(seas_plus_resid_var):
            seasonal_strength = 0.0
        else:
            seasonal_strength = max(0, min(1, seas_var / seas_plus_resid_var))

        stats = {
            "trend_strength": round(float(trend_strength), 4),
            "seasonal_strength": round(float(seasonal_strength), 4),
            "residual_variance": round(float(resid_var), 4) if resid_var is not None else 0.0,
            "period": period,
            "model": model,
        }

        return {
            "observed": decomposition.observed,
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "residual": decomposition.resid,
            "stats": stats,
        }

    except Exception as e:
        raise ValueError(f"Decomposition failed: {str(e)}")


# =============================================================================
# FORECASTING
# =============================================================================

def forecast_metric(
    series: pd.Series,
    periods: int = 7,
    method: str = "exponential_smoothing",
    confidence_level: float = 0.95,
    **kwargs
) -> Dict[str, any]:
    """
    Forecast future values of a time series.

    Methods:
    - exponential_smoothing: Simple Exponential Smoothing (SES)
    - holt: Holt's linear trend method
    - moving_average: Forecast using moving average
    - naive: Naive forecast (last value repeats)

    Formula (Simple Exponential Smoothing):
    F_{t+1} = α * Y_t + (1-α) * F_t
    where α is the smoothing parameter

    Args:
        series: Historical time series data
        periods: Number of periods to forecast
        method: Forecasting method
        confidence_level: Confidence level for intervals (0-1)
        **kwargs: Additional parameters for specific methods

    Returns:
        Dictionary with keys:
        - forecast: Array of forecasted values
        - lower_bound: Lower confidence bound
        - upper_bound: Upper confidence bound
        - method: Method used
        - fitted_values: In-sample fitted values
        - mse: Mean squared error of fit
        - parameters: Model parameters used

    Examples:
        >>> series = pd.Series([10, 12, 11, 13, 12, 14, 13, 15])
        >>> forecast = forecast_metric(series, periods=3, method="exponential_smoothing")
        >>> print(forecast['forecast'])
        [15.2, 15.2, 15.2]
    """
    # Validate inputs
    if not isinstance(series, (pd.Series, list, np.ndarray)):
        raise TypeError("series must be a pandas Series, list, or numpy array")

    if isinstance(series, (list, np.ndarray)):
        series = pd.Series(series)

    if len(series) < 3:
        raise ValueError(f"Series length ({len(series)}) must be at least 3")

    if periods <= 0:
        raise ValueError("periods must be positive")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    # Handle missing values
    clean_series = series.ffill().bfill()

    if clean_series.isna().all():
        raise ValueError("Series contains only NaN values")

    # Handle constant series
    if clean_series.std() < 1e-10:
        # All values are the same
        constant_value = clean_series.iloc[0]
        se = 0  # No error for constant series
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

        return {
            "forecast": np.array([constant_value] * periods),
            "lower_bound": np.array([constant_value] * periods),
            "upper_bound": np.array([constant_value] * periods),
            "method": method,
            "fitted_values": clean_series.values,
            "mse": 0.0,
            "parameters": {"note": "constant series"},
        }

    try:
        if method == "exponential_smoothing":
            # Simple Exponential Smoothing
            model = SimpleExpSmoothing(clean_series, initialization_method="estimated")
            fit = model.fit(optimized=True, **kwargs)

            # Generate forecast
            forecast = fit.forecast(steps=periods)

            # Calculate prediction intervals
            residuals = clean_series - fit.fittedvalues
            se = np.sqrt(np.sum(residuals**2) / (len(clean_series) - 1))

            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            margin = z_score * se * np.sqrt(np.arange(1, periods + 1))

            lower_bound = forecast - margin
            upper_bound = forecast + margin

            return {
                "forecast": forecast.values,
                "lower_bound": lower_bound.values,
                "upper_bound": upper_bound.values,
                "method": "exponential_smoothing",
                "fitted_values": fit.fittedvalues.values,
                "mse": float(fit.mse),
                "parameters": {
                    "smoothing_level": float(fit.params["smoothing_level"]),
                    "initial_level": float(fit.params["initial_level"]),
                },
            }

        elif method == "holt":
            # Holt's linear trend method
            if len(clean_series) < 4:
                raise ValueError("Holt's method requires at least 4 observations")

            model = Holt(clean_series, initialization_method="estimated")
            fit = model.fit(optimized=True, **kwargs)

            forecast = fit.forecast(steps=periods)

            # Calculate prediction intervals
            residuals = clean_series - fit.fittedvalues
            se = np.sqrt(np.sum(residuals**2) / (len(clean_series) - 2))

            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            margin = z_score * se * np.sqrt(np.arange(1, periods + 1))

            lower_bound = forecast - margin
            upper_bound = forecast + margin

            return {
                "forecast": forecast.values,
                "lower_bound": lower_bound.values,
                "upper_bound": upper_bound.values,
                "method": "holt",
                "fitted_values": fit.fittedvalues.values,
                "mse": float(fit.mse),
                "parameters": {
                    "smoothing_level": float(fit.params["smoothing_level"]),
                    "smoothing_trend": float(fit.params["smoothing_trend"]),
                    "initial_level": float(fit.params["initial_level"]),
                    "initial_trend": float(fit.params["initial_trend"]),
                },
            }

        elif method == "moving_average":
            # Moving average forecast
            window = kwargs.get("window", min(7, len(clean_series) // 2))
            ma = calculate_moving_average(clean_series, window=window, method="simple")
            last_ma = ma.iloc[-1]

            # Forecast is constant at last MA value
            forecast = np.array([last_ma] * periods)

            # Calculate prediction intervals
            residuals = clean_series - ma
            residuals = residuals.dropna()
            se = np.std(residuals)

            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            margin = z_score * se * np.sqrt(np.arange(1, periods + 1))

            lower_bound = forecast - margin
            upper_bound = forecast + margin

            return {
                "forecast": forecast,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "method": "moving_average",
                "fitted_values": ma.values,
                "mse": float(np.mean(residuals**2)),
                "parameters": {"window": window},
            }

        elif method == "naive":
            # Naive forecast (last value repeats)
            last_value = clean_series.iloc[-1]

            forecast = np.array([last_value] * periods)

            # Calculate prediction intervals based on historical changes
            changes = clean_series.diff().dropna()
            se = np.std(changes)

            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            margin = z_score * se * np.sqrt(np.arange(1, periods + 1))

            lower_bound = forecast - margin
            upper_bound = forecast + margin

            return {
                "forecast": forecast,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "method": "naive",
                "fitted_values": clean_series.shift(1).fillna(clean_series.iloc[0]).values,
                "mse": float(np.mean(changes**2)),
                "parameters": {},
            }

        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Use 'exponential_smoothing', 'holt', 'moving_average', or 'naive'"
            )

    except Exception as e:
        raise ValueError(f"Forecasting failed: {str(e)}")


# =============================================================================
# A/B TESTING
# =============================================================================

def ab_test_significance(
    control: pd.Series,
    treatment: pd.Series,
    metric_type: str = "continuous",
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
) -> Dict[str, any]:
    """
    Calculate statistical significance of A/B test results.

    For continuous metrics (e.g., revenue, session duration):
    - Uses two-sample t-test (independent samples)
    - Assumes unequal variances (Welch's t-test)

    For proportion metrics (e.g., conversion rate, CTR):
    - Uses two-proportion z-test
    - Calculates pooled proportion for standard error

    Formula (t-test statistic):
    t = (mean_treatment - mean_control) / sqrt(s1²/n1 + s2²/n2)

    Formula (z-test statistic for proportions):
    z = (p1 - p2) / sqrt(p_pool * (1-p_pool) * (1/n1 + 1/n2))

    Args:
        control: Control group observations
        treatment: Treatment group observations
        metric_type: 'continuous' or 'proportion'
        alternative: 'two-sided', 'larger', or 'smaller'
        confidence_level: Confidence level for interval (0-1)

    Returns:
        Dictionary with keys:
        - p_value: Statistical significance p-value
        - is_significant: True if p < alpha (0.05 by default)
        - confidence_level: Confidence level achieved
        - control_mean: Mean of control group
        - treatment_mean: Mean of treatment group
        - absolute_difference: treatment - control
        - relative_lift: (treatment - control) / control * 100
        - confidence_interval: CI for the difference
        - test_statistic: t or z statistic
        - degrees_of_freedom: For t-test
        - recommended_action: Action based on results

    Examples:
        >>> control = pd.Series([100, 105, 98, 102, 99])
        >>> treatment = pd.Series([110, 115, 108, 112, 109])
        >>> result = ab_test_significance(control, treatment)
        >>> print(result['is_significant'])
        True
    """
    # Validate inputs
    if not isinstance(control, (pd.Series, list, np.ndarray)):
        raise TypeError("control must be a pandas Series, list, or numpy array")

    if not isinstance(treatment, (pd.Series, list, np.ndarray)):
        raise TypeError("treatment must be a pandas Series, list, or numpy array")

    if isinstance(control, (list, np.ndarray)):
        control = pd.Series(control)

    if isinstance(treatment, (list, np.ndarray)):
        treatment = pd.Series(treatment)

    # Remove NaN values
    control_clean = control.dropna()
    treatment_clean = treatment.dropna()

    if len(control_clean) < 2:
        raise ValueError(f"Control group has only {len(control_clean)} valid observations (need at least 2)")

    if len(treatment_clean) < 2:
        raise ValueError(f"Treatment group has only {len(treatment_clean)} valid observations (need at least 2)")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    alpha = 1 - confidence_level

    try:
        if metric_type == "continuous":
            # Two-sample t-test (Welch's t-test, unequal variances)
            control_mean = float(control_clean.mean())
            treatment_mean = float(treatment_clean.mean())
            control_std = float(control_clean.std(ddof=1))
            treatment_std = float(treatment_clean.std(ddof=1))

            n1, n2 = len(control_clean), len(treatment_clean)

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(
                treatment_clean,
                control_clean,
                equal_var=False,  # Welch's t-test
                alternative=alternative
            )

            # Calculate degrees of freedom (Welch-Satterthwaite equation)
            df_numerator = (control_std**2 / n1 + treatment_std**2 / n2)**2
            df_denominator = (
                (control_std**2 / n1)**2 / (n1 - 1) +
                (treatment_std**2 / n2)**2 / (n2 - 1)
            )
            df = df_numerator / df_denominator if df_denominator != 0 else n1 + n2 - 2

            # Calculate confidence interval for difference
            se_diff = np.sqrt(control_std**2 / n1 + treatment_std**2 / n2)
            t_crit = stats.t.ppf(1 - alpha / 2, df)

            diff = treatment_mean - control_mean
            ci_lower = diff - t_crit * se_diff
            ci_upper = diff + t_crit * se_diff

        elif metric_type == "proportion":
            # Two-proportion z-test
            # Convert to success/failure if needed
            if control_clean.max() <= 1 and treatment_clean.max() <= 1:
                # Already in proportion format
                control_conversions = control_clean.sum()
                treatment_conversions = treatment_clean.sum()
                n1, n2 = len(control_clean), len(treatment_clean)
            else:
                # Assume raw counts, calculate proportions
                control_conversions = control_clean.sum()
                treatment_conversions = treatment_clean.sum()
                n1 = len(control_clean)
                n2 = len(treatment_clean)

            p1 = control_conversions / n1
            p2 = treatment_conversions / n2

            control_mean = float(p1)
            treatment_mean = float(p2)

            # Pooled proportion
            p_pool = (control_conversions + treatment_conversions) / (n1 + n2)

            # Z-test
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            z_stat = (p2 - p1) / se

            # P-value
            if alternative == "two-sided":
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            elif alternative == "larger":
                p_value = 1 - stats.norm.cdf(z_stat)
            else:  # smaller
                p_value = stats.norm.cdf(z_stat)

            # Confidence interval (using unpooled standard error)
            se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            z_crit = stats.norm.ppf(1 - alpha / 2)

            diff = p2 - p1
            ci_lower = diff - z_crit * se_diff
            ci_upper = diff + z_crit * se_diff

            t_stat = z_stat
            df = None
            control_std = np.sqrt(p1 * (1 - p1))
            treatment_std = np.sqrt(p2 * (1 - p2))

        else:
            raise ValueError(f"Unknown metric_type: {metric_type}. Use 'continuous' or 'proportion'")

        # Calculate lift
        absolute_diff = treatment_mean - control_mean
        relative_lift = (absolute_diff / control_mean * 100) if control_mean != 0 else float('inf')

        # Determine significance
        is_significant = p_value < alpha

        # Generate recommendation
        if is_significant:
            if absolute_diff > 0:
                recommended_action = (
                    f"Roll out treatment to all users. "
                    f"Significant {relative_lift:+.1f}% lift detected (p={p_value:.4f})."
                )
            else:
                recommended_action = (
                    f"Do not roll out treatment. "
                    f"Significant {relative_lift:+.1f}% decline detected (p={p_value:.4f})."
                )
        else:
            recommended_action = (
                f"Results inconclusive. No significant difference detected (p={p_value:.4f}). "
                f"Consider running test longer or with larger sample size."
            )

        return {
            "p_value": round(float(p_value), 4),
            "is_significant": is_significant,
            "confidence_level": confidence_level,
            "control_mean": round(control_mean, 4),
            "treatment_mean": round(treatment_mean, 4),
            "control_std": round(control_std, 4) if control_std is not None else None,
            "treatment_std": round(treatment_std, 4) if treatment_std is not None else None,
            "control_n": n1,
            "treatment_n": n2,
            "absolute_difference": round(absolute_diff, 4),
            "relative_lift": round(relative_lift, 2),
            "confidence_interval": {
                "lower": round(ci_lower, 4),
                "upper": round(ci_upper, 4),
            },
            "test_statistic": round(t_stat, 4),
            "degrees_of_freedom": round(df, 2) if df is not None else None,
            "alpha": alpha,
            "metric_type": metric_type,
            "alternative": alternative,
            "recommended_action": recommended_action,
        }

    except Exception as e:
        raise ValueError(f"A/B test analysis failed: {str(e)}")


# =============================================================================
# LIFT CALCULATION
# =============================================================================

def calculate_lift(
    baseline: float,
    current: float,
    baseline_n: Optional[int] = None,
    current_n: Optional[int] = None,
    confidence_level: float = 0.95,
) -> Dict[str, any]:
    """
    Calculate lift and statistical significance of performance change.

    Lift Formula:
    Lift = (Current - Baseline) / Baseline × 100%

    Standard Error (if sample sizes provided):
    SE = sqrt(Var_current/n_current + Var_baseline/n_baseline)

    Args:
        baseline: Baseline metric value (mean or proportion)
        current: Current metric value (mean or proportion)
        baseline_n: Sample size for baseline (optional, for significance)
        current_n: Sample size for current (optional, for significance)
        confidence_level: Confidence level (0-1)

    Returns:
        Dictionary with keys:
        - lift_percent: Percentage lift
        - lift_type: 'positive', 'negative', or 'neutral'
        - absolute_change: Current - Baseline
        - is_significant: Whether lift is statistically significant
        - p_value: P-value (if sample sizes provided)
        - confidence_interval: CI for lift (if sample sizes provided)
        - confidence_level: Achieved confidence level
        - interpretation: Natural language summary

    Examples:
        >>> result = calculate_lift(0.02, 0.025, baseline_n=1000, current_n=1000)
        >>> print(result['lift_percent'])
        25.0
    """
    # Validate inputs
    if baseline == 0:
        raise ValueError("baseline cannot be zero (division by zero)")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    # Calculate lift
    absolute_change = current - baseline
    lift_percent = (absolute_change / baseline) * 100

    # Determine lift type
    if abs(lift_percent) < 0.1:  # Within 0.1%
        lift_type = "neutral"
    elif lift_percent > 0:
        lift_type = "positive"
    else:
        lift_type = "negative"

    result = {
        "baseline": round(baseline, 4),
        "current": round(current, 4),
        "absolute_change": round(absolute_change, 4),
        "lift_percent": round(lift_percent, 2),
        "lift_type": lift_type,
        "confidence_level": confidence_level,
    }

    # Calculate statistical significance if sample sizes provided
    if baseline_n is not None and current_n is not None:
        if baseline_n <= 0 or current_n <= 0:
            raise ValueError("Sample sizes must be positive")

        # Estimate variance based on metric type
        if 0 <= baseline <= 1 and 0 <= current <= 1:
            # Assume proportion metric
            var_baseline = baseline * (1 - baseline)
            var_current = current * (1 - current)
        else:
            # Assume continuous metric - use coefficient of variation
            # This is an approximation
            cv = 0.5  # Typical CV for marketing metrics
            var_baseline = (baseline * cv) ** 2
            var_current = (current * cv) ** 2

        # Standard error of difference
        se = np.sqrt(var_current / current_n + var_baseline / baseline_n)

        # Z-score
        z_score = absolute_change / se

        # P-value (two-sided)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # Confidence interval
        alpha = 1 - confidence_level
        z_crit = stats.norm.ppf(1 - alpha / 2)
        margin = z_crit * se

        ci_lower = absolute_change - margin
        ci_upper = absolute_change + margin

        # Convert to lift percentage CI
        lift_ci_lower = (ci_lower / baseline) * 100
        lift_ci_upper = (ci_upper / baseline) * 100

        result.update({
            "is_significant": p_value < alpha,
            "p_value": round(float(p_value), 4),
            "confidence_interval": {
                "absolute": {"lower": round(ci_lower, 4), "upper": round(ci_upper, 4)},
                "percent": {"lower": round(lift_ci_lower, 2), "upper": round(lift_ci_upper, 2)},
            },
            "standard_error": round(float(se), 4),
            "z_score": round(float(z_score), 4),
            "baseline_n": baseline_n,
            "current_n": current_n,
        })

        # Update interpretation based on significance
        if result["is_significant"]:
            if lift_type == "positive":
                interpretation = (
                    f"Significant {lift_percent:+.1f}% lift detected (p={p_value:.4f}). "
                    f"The improvement is statistically significant at {confidence_level*100:.0f}% confidence."
                )
            elif lift_type == "negative":
                interpretation = (
                    f"Significant {lift_percent:+.1f}% decline detected (p={p_value:.4f}). "
                    f"The decrease is statistically significant at {confidence_level*100:.0f}% confidence."
                )
            else:
                interpretation = (
                    f"No significant change detected (p={p_value:.4f}). "
                    f"Results are within expected variation range."
                )
        else:
            interpretation = (
                f"{lift_percent:+.1f}% lift detected but not statistically significant (p={p_value:.4f}). "
                f"Consider increasing sample size or running test longer."
            )

    else:
        result.update({
            "is_significant": None,
            "p_value": None,
            "confidence_interval": None,
        })

        interpretation = (
            f"{lift_percent:+.1f}% lift detected. "
            f"Sample sizes not provided, cannot determine statistical significance."
        )

    result["interpretation"] = interpretation

    return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def summarize_statistics(series: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive summary statistics for a series.

    Args:
        series: Input series

    Returns:
        Dictionary with statistical summaries
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    clean_series = series.dropna()

    if len(clean_series) == 0:
        return {}

    return {
        "count": len(clean_series),
        "mean": float(clean_series.mean()),
        "median": float(clean_series.median()),
        "mode": float(clean_series.mode().iloc[0]) if len(clean_series.mode()) > 0 else None,
        "std": float(clean_series.std()),
        "var": float(clean_series.var()),
        "min": float(clean_series.min()),
        "max": float(clean_series.max()),
        "q1": float(clean_series.quantile(0.25)),
        "q3": float(clean_series.quantile(0.75)),
        "iqr": float(clean_series.quantile(0.75) - clean_series.quantile(0.25)),
        "skewness": float(clean_series.skew()),
        "kurtosis": float(clean_series.kurtosis()),
        "cv": float(clean_series.std() / clean_series.mean()) if clean_series.mean() != 0 else None,
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Statistical Analytics Module - Test")
    print("=" * 60)

    # Test data
    test_series = pd.Series([
        10, 12, 11, 13, 15, 14, 16, 18, 17, 19,
        25, 26, 24, 27, 25, 28, 26, 29, 27, 30
    ])

    # Moving average
    ma = calculate_moving_average(test_series, window=5, method="simple")
    print(f"\nMoving Average (last value): {ma.iloc[-1]:.2f}")

    # Changepoints
    changepoints = detect_changepoints(test_series, min_size=7, method="mean_shift")
    print(f"\nChangepoints detected: {len(changepoints)}")
    for cp in changepoints:
        print(f"  - Index {cp['index']}: {cp['direction']} by {cp['magnitude_pct']}%")

    # Forecast
    forecast = forecast_metric(test_series, periods=5, method="exponential_smoothing")
    print(f"\nForecast (next 5): {forecast['forecast']}")

    # A/B test
    control = pd.Series([100, 102, 98, 101, 99, 103, 97, 100])
    treatment = pd.Series([108, 112, 106, 110, 107, 111, 105, 109])
    ab_result = ab_test_significance(control, treatment)
    print(f"\nA/B Test Significant: {ab_result['is_significant']}")
    print(f"Lift: {ab_result['relative_lift']:.1f}%")

    # Lift calculation
    lift = calculate_lift(0.02, 0.025, baseline_n=1000, current_n=1000)
    print(f"\nLift: {lift['lift_percent']:.1f}%")
    print(f"Significant: {lift['is_significant']}")
    print(f"Interpretation: {lift['interpretation']}")
