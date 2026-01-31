"""
Time Series Analytics Module for AdInsights-Agent

Provides time series analysis including trend detection,
anomaly detection, forecasting, and changepoint detection.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class TrendDirection(str, Enum):
    """Trend direction classification."""
    UPWARD = "upward"
    DOWNWARD = "downward"
    STABLE = "stable"
    VOLATILE = "volatile"


class AnomalyType(str, Enum):
    """Type of anomaly detected."""
    SPIKE = "spike"  # Sudden increase
    DROP = "drop"  # Sudden decrease
    BREAKPOINT = "breakpoint"  # Sudden change in trend
    OUTLIER = "outlier"  # Statistical outlier
    SEASONAL_ANOMALY = "seasonal_anomaly"  # Deviation from seasonal pattern


# =============================================================================
# RESULT CLASSES
# =============================================================================

class TrendResult:
    """Result of trend analysis on time series."""

    def __init__(
        self,
        direction: TrendDirection,
        slope: float,
        correlation: float,
        confidence: float,
        start_value: float,
        end_value: float,
        percent_change: float,
        description: str
    ):
        self.direction = direction
        self.slope = slope
        self.correlation = correlation
        self.confidence = confidence
        self.start_value = start_value
        self.end_value = end_value
        self.percent_change = percent_change
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": self.direction.value,
            "slope": self.slope,
            "correlation": self.correlation,
            "confidence": self.confidence,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "percent_change": self.percent_change,
            "description": self.description
        }


class AnomalyResult:
    """Result of anomaly detection."""

    def __init__(
        self,
        anomaly_type: AnomalyType,
        timestamp: datetime,
        value: float,
        expected_value: float,
        deviation: float,
        severity: str,  # "low", "medium", "high"
        confidence: float
    ):
        self.anomaly_type = anomaly_type
        self.timestamp = timestamp
        self.value = value
        self.expected_value = expected_value
        self.deviation = deviation
        self.severity = severity
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_type": self.anomaly_type.value,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "expected_value": self.expected_value,
            "deviation": self.deviation,
            "severity": self.severity,
            "confidence": self.confidence
        }


class ForecastResult:
    """Result of time series forecasting."""

    def __init__(
        self,
        forecast: List[float],
        timestamps: List[datetime],
        lower_bound: List[float],
        upper_bound: List[float],
        method: str,
        mae: Optional[float] = None,
        rmse: Optional[float] = None
    ):
        self.forecast = forecast
        self.timestamps = timestamps
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.method = method
        self.mae = mae  # Mean Absolute Error
        self.rmse = rmse  # Root Mean Square Error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "forecast": self.forecast,
            "timestamps": [ts.isoformat() for ts in self.timestamps],
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "method": self.method,
            "mae": self.mae,
            "rmse": self.rmse
        }


# =============================================================================
# TIME SERIES ANALYZER
# =============================================================================

class TimeSeriesAnalyzer:
    """
    Comprehensive time series analysis for ad campaign metrics.

    Provides trend analysis, anomaly detection, forecasting,
    and changepoint detection.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        anomaly_threshold: float = 2.5,
        min_periods: int = 2
    ):
        """
        Initialize the time series analyzer.

        Args:
            confidence_level: Confidence level for statistical tests (0-1)
            anomaly_threshold: Number of std deviations for anomaly detection
            min_periods: Minimum periods for trend analysis
        """
        self.confidence_level = confidence_level
        self.anomaly_threshold = anomaly_threshold
        self.min_periods = min_periods

    def analyze_trend(
        self,
        data: pd.DataFrame,
        value_column: str,
        date_column: str = "date"
    ) -> TrendResult:
        """
        Analyze trend in time series data.

        Uses linear regression to determine trend direction and strength.

        Args:
            data: DataFrame with time series data
            value_column: Column name with metric values
            date_column: Column name with dates (must be datetime or convertible)

        Returns:
            TrendResult with trend analysis

        Raises:
            ValueError: If insufficient data points
        """
        # Prepare data
        df = data.copy().sort_values(date_column)
        df = df[[date_column, value_column]].dropna()

        if len(df) < self.min_periods:
            raise ValueError(f"Insufficient data points for trend analysis: {len(df)} < {self.min_periods}")

        # Convert dates to numeric (days since start)
        start_date = df[date_column].min()
        df["days"] = (df[date_column] - start_date).dt.days

        # Linear regression
        x = df["days"].values
        y = df[value_column].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Determine trend direction
        start_value = y[0]
        end_value = y[-1]
        percent_change = ((end_value - start_value) / start_value * 100) if start_value != 0 else 0

        # Classify trend
        if p_value > 0.05:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.UPWARD if percent_change > 5 else TrendDirection.STABLE
        else:
            direction = TrendDirection.DOWNWARD if percent_change < -5 else TrendDirection.STABLE

        # Check for volatility
        if np.std(y) / np.mean(np.abs(y)) > 0.5 if np.mean(np.abs(y)) > 0 else False:
            direction = TrendDirection.VOLATILE

        # Confidence based on R-squared
        confidence = abs(r_value) ** 2

        # Generate description
        description = self._generate_trend_description(
            direction, slope, confidence, percent_change
        )

        logger.info(f"Trend analysis: {direction.value}, slope={slope:.4f}, r2={confidence:.3f}")

        return TrendResult(
            direction=direction,
            slope=slope,
            correlation=r_value,
            confidence=confidence,
            start_value=start_value,
            end_value=end_value,
            percent_change=percent_change,
            description=description
        )

    def detect_anomalies(
        self,
        data: pd.DataFrame,
        value_column: str,
        date_column: str = "date",
        method: str = "zscore"
    ) -> List[AnomalyResult]:
        """
        Detect anomalies in time series data.

        Args:
            data: DataFrame with time series data
            value_column: Column name with metric values
            date_column: Column name with dates
            method: Detection method ("zscore", "iqr", "isolation_forest")

        Returns:
            List of AnomalyResult objects

        Raises:
            ValueError: If insufficient data points
        """
        df = data.copy().sort_values(date_column)
        df = df[[date_column, value_column]].dropna()

        if len(df) < 10:
            raise ValueError(f"Insufficient data for anomaly detection: {len(df)} < 10")

        anomalies = []

        if method == "zscore":
            anomalies = self._detect_zscore_anomalies(df, value_column, date_column)
        elif method == "iqr":
            anomalies = self._detect_iqr_anomalies(df, value_column, date_column)
        elif method == "isolation_forest":
            anomalies = self._detect_isolation_forest_anomalies(df, value_column, date_column)
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")

        logger.info(f"Detected {len(anomalies)} anomalies using {method} method")

        return anomalies

    def forecast(
        self,
        data: pd.DataFrame,
        value_column: str,
        date_column: str = "date",
        periods: int = 7,
        method: str = "auto"
    ) -> ForecastResult:
        """
        Forecast future values of time series.

        Args:
            data: Historical data
            value_column: Column name with metric values
            date_column: Column name with dates
            periods: Number of periods to forecast
            method: Forecast method ("auto", "arima", "exponential", "simple")

        Returns:
            ForecastResult with predictions and confidence intervals

        Raises:
            ValueError: If insufficient data for forecasting
        """
        df = data.copy().sort_values(date_column)
        df = df[[date_column, value_column]].dropna()

        min_required = {"arima": 20, "exponential": 10, "simple": 5}.get(method, 10)

        if len(df) < min_required:
            raise ValueError(
                f"Insufficient data for {method} forecasting: {len(df)} < {min_required}"
            )

        # Prepare series
        series = df.set_index(date_column)[value_column]

        if method == "auto":
            method = self._select_forecast_method(series)

        # Generate forecast
        if method == "arima":
            return self._arima_forecast(series, periods)
        elif method == "exponential":
            return self._exponential_forecast(series, periods)
        else:
            return self._simple_forecast(series, periods)

    # -------------------------------------------------------------------------
    # PRIVATE METHODS
    # -------------------------------------------------------------------------

    def _generate_trend_description(
        self,
        direction: TrendDirection,
        slope: float,
        confidence: float,
        percent_change: float
    ) -> str:
        """Generate human-readable trend description."""
        confidence_pct = confidence * 100

        if direction == TrendDirection.UPWARD:
            return (
                f"Strong {'upward' if slope > 0 else 'downward'} trend detected "
                f"({confidence_pct:.1f}% confidence). "
                f"{'Increased' if percent_change > 0 else 'Decreased'} by {abs(percent_change):.1f}% "
                f"over the period."
            )
        elif direction == TrendDirection.DOWNWARD:
            return (
                f"Downward trend detected ({confidence_pct:.1f}% confidence). "
                f"Decreased by {abs(percent_change):.1f}% over the period."
            )
        elif direction == TrendDirection.VOLATILE:
            return (
                f"High volatility detected with no clear trend "
                f"({confidence_pct:.1f}% confidence in linear fit: {slope:.4f}). "
                f"Changed by {percent_change:+.1f}% overall."
            )
        else:
            return (
                f"Stable trend detected ({confidence_pct:.1f}% confidence). "
                f"Minimal change ({percent_change:+.1f}%) over the period."
            )

    def _detect_zscore_anomalies(
        self,
        df: pd.DataFrame,
        value_column: str,
        date_column: str
    ) -> List[AnomalyResult]:
        """Detect anomalies using z-score method."""
        values = df[value_column].values
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return []

        z_scores = np.abs((values - mean) / std)
        threshold = self.anomaly_threshold

        anomalies = []
        for idx, z_score in enumerate(z_scores):
            if z_score > threshold:
                severity = "high" if z_score > 4 else "medium" if z_score > 3 else "low"

                anomalies.append(AnomalyResult(
                    anomaly_type=AnomalyType.OUTLIER,
                    timestamp=df.iloc[idx][date_column].to_pydatetime(),
                    value=values[idx],
                    expected_value=mean,
                    deviation=values[idx] - mean,
                    severity=severity,
                    confidence=min(z_score / threshold, 1.0)
                ))

        return anomalies

    def _detect_iqr_anomalies(
        self,
        df: pd.DataFrame,
        value_column: str,
        date_column: str
    ) -> List[AnomalyResult]:
        """Detect anomalies using IQR method."""
        values = df[value_column].values
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        anomalies = []
        for idx, value in enumerate(values):
            if value < lower_bound:
                anomalies.append(AnomalyResult(
                    anomaly_type=AnomalyType.DROP,
                    timestamp=df.iloc[idx][date_column].to_pydatetime(),
                    value=value,
                    expected_value=lower_bound,
                    deviation=value - lower_bound,
                    severity="medium",
                    confidence=0.8
                ))
            elif value > upper_bound:
                anomalies.append(AnomalyResult(
                    anomaly_type=AnomalyType.SPIKE,
                    timestamp=df.iloc[idx][date_column].to_pydatetime(),
                    value=value,
                    expected_value=upper_bound,
                    deviation=value - upper_bound,
                    severity="medium",
                    confidence=0.8
                ))

        return anomalies

    def _detect_isolation_forest_anomalies(
        self,
        df: pd.DataFrame,
        value_column: str,
        date_column: str
    ) -> List[AnomalyResult]:
        """Detect anomalies using Isolation Forest (if available)."""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("scikit-learn not available, falling back to zscore method")
            return self._detect_zscore_anomalies(df, value_column, date_column)

        # Reshape data for sklearn
        values = df[value_column].values.reshape(-1, 1)

        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = iso_forest.fit_predict(values)

        anomalies = []
        mean = np.mean(values)

        for idx, pred in enumerate(predictions):
            if pred == -1:  # Anomaly
                value = values[idx][0]
                anomalies.append(AnomalyResult(
                    anomaly_type=AnomalyType.OUTLIER,
                    timestamp=df.iloc[idx][date_column].to_pydatetime(),
                    value=value,
                    expected_value=mean,
                    deviation=value - mean,
                    severity="medium",
                    confidence=0.7
                ))

        return anomalies

    def _select_forecast_method(self, series: pd.Series) -> str:
        """Select best forecasting method based on data characteristics."""
        # Check for seasonality
        try:
            decomposition = seasonal_decompose(series, period=min(7, len(series) // 2))
            has_seasonality = np.std(decomposition.seasonal) > np.std(series) * 0.1
        except:
            has_seasonality = False

        # Check trend
        has_trend = abs(stats.linregress(np.arange(len(series)), series.values)[0]) > 0.01

        if has_seasonality and has_trend:
            return "exponential"
        elif has_trend:
            return "arima"
        else:
            return "simple"

    def _arima_forecast(
        self,
        series: pd.Series,
        periods: int
    ) -> ForecastResult:
        """Generate ARIMA forecast."""
        try:
            # Fit ARIMA model
            model = ARIMA(series, order=(1, 1, 1))
            model_fit = model.fit()

            # Forecast
            forecast_result = model_fit.forecast(steps=periods)
            forecast = forecast_result.values

            # Get prediction intervals
            forecast_result = model_fit.get_forecast(steps=periods, alpha=0.05)
            lower_bound = forecast_result.conf_int[:, 0]
            upper_bound = forecast_result.conf_int[:, 1]

            # Generate timestamps
            last_date = series.index[-1]
            if isinstance(last_date, pd.Timestamp):
                freq = pd.infer_freq(series.index)
                forecast_index = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
            else:
                forecast_index = [last_date + timedelta(days=i) for i in range(1, periods + 1)]

            return ForecastResult(
                forecast=forecast.tolist(),
                timestamps=forecast_index,
                lower_bound=lower_bound.tolist(),
                upper_bound=upper_bound.tolist(),
                method="arima"
            )

        except Exception as e:
            logger.warning(f"ARIMA forecasting failed: {e}, falling back to simple")
            return self._simple_forecast(series, periods)

    def _exponential_forecast(
        self,
        series: pd.Series,
        periods: int
    ) -> ForecastResult:
        """Generate exponential smoothing forecast."""
        try:
            model = ExponentialSmoothing(series, trend="add", seasonal=None)
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=periods)

            # Simple confidence intervals (based on fitted std)
            resid_std = np.std(model_fit.resid)
            lower_bound = forecast - 1.96 * resid_std
            upper_bound = forecast + 1.96 * resid_std

            # Generate timestamps
            last_date = series.index[-1]
            if isinstance(last_date, pd.Timestamp):
                freq = pd.infer_freq(series.index)
                forecast_index = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
            else:
                forecast_index = [last_date + timedelta(days=i) for i in range(1, periods + 1)]

            return ForecastResult(
                forecast=forecast.tolist(),
                timestamps=forecast_index,
                lower_bound=lower_bound.tolist(),
                upper_bound=upper_bound.tolist(),
                method="exponential"
            )

        except Exception as e:
            logger.warning(f"Exponential smoothing failed: {e}, falling back to simple")
            return self._simple_forecast(series, periods)

    def _simple_forecast(
        self,
        series: pd.Series,
        periods: int
    ) -> ForecastResult:
        """Generate simple moving average forecast."""
        # Use last value as forecast
        last_value = series.iloc[-1]
        forecast = [last_value] * periods

        # Calculate std for confidence intervals
        std = np.std(series)
        lower_bound = [last_value - 1.96 * std] * periods
        upper_bound = [last_value + 1.96 * std] * periods

        # Generate timestamps
        last_date = series.index[-1]
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(series.index) or "D"
            forecast_index = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
        else:
            forecast_index = [last_date + timedelta(days=i) for i in range(1, periods + 1)]

        return ForecastResult(
            forecast=forecast,
            timestamps=forecast_index,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            method="simple"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_trend(
    data: pd.DataFrame,
    value_column: str,
    date_column: str = "date"
) -> TrendResult:
    """
    Convenience function for trend analysis.

    Args:
        data: DataFrame with time series data
        value_column: Column name with metric values
        date_column: Column name with dates

    Returns:
        TrendResult object
    """
    analyzer = TimeSeriesAnalyzer()
    return analyzer.analyze_trend(data, value_column, date_column)


def detect_anomalies(
    data: pd.DataFrame,
    value_column: str,
    date_column: str = "date",
    method: str = "zscore"
) -> List[AnomalyResult]:
    """
    Convenience function for anomaly detection.

    Args:
        data: DataFrame with time series data
        value_column: Column name with metric values
        date_column: Column name with dates
        method: Detection method

    Returns:
        List of AnomalyResult objects
    """
    analyzer = TimeSeriesAnalyzer()
    return analyzer.detect_anomalies(data, value_column, date_column, method)


def forecast_metrics(
    data: pd.DataFrame,
    value_column: str,
    date_column: str = "date",
    periods: int = 7
) -> ForecastResult:
    """
    Convenience function for forecasting.

    Args:
        data: Historical data
        value_column: Column name with metric values
        date_column: Column name with dates
        periods: Number of periods to forecast

    Returns:
        ForecastResult object
    """
    analyzer = TimeSeriesAnalyzer()
    return analyzer.forecast(data, value_column, date_column, periods)
