"""
AdInsights-Agent - Autonomous Analytics Agent for AdTech

A LangGraph-powered agent for automated analysis of advertising campaign data.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from src.agents.insights_agent import AdInsightsAgent, AdInsightsState
from src.tools.analysis_tools import (
    fetch_campaign_metrics,
    calculate_period_comparison,
    detect_anomalies,
    analyze_trend,
    find_correlations,
    compare_to_benchmark,
    generate_chart,
)

__all__ = [
    "AdInsightsAgent",
    "AdInsightsState",
    "fetch_campaign_metrics",
    "calculate_period_comparison",
    "detect_anomalies",
    "analyze_trend",
    "find_correlations",
    "compare_to_benchmark",
    "generate_chart",
    "__version__",
]
