"""
Report generation and visualization for LLM evaluation results.

This module provides comprehensive reporting capabilities including markdown,
HTML with interactive charts, statistical analysis, and model comparisons.
"""

from src.reporting.report_generator import (
    ReportGenerator,
    generate_report,
)

__all__ = [
    "ReportGenerator",
    "generate_report",
]
