"""
Formatting utilities for displaying results.

Provides functions for formatting risk summaries and triage results
for display in logs, API responses, or UI.
"""

from typing import Any


def format_risk_summary(risk_score: int, risk_factors: list[str]) -> str:
    """
    Format risk score and factors into a readable summary.

    Args:
        risk_score: Risk score (0-100)
        risk_factors: List of risk factors

    Returns:
        Formatted risk summary
    """
    # Determine risk level
    if risk_score >= 80:
        risk_level = "CRITICAL"
    elif risk_score >= 60:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    summary = f"Risk Level: {risk_level} (Score: {risk_score}/100)\n"

    if risk_factors:
        summary += "\nRisk Factors:\n"
        for i, factor in enumerate(risk_factors, 1):
            summary += f"  {i}. {factor}\n"
    else:
        summary += "\nNo specific risk factors identified."

    return summary


def format_triage_result(result: dict[str, Any]) -> str:
    """
    Format complete triage result for display.

    Args:
        result: Triage result dictionary

    Returns:
        Formatted triage result
    """
    lines = [
        "=" * 60,
        "FRAUD TRIAGE RESULT",
        "=" * 60,
        f"Alert ID: {result.get('alert_id', 'N/A')}",
        f"Risk Score: {result.get('risk_score', 0)}/100",
        f"Confidence: {result.get('confidence', 0):.0%}",
        "",
    ]

    # Risk factors
    if result.get("risk_factors"):
        lines.append("Risk Factors:")
        for factor in result["risk_factors"]:
            lines.append(f"  - {factor}")
        lines.append("")

    # Recommendation
    if result.get("recommendation"):
        lines.append("Recommendation:")
        lines.append(f"  {result['recommendation']}")
        lines.append("")

    # Next action
    if result.get("next_action"):
        lines.append(f"Next Action: {result['next_action']}")

    # Human review
    if result.get("requires_human_review"):
        lines.append("\n⚠️  HUMAN REVIEW REQUIRED")

    lines.append("=" * 60)

    return "\n".join(lines)
