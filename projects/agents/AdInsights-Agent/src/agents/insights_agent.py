"""
AdInsights-Agent - LangGraph Agent for AdTech Analytics

An autonomous agent that analyzes advertising campaign data, detects
trends and anomalies, and generates actionable insights.

Usage:
    agent = AdInsightsAgent()
    result = agent.analyze("Analyze campaign CAMP-001 for the last 30 days")
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any, Literal, TypedDict, Annotated
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Required

# Import analysis tools
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.analysis_tools import (
    fetch_campaign_metrics,
    calculate_period_comparison,
    detect_anomalies,
    analyze_trend,
    find_correlations,
    compare_to_benchmark,
    generate_chart,
)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AdInsightsState(TypedDict, total=False):
    """
    State for the AdInsights analysis workflow.

    All fields are optional and populated as the analysis progresses.
    """
    # Input fields
    request: Required[str]  # Original user request
    campaign_id: Optional[str]  # Extracted campaign ID
    date_range: Optional[Tuple[str, str]]  # (start_date, end_date)
    campaign_type: Optional[str]  # Type of healthcare campaign
    analysis_plan: List[str]  # Planned analysis steps

    # Data fields
    raw_data: Optional[Dict[str, Any]]  # Raw campaign data from API
    metrics_summary: Optional[Dict[str, Any]]  # Summary statistics

    # Analysis results
    anomalies: List[Dict[str, Any]]  # Detected anomalies
    trends: Dict[str, Any]  # Trend analysis results
    correlations: List[Dict[str, Any]]  # Correlation analysis
    benchmark_comparison: Optional[Dict[str, Any]]  # Benchmark comparison
    period_comparison: Optional[Dict[str, Any]]  # Period-over-period comparison

    # Outputs
    insights: List[str]  # Generated insights
    recommendations: List[str]  # Actionable recommendations
    charts: List[str]  # Paths to generated charts
    final_report: str  # Final markdown report

    # Execution tracking
    current_step: str  # Current step in workflow
    errors: List[str]  # Any errors encountered
    completed_steps: List[str]  # Steps completed


# =============================================================================
# AGENT CLASS
# =============================================================================

class AdInsightsAgent:
    """
    Autonomous LangGraph agent for AdTech campaign analysis.

    Features:
    - Parses natural language requests
    - Plans multi-step analysis
    - Detects anomalies and trends
    - Compares to benchmarks
    - Generates insights with charts
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        enable_parallel: bool = True,
    ):
        """
        Initialize the AdInsights agent.

        Args:
            model: OpenAI model to use
            temperature: LLM temperature (lower = more deterministic)
            enable_parallel: Enable parallel execution of independent analyses
        """
        self.model_name = model
        self.temperature = temperature
        self.enable_parallel = enable_parallel

        # Initialize LLM (lazy loading)
        self._llm = None

        # Build the graph
        self.graph = self._build_graph()

        # Create charts directory
        self.charts_dir = Path("./data/charts")
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def llm(self):
        """Lazy-load LLM only when needed."""
        if self._llm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self._llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=api_key,
            )
        return self._llm

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        # Create graph with state
        workflow = StateGraph(AdInsightsState)

        # Add nodes
        workflow.add_node("parse_request", self._parse_request_node)
        workflow.add_node("plan_analysis", self._plan_analysis_node)
        workflow.add_node("gather_data", self._gather_data_node)
        workflow.add_node("analyze_metrics", self._analyze_metrics_node)
        workflow.add_node("detect_anomalies", self._detect_anomalies_node)
        workflow.add_node("analyze_trends", self._analyze_trends_node)
        workflow.add_node("compare_benchmarks", self._compare_benchmarks_node)
        workflow.add_node("find_correlations", self._find_correlations_node)
        workflow.add_node("generate_insights", self._generate_insights_node)
        workflow.add_node("create_report", self._create_report_node)

        # Set entry point
        workflow.set_entry_point("parse_request")

        # Define edges (sequential flow with conditional routing)
        workflow.add_edge("parse_request", "plan_analysis")
        workflow.add_edge("plan_analysis", "gather_data")
        workflow.add_edge("gather_data", "analyze_metrics")

        # Conditional routing after analyze_metrics
        workflow.add_conditional_edges(
            "analyze_metrics",
            self._route_after_metrics,
            {
                "detect_anomalies": "detect_anomalies",
                "analyze_trends": "analyze_trends",
                "compare_benchmarks": "compare_benchmarks",
                "find_correlations": "find_correlations",
                "generate_insights": "generate_insights",
            }
        )

        # Analysis nodes -> insights
        workflow.add_edge("detect_anomalies", "generate_insights")
        workflow.add_edge("analyze_trends", "generate_insights")
        workflow.add_edge("compare_benchmarks", "generate_insights")
        workflow.add_edge("find_correlations", "generate_insights")

        # Final steps
        workflow.add_edge("generate_insights", "create_report")
        workflow.add_edge("create_report", END)

        return workflow.compile()

    # =========================================================================
    # NODE FUNCTIONS
    # =========================================================================

    def _parse_request_node(self, state: AdInsightsState) -> AdInsightsState:
        """
        Parse user request and extract campaign ID, date range, and intent.

        Extracts:
        - Campaign ID (pattern: CAMP-XXX or similar)
        - Date range (last N days, date range, etc.)
        - Campaign type (healthcare segment)
        """
        request = state["request"].lower()
        state["current_step"] = "parse_request"
        state["completed_steps"] = state.get("completed_steps", [])

        # Extract campaign ID
        campaign_match = re.search(r'camp[\s_-]?(\d+|[a-z0-9\-]+)', request, re.IGNORECASE)
        if campaign_match:
            state["campaign_id"] = f"CAMP-{campaign_match.group(1).upper()}"
        else:
            state["campaign_id"] = "CAMP-DEMO-001"  # Default

        # Extract date range
        # Look for "last N days", "past N days", "N days", date ranges
        days_match = re.search(r'(?:last|past)\s+(\d+)\s+days?', request)
        if days_match:
            days = int(days_match.group(1))
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            state["date_range"] = (
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
        else:
            # Default to last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            state["date_range"] = (
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )

        # Extract campaign type
        campaign_types = {
            "pharma": "healthcare_pharma",
            "pharmaceutical": "healthcare_pharma",
            "hospital": "healthcare_hospitals",
            "hospitals": "healthcare_hospitals",
            "telehealth": "healthcare_telehealth",
            "telemedicine": "healthcare_telehealth",
            "insurance": "healthcare_insurance",
            "health plan": "healthcare_insurance",
        }

        for keyword, campaign_type in campaign_types.items():
            if keyword in request:
                state["campaign_type"] = campaign_type
                break
        else:
            state["campaign_type"] = "healthcare_pharma"  # Default

        state["completed_steps"].append("parse_request")

        return state

    def _plan_analysis_node(self, state: AdInsightsState) -> AdInsightsState:
        """
        Plan the analysis based on user request.

        Determines which analyses are needed:
        - Anomaly detection (always)
        - Trend analysis (always)
        - Benchmark comparison (if requested)
        - Correlation analysis (if looking for relationships)
        - Period comparison (if comparing time periods)
        """
        request = state["request"].lower()
        state["current_step"] = "plan_analysis"

        analysis_plan = ["detect_anomalies", "analyze_trends"]

        # Add analysis based on keywords
        if any(word in request for word in ["benchmark", "compare", "industry", "standard"]):
            analysis_plan.append("compare_benchmarks")

        if any(word in request for word in ["correlation", "relationship", "factor", "impact"]):
            analysis_plan.append("find_correlations")

        if any(word in request for word in ["period", "vs", "versus", "week over week", "wow"]):
            analysis_plan.append("period_comparison")

        # Always analyze metrics
        analysis_plan.insert(0, "analyze_metrics")

        state["analysis_plan"] = analysis_plan
        state["completed_steps"].append("plan_analysis")

        return state

    def _gather_data_node(self, state: AdInsightsState) -> AdInsightsState:
        """Fetch campaign data for the specified date range."""
        state["current_step"] = "gather_data"

        try:
            campaign_id = state["campaign_id"]
            start_date, end_date = state["date_range"]
            campaign_type = state.get("campaign_type", "healthcare_pharma")

            # Fetch data using tool
            result = fetch_campaign_metrics.invoke({
                "campaign_id": campaign_id,
                "start_date": start_date,
                "end_date": end_date,
                "campaign_type": campaign_type,
            })

            if "error" in result:
                state["errors"] = state.get("errors", []) + [result["error"]]
                state["raw_data"] = None
            else:
                state["raw_data"] = result

            state["completed_steps"].append("gather_data")

        except Exception as e:
            state["errors"] = state.get("errors", []) + [f"Data gathering failed: {str(e)}"]
            state["raw_data"] = None

        return state

    def _analyze_metrics_node(self, state: AdInsightsState) -> AdInsightsState:
        """Calculate summary statistics for key metrics."""
        state["current_step"] = "analyze_metrics"

        if not state.get("raw_data"):
            state["metrics_summary"] = None
            state["completed_steps"].append("analyze_metrics")
            return state

        try:
            raw_data = state["raw_data"]
            summary = raw_data.get("summary", {})

            # Enhanced summary with additional insights
            metrics_summary = {
                **summary,
                "data_quality": {
                    "total_days": raw_data.get("record_count", 0),
                    "date_range": raw_data.get("date_range", {}),
                    "campaign_id": raw_data.get("campaign_id"),
                },
                "performance_indicators": {
                    "is_performing_well": summary.get("avg_ctr", 0) > 1.0 and summary.get("avg_cvr", 0) > 2.0,
                    "ctr_rating": "good" if summary.get("avg_ctr", 0) > 1.5 else "average" if summary.get("avg_ctr", 0) > 1.0 else "poor",
                    "cvr_rating": "good" if summary.get("avg_cvr", 0) > 3.0 else "average" if summary.get("avg_cvr", 0) > 2.0 else "poor",
                }
            }

            state["metrics_summary"] = metrics_summary
            state["completed_steps"].append("analyze_metrics")

        except Exception as e:
            state["errors"] = state.get("errors", []) + [f"Metrics analysis failed: {str(e)}"]
            state["metrics_summary"] = None

        return state

    def _detect_anomalies_node(self, state: AdInsightsState) -> AdInsightsState:
        """Detect anomalies in key metrics."""
        state["current_step"] = "detect_anomalies"

        if not state.get("raw_data"):
            state["anomalies"] = []
            state["completed_steps"].append("detect_anomalies")
            return state

        try:
            raw_data = state["raw_data"]
            anomalies = []

            # Check for anomalies in key metrics
            for metric in ["ctr", "cvr", "cpa", "roi"]:
                result = detect_anomalies.invoke({
                    "data": raw_data,
                    "metric": metric,
                    "method": "zscore",
                    "threshold": 2.5,
                })

                if result.get("anomalies"):
                    for anomaly in result["anomalies"]:
                        anomaly["metric_name"] = metric.upper()
                        anomalies.append(anomaly)

            # Sort by severity and score
            anomalies.sort(key=lambda x: (x["severity"], x["score"]), reverse=True)

            state["anomalies"] = anomalies
            state["completed_steps"].append("detect_anomalies")

        except Exception as e:
            state["errors"] = state.get("errors", []) + [f"Anomaly detection failed: {str(e)}"]
            state["anomalies"] = []

        return state

    def _analyze_trends_node(self, state: AdInsightsState) -> AdInsightsState:
        """Analyze trends in key metrics and forecast."""
        state["current_step"] = "analyze_trends"

        if not state.get("raw_data"):
            state["trends"] = {}
            state["completed_steps"].append("analyze_trends")
            return state

        try:
            raw_data = state["raw_data"]
            trends = {}

            # Analyze trends for key metrics
            for metric in ["ctr", "cvr", "spend", "conversions"]:
                result = analyze_trend.invoke({
                    "data": raw_data,
                    "metric": metric,
                    "forecast_days": 7,
                })

                if "error" not in result:
                    trends[metric] = result

            state["trends"] = trends
            state["completed_steps"].append("analyze_trends")

        except Exception as e:
            state["errors"] = state.get("errors", []) + [f"Trend analysis failed: {str(e)}"]
            state["trends"] = {}

        return state

    def _compare_benchmarks_node(self, state: AdInsightsState) -> AdInsightsState:
        """Compare campaign performance to industry benchmarks."""
        state["current_step"] = "compare_benchmarks"

        if not state.get("metrics_summary"):
            state["benchmark_comparison"] = None
            state["completed_steps"].append("compare_benchmarks")
            return state

        try:
            summary = state["metrics_summary"]
            campaign_type = state.get("campaign_type", "healthcare_pharma")

            # Prepare metrics for comparison
            campaign_metrics = {
                "avg_ctr": summary.get("avg_ctr", 0),
                "avg_cvr": summary.get("avg_cvr", 0),
                "avg_cpa": summary.get("avg_cpa", 0),
                "avg_roi": summary.get("avg_roi", 0),
            }

            result = compare_to_benchmark.invoke({
                "campaign_metrics": campaign_metrics,
                "industry": campaign_type,
            })

            state["benchmark_comparison"] = result
            state["completed_steps"].append("compare_benchmarks")

        except Exception as e:
            state["errors"] = state.get("errors", []) + [f"Benchmark comparison failed: {str(e)}"]
            state["benchmark_comparison"] = None

        return state

    def _find_correlations_node(self, state: AdInsightsState) -> AdInsightsState:
        """Find correlations between metrics."""
        state["current_step"] = "find_correlations"

        if not state.get("raw_data"):
            state["correlations"] = []
            state["completed_steps"].append("find_correlations")
            return state

        try:
            raw_data = state["raw_data"]
            correlations = []

            # Find correlations for key metrics
            for target_metric in ["roi", "conversions", "ctr"]:
                result = find_correlations.invoke({
                    "data": raw_data,
                    "target_metric": target_metric,
                    "min_correlation": 0.3,
                })

                if result.get("correlations"):
                    correlations.extend(result["correlations"])

            # Remove duplicates and sort
            seen = set()
            unique_correlations = []
            for corr in correlations:
                key = (corr["metric"], corr.get("target_metric", ""))
                if key not in seen:
                    seen.add(key)
                    unique_correlations.append(corr)

            unique_correlations.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

            state["correlations"] = unique_correlations
            state["completed_steps"].append("find_correlations")

        except Exception as e:
            state["errors"] = state.get("errors", []) + [f"Correlation analysis failed: {str(e)}"]
            state["correlations"] = []

        return state

    def _generate_insights_node(self, state: AdInsightsState) -> AdInsightsState:
        """Generate natural language insights and recommendations."""
        state["current_step"] = "generate_insights"

        insights = []
        recommendations = []

        try:
            # Metrics summary insights
            if state.get("metrics_summary"):
                summary = state["metrics_summary"]
                ctr = summary.get("avg_ctr", 0)
                cvr = summary.get("avg_cvr", 0)
                roi = summary.get("avg_roi", 0)

                insights.append(
                    f"**Campaign Performance Summary:**\n"
                    f"- Average CTR: {ctr:.2f}% ({summary.get('performance_indicators', {}).get('ctr_rating', 'N/A')} rating)\n"
                    f"- Average CVR: {cvr:.2f}% ({summary.get('performance_indicators', {}).get('cvr_rating', 'N/A')} rating)\n"
                    f"- Average ROI: {roi:.2f}x\n"
                    f"- Total Spend: ${summary.get('total_spend', 0):.2f}\n"
                    f"- Total Conversions: {summary.get('total_conversions', 0)}"
                )

            # Anomaly insights
            if state.get("anomalies"):
                anomalies = state["anomalies"]
                high_severity = [a for a in anomalies if a["severity"] == "high"]

                if high_severity:
                    insights.append(
                        f"**Anomalies Detected:**\n"
                        f"- Found {len(anomalies)} total anomalies ({len(high_severity)} high severity)\n"
                        f"- Metrics affected: {', '.join(set(a['metric_name'] for a in anomalies[:5]))}\n"
                        f"- Most significant: {anomalies[0]['metric_name']} on {anomalies[0]['date']} "
                        f"({anomalies[0]['direction'] if 'direction' in anomalies[0] else 'value'}: {anomalies[0]['value']:.2f})"
                    )

                    recommendations.append(
                        "**Investigate Anomalies:** Review the high-severity anomalies, "
                        "particularly those affecting CTR and CVR, as they may indicate "
                        "technical issues or unusual market conditions."
                    )
                else:
                    insights.append("**Anomalies:** No significant anomalies detected.")

            # Trend insights
            if state.get("trends"):
                trends = state["trends"]

                trend_summary = []
                for metric, trend_data in trends.items():
                    direction = trend_data.get("trend_direction", "unknown")
                    strength = trend_data.get("trend_strength", "unknown")
                    metric_upper = metric.upper()

                    if direction == "up":
                        trend_summary.append(f"{metric_upper} is trending {direction} ({strength})")
                    elif direction == "down":
                        trend_summary.append(f"⚠️ {metric_upper} is trending {direction} ({strength})")
                        recommendations.append(
                            f"**Address {metric_upper} Decline:** The downward trend in {metric_upper} "
                            f"requires attention. Consider reviewing ad creative, targeting, or bidding strategy."
                        )

                if trend_summary:
                    insights.append("**Trend Analysis:**\n" + "\n".join(f"- {t}" for t in trend_summary))

            # Benchmark comparison insights
            if state.get("benchmark_comparison"):
                benchmark = state["benchmark_comparison"]
                overall = benchmark.get("overall_performance", "unknown")

                insights.append(
                    f"**Industry Benchmark Comparison:**\n"
                    f"- Overall performance: {overall.upper()}\n"
                    f"- Percentile ranking: {benchmark.get('percentile_ranking', 0):.1f}th percentile"
                )

                if benchmark.get("recommendations"):
                    recommendations.extend(benchmark["recommendations"])

            # Correlation insights
            if state.get("correlations"):
                top_corrs = state["correlations"][:3]

                if top_corrs:
                    corr_text = "\n".join([
                        f"- {corr['metric']} and {corr.get('target_metric', 'target')}: "
                        f"{corr['pearson_r']:.2f} ({corr['strength']} {corr['direction']} correlation)"
                        for corr in top_corrs
                    ])
                    insights.append(f"**Key Correlations:**\n{corr_text}")

            # Generate chart for trends
            if state.get("raw_data") and state.get("trends"):
                try:
                    chart_result = generate_chart.invoke({
                        "data": state["raw_data"],
                        "chart_type": "line",
                        "metrics": ["ctr", "cvr", "roi"],
                        "title": f"Campaign {state['campaign_id']} - Key Metrics Over Time",
                    })

                    if chart_result.get("success"):
                        state["charts"] = state.get("charts", []) + [chart_result["file_path"]]
                except:
                    pass  # Chart generation is optional

            state["insights"] = insights
            state["recommendations"] = recommendations
            state["completed_steps"].append("generate_insights")

        except Exception as e:
            state["errors"] = state.get("errors", []) + [f"Insight generation failed: {str(e)}"]
            state["insights"] = [f"Error generating insights: {str(e)}"]
            state["recommendations"] = []

        return state

    def _create_report_node(self, state: AdInsightsState) -> AdInsightsState:
        """Create final markdown report."""
        state["current_step"] = "create_report"

        try:
            report_lines = []

            # Header
            report_lines.append(f"# Campaign Analysis Report")
            report_lines.append(f"\n**Campaign ID:** {state.get('campaign_id', 'Unknown')}")
            report_lines.append(f"**Date Range:** {state.get('date_range', ('Unknown', 'Unknown'))[0]} to {state.get('date_range', ('Unknown', 'Unknown'))[1]}")
            report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("\n---\n")

            # Executive Summary
            report_lines.append("## Executive Summary")
            report_lines.append(f"\n**Original Request:** {state['request']}")
            report_lines.append(f"\n**Analysis Plan:** {', '.join(state.get('analysis_plan', []))}")
            report_lines.append("\n")

            # Insights
            if state.get("insights"):
                report_lines.append("## Key Insights")
                report_lines.append("")
                report_lines.extend(state["insights"])
                report_lines.append("\n")

            # Recommendations
            if state.get("recommendations"):
                report_lines.append("## Recommendations")
                report_lines.append("")
                for i, rec in enumerate(state["recommendations"], 1):
                    report_lines.append(f"{i}. {rec}")
                report_lines.append("\n")

            # Detailed Analysis
            report_lines.append("## Detailed Analysis")
            report_lines.append("")

            # Metrics Summary
            if state.get("metrics_summary"):
                report_lines.append("### Performance Metrics")
                summary = state["metrics_summary"]
                report_lines.append(f"- **Impressions:** {summary.get('total_impressions', 0):,}")
                report_lines.append(f"- **Clicks:** {summary.get('total_clicks', 0):,}")
                report_lines.append(f"- **Conversions:** {summary.get('total_conversions', 0):,}")
                report_lines.append(f"- **Total Spend:** ${summary.get('total_spend', 0):,.2f}")
                report_lines.append(f"- **Avg CTR:** {summary.get('avg_ctr', 0):.2f}%")
                report_lines.append(f"- **Avg CVR:** {summary.get('avg_cvr', 0):.2f}%")
                report_lines.append(f"- **Avg CPA:** ${summary.get('avg_cpa', 0):.2f}")
                report_lines.append(f"- **Avg ROI:** {summary.get('avg_roi', 0):.2f}x")
                report_lines.append("\n")

            # Trend Details
            if state.get("trends"):
                report_lines.append("### Trend Analysis")
                for metric, trend_data in state["trends"].items():
                    direction = trend_data.get("trend_direction", "unknown")
                    strength = trend_data.get("trend_strength", "unknown")
                    r_squared = trend_data.get("r_squared", 0)
                    interpretation = trend_data.get("interpretation", "")

                    report_lines.append(f"\n**{metric.upper()}:**")
                    report_lines.append(f"- Direction: {direction} ({strength})")
                    report_lines.append(f"- R²: {r_squared:.3f}")
                    report_lines.append(f"- {interpretation}")
                report_lines.append("\n")

            # Anomalies
            if state.get("anomalies"):
                report_lines.append("### Anomalies Detected")
                anomalies = state["anomalies"][:10]  # Limit to top 10
                for anomaly in anomalies:
                    report_lines.append(
                        f"- **{anomaly['metric_name']}** on {anomaly['date']}: "
                        f"value={anomaly['value']:.2f}, "
                        f"severity={anomaly['severity']}, "
                        f"score={anomaly['score']:.2f}"
                    )
                report_lines.append("\n")

            # Charts
            if state.get("charts"):
                report_lines.append("### Visualizations")
                for chart_path in state["charts"]:
                    chart_name = Path(chart_path).name
                    report_lines.append(f"- ![Chart](./{chart_path})")
                report_lines.append("\n")

            # Errors
            if state.get("errors"):
                report_lines.append("## Errors Encountered")
                for error in state["errors"]:
                    report_lines.append(f"- {error}")
                report_lines.append("\n")

            # Footer
            report_lines.append("---\n")
            report_lines.append("*Report generated by AdInsights-Agent*")

            state["final_report"] = "\n".join(report_lines)
            state["completed_steps"].append("create_report")

        except Exception as e:
            state["final_report"] = f"Error creating report: {str(e)}"

        return state

    # =========================================================================
    # ROUTING FUNCTIONS
    # =========================================================================

    def _route_after_metrics(self, state: AdInsightsState) -> str:
        """
        Determine next step after metrics analysis.

        Routes to the next planned analysis step.
        """
        analysis_plan = state.get("analysis_plan", [])
        completed = state.get("completed_steps", [])

        # Find next uncompleted analysis step
        for step in analysis_plan:
            if step not in completed and step in [
                "detect_anomalies",
                "analyze_trends",
                "compare_benchmarks",
                "find_correlations",
            ]:
                return step

        # All analyses complete, go to insights
        return "generate_insights"

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def analyze(
        self,
        request: str,
        campaign_id: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a campaign based on natural language request.

        Args:
            request: Natural language analysis request
            campaign_id: Optional campaign ID (overrides extracted ID)
            date_range: Optional date range (start, end)

        Returns:
            Dictionary containing:
            - report: Markdown report
            - insights: List of insights
            - recommendations: List of recommendations
            - charts: List of chart paths
            - state: Full analysis state
        """
        # Initialize state
        initial_state: AdInsightsState = {
            "request": request,
            "campaign_id": campaign_id,
            "date_range": date_range,
            "analysis_plan": [],
            "raw_data": None,
            "metrics_summary": None,
            "anomalies": [],
            "trends": {},
            "correlations": [],
            "benchmark_comparison": None,
            "insights": [],
            "recommendations": [],
            "charts": [],
            "final_report": "",
            "current_step": "",
            "errors": [],
            "completed_steps": [],
        }

        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)

            return {
                "success": True,
                "report": final_state.get("final_report", ""),
                "insights": final_state.get("insights", []),
                "recommendations": final_state.get("recommendations", []),
                "charts": final_state.get("charts", []),
                "state": final_state,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "report": f"Analysis failed: {str(e)}",
                "insights": [],
                "recommendations": [],
                "charts": [],
                "state": initial_state,
            }

    def run_interactive(self):
        """Run the agent in interactive mode."""
        print("=" * 60)
        print("AdInsights-Agent - Interactive Analysis")
        print("=" * 60)
        print("Type 'quit' or 'exit' to stop\n")

        while True:
            try:
                request = input("\nEnter your analysis request: ").strip()

                if request.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye!")
                    break

                if not request:
                    continue

                print("\nAnalyzing...")
                result = self.analyze(request)

                if result["success"]:
                    print("\n" + "=" * 60)
                    print(result["report"])
                    print("=" * 60)

                    if result["recommendations"]:
                        print("\n📌 Action Items:")
                        for i, rec in enumerate(result["recommendations"], 1):
                            print(f"  {i}. {rec}")
                else:
                    print(f"\n❌ Error: {result['error']}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    import sys

    print("AdInsights-Agent - LangGraph Analytics Agent")
    print("=" * 60)

    # Sample request
    sample_request = "Analyze campaign CAMP-001 for the last 30 days and explain any performance issues"

    print(f"\nSample Request: '{sample_request}'\n")

    # Create agent
    try:
        agent = AdInsightsAgent()

        # Run analysis
        print("Running analysis...")
        result = agent.analyze(sample_request)

        if result["success"]:
            print("\n" + "=" * 60)
            print(result["report"])
            print("=" * 60)

            print("\n📊 Charts Generated:")
            for chart in result["charts"]:
                print(f"  - {chart}")

            print("\n💡 Key Insights:")
            for insight in result["insights"]:
                print(f"  • {insight[:100]}...")

            print("\n📌 Recommendations:")
            for i, rec in enumerate(result["recommendations"], 1):
                print(f"  {i}. {rec}")

        else:
            print(f"\n❌ Analysis failed: {result['error']}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: Make sure OPENAI_API_KEY is set in your environment")
        print("Install dependencies: pip install -r requirements.txt")

    # Interactive mode option
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        agent.run_interactive()
