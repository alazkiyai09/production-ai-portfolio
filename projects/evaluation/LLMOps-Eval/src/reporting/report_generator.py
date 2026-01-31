"""
Report generation and visualization for LLM evaluation results.

This module provides comprehensive reporting capabilities including markdown,
HTML with interactive charts, statistical analysis, and model comparisons.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import json
import logging
from io import StringIO
import statistics
from collections import defaultdict

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

try:
    from jinja2 import Template, Environment, BaseLoader
except ImportError:
    Template = None
    Environment = None

from src.runners.eval_runner import EvaluationResult, TestResult

logger = logging.getLogger(__name__)


# ============================================================================
# Jinja2 Templates
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f7fa;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }
        .header h1 { margin-bottom: 10px; }
        .header .meta { opacity: 0.9; font-size: 0.9em; }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .stat-card .label {
            font-size: 0.85em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-top: 5px;
        }
        .section {
            padding: 30px;
            border-bottom: 1px solid #eee;
        }
        .section h2 {
            margin-bottom: 20px;
            color: #2d3748;
            font-size: 1.5em;
        }
        .chart-container {
            margin: 20px 0;
            border-radius: 8px;
            overflow: hidden;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        th {
            background: #f7fafc;
            font-weight: 600;
            color: #4a5568;
        }
        tr:hover { background: #f7fafc; }
        .pass { color: #48bb78; font-weight: bold; }
        .fail { color: #f56565; font-weight: bold; }
        .score-bar {
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }
        .score-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s;
        }
        .model-badge {
            display: inline-block;
            padding: 4px 12px;
            background: #edf2f7;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }
        .footer {
            padding: 20px 30px;
            background: #f7fafc;
            text-align: center;
            color: #718096;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <div class="meta">
                Generated: {{ timestamp }} |
                Dataset: {{ dataset }} |
                Duration: {{ duration }}s
            </div>
        </div>

        <div class="summary">
            <div class="stat-card">
                <div class="label">Total Tests</div>
                <div class="value">{{ summary.total_tests }}</div>
            </div>
            <div class="stat-card">
                <div class="label">Success Rate</div>
                <div class="value">{{ "%.1f"|format(summary.success_rate) }}%</div>
            </div>
            <div class="stat-card">
                <div class="label">Total Cost</div>
                <div class="value">${{ "%.4f"|format(summary.total_cost_usd) }}</div>
            </div>
            <div class="stat-card">
                <div class="label">Avg Latency</div>
                <div class="value">{{ "%.0f"|format(summary.avg_latency_ms) }}ms</div>
            </div>
        </div>

        {% if charts %}
        <div class="section">
            <h2>Visualizations</h2>
            {% for chart_id, chart_html in charts.items() %}
            <div class="chart-container">
                {{ chart_html|safe }}
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <div class="section">
            <h2>Model Comparison</h2>
            {{ model_comparison_table|safe }}
        </div>

        {% if metric_table %}
        <div class="section">
            <h2>Metric Summary</h2>
            {{ metric_table|safe }}
        </div>
        {% endif %}

        {% if details_table %}
        <div class="section">
            <h2>Detailed Results</h2>
            {{ details_table|safe }}
        </div>
        {% endif %}

        <div class="footer">
            Generated by LLMOps-Eval | {{ timestamp }}
        </div>
    </div>
</body>
</html>
"""


# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """
    Generate comprehensive evaluation reports.

    Supports markdown, HTML with interactive charts, and statistical analysis.
    """

    def __init__(
        self,
        output_dir: str | Path = "./reports",
        include_charts: bool = True,
    ):
        """
        Initialize the report generator.

        Args:
            output_dir: Directory to save reports
            include_charts: Whether to include charts in HTML reports
        """
        self.output_dir = Path(output_dir)
        self.include_charts = include_charts

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Markdown Reports
    # ========================================================================

    def generate_markdown_report(
        self,
        result: EvaluationResult,
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Generate a markdown evaluation report.

        Args:
            result: Evaluation result
            save_path: Optional path to save the report

        Returns:
            Markdown report content
        """
        md = StringIO()

        # Title
        md.write(f"# Evaluation Report: {result.config.name}\n\n")

        # Metadata
        md.write("## Metadata\n\n")
        md.write(f"- **Dataset:** {result.config.dataset}\n")
        md.write(f"- **Start Time:** {result.start_time}\n")
        md.write(f"- **End Time:** {result.end_time}\n")
        md.write(f"- **Duration:** {result.duration_seconds:.2f}s\n")
        md.write(f"- **Models Evaluated:** {result.summary.get('models_evaluated', 0)}\n\n")

        # Summary
        md.write("## Summary\n\n")
        summary = result.summary
        md.write(f"- **Total Tests:** {summary.get('total_tests', 0)}\n")
        md.write(f"- **Successful:** {summary.get('successful_tests', 0)}\n")
        md.write(f"- **Failed:** {summary.get('failed_tests', 0)}\n")
        md.write(f"- **Success Rate:** {summary.get('success_rate', 0):.2f}%\n")
        md.write(f"- **Total Cost:** ${summary.get('total_cost_usd', 0):.6f}\n")
        md.write(f"- **Total Tokens:** {summary.get('total_tokens', 0):,}\n")
        md.write(f"- **Avg Latency:** {summary.get('avg_latency_ms', 0):.2f}ms\n\n")

        # Model comparison
        md.write("## Model Comparison\n\n")
        md.write(self._generate_markdown_model_table(result))

        # Metric summary
        if "metric_summary" in summary:
            md.write("## Metric Summary\n\n")
            md.write(self._generate_markdown_metric_table(summary["metric_summary"]))

        # Detailed results
        md.write("## Detailed Results\n\n")
        md.write(self._generate_markdown_details(result))

        content = md.getvalue()

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(content, encoding="utf-8")
            logger.info(f"Saved markdown report to {save_path}")

        return content

    def _generate_markdown_model_table(self, result: EvaluationResult) -> str:
        """Generate markdown table for model comparison."""
        md = StringIO()

        model_summaries = result.summary.get("model_summaries", {})
        if not model_summaries:
            return "No model summaries available.\n"

        md.write("| Model | Tests | Success Rate | Avg Score | Avg Latency | Cost |\n")
        md.write("|-------|-------|--------------|-----------|--------------|------|\n")

        for model, stats in model_summaries.items():
            md.write(
                f"| {model} | "
                f"{stats['total_tests']} | "
                f"{stats['success_rate']:.1f}% | "
                f"{stats['avg_score']:.3f} | "
                f"{stats['avg_latency_ms']:.1f}ms | "
                f"${stats['total_cost_usd']:.6f} |\n"
            )

        md.write("\n")
        return md.getvalue()

    def _generate_markdown_metric_table(self, metric_summary: dict) -> str:
        """Generate markdown table for metric summary."""
        md = StringIO()

        if not metric_summary:
            return "No metric summary available.\n"

        md.write("| Metric | Mean | Min | Max | Pass Rate |\n")
        md.write("|--------|------|-----|-----|-----------|\n")

        for metric_name, stats in metric_summary.items():
            md.write(
                f"| {metric_name} | "
                f"{stats['mean']:.3f} | "
                f"{stats['min']:.3f} | "
                f"{stats['max']:.3f} | "
                f"{stats['pass_rate']:.1f}% |\n"
            )

        md.write("\n")
        return md.getvalue()

    def _generate_markdown_details(self, result: EvaluationResult) -> str:
        """Generate detailed results section."""
        md = StringIO()

        # Group by model
        by_model: dict[str, list[TestResult]] = defaultdict(list)
        for test_result in result.results:
            by_model[test_result.model].append(test_result)

        for model, model_results in by_model.items():
            md.write(f"### {model}\n\n")

            for test_result in model_results[:10]:  # Limit to first 10
                status = "✅" if test_result.passed else "❌"
                md.write(f"{status} **{test_result.test_id}**\n\n")
                md.write(f"- *Prompt:* {test_result.prompt[:100]}...\n")
                md.write(f"- *Score:* {test_result.overall_score:.3f}\n")
                md.write(f"- *Latency:* {test_result.latency_ms:.1f}ms\n")
                md.write(f"- *Cost:* ${test_result.cost_usd:.6f}\n\n")

            if len(model_results) > 10:
                md.write(f"*... and {len(model_results) - 10} more results*\n\n")

        return md.getvalue()

    # ========================================================================
    # HTML Reports
    # ========================================================================

    def generate_html_report(
        self,
        result: EvaluationResult,
        save_path: Optional[Path] = None,
    ) -> str:
        """
        Generate an HTML report with embedded charts.

        Args:
            result: Evaluation result
            save_path: Optional path to save the report

        Returns:
            HTML report content
        """
        if Environment is None:
            raise ImportError("jinja2 is required for HTML reports")

        # Generate charts
        charts = {}
        if self.include_charts:
            charts = {
                "comparison": self.create_comparison_chart(result),
                "latency": self.create_latency_distribution(result),
                "cost": self.create_cost_breakdown(result),
            }

        # Generate tables
        model_comparison = self._generate_html_model_table(result)
        metric_table = self._generate_html_metric_table(result)
        details_table = self._generate_html_details_table(result)

        # Render template
        template = Environment(loader=BaseLoader()).from_string(HTML_TEMPLATE)
        html = template.render(
            title=f"Evaluation Report: {result.config.name}",
            timestamp=datetime.fromisoformat(result.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            dataset=result.config.dataset,
            duration=f"{result.duration_seconds:.2f}",
            summary=result.summary,
            charts=charts,
            model_comparison_table=model_comparison,
            metric_table=metric_table,
            details_table=details_table,
        )

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(html, encoding="utf-8")
            logger.info(f"Saved HTML report to {save_path}")

        return html

    def _generate_html_model_table(self, result: EvaluationResult) -> str:
        """Generate HTML table for model comparison."""
        model_summaries = result.summary.get("model_summaries", {})
        if not model_summaries:
            return "<p>No model summaries available.</p>"

        html = ["<table>"]
        html.append("<thead><tr>")
        html.append("<th>Model</th><th>Tests</th><th>Success Rate</th><th>Avg Score</th><th>Avg Latency</th><th>Cost</th>")
        html.append("</tr></thead><tbody>")

        for model, stats in model_summaries.items():
            html.append("<tr>")
            html.append(f'<td><span class="model-badge">{model}</span></td>')
            html.append(f"<td>{stats['total_tests']}</td>")
            html.append(f'<td class="{"pass" if stats["success_rate"] >= 80 else "fail"}">{stats["success_rate"]:.1f}%</td>')
            html.append(f"<td>{stats['avg_score']:.3f}</td>")
            html.append(f"<td>{stats['avg_latency_ms']:.1f}ms</td>")
            html.append(f"<td>${stats['total_cost_usd']:.6f}</td>")
            html.append("</tr>")

        html.append("</tbody></table>")
        return "\n".join(html)

    def _generate_html_metric_table(self, result: EvaluationResult) -> str:
        """Generate HTML table for metric summary."""
        metric_summary = result.summary.get("metric_summary", {})
        if not metric_summary:
            return "<p>No metric summary available.</p>"

        html = ["<table>"]
        html.append("<thead><tr>")
        html.append("<th>Metric</th><th>Mean</th><th>Min</th><th>Max</th><th>Pass Rate</th>")
        html.append("</tr></thead><tbody>")

        for metric_name, stats in metric_summary.items():
            pass_class = "pass" if stats["pass_rate"] >= 80 else "fail"
            html.append("<tr>")
            html.append(f"<td><code>{metric_name}</code></td>")
            html.append(f"<td>{stats['mean']:.3f}</td>")
            html.append(f"<td>{stats['min']:.3f}</td>")
            html.append(f"<td>{stats['max']:.3f}</td>")
            html.append(f'<td class="{pass_class}">{stats["pass_rate"]:.1f}%</td>')
            html.append("</tr>")

        html.append("</tbody></table>")
        return "\n".join(html)

    def _generate_html_details_table(self, result: EvaluationResult) -> str:
        """Generate HTML table for detailed results."""
        html = ["<table>"]
        html.append("<thead><tr>")
        html.append("<th>Test ID</th><th>Model</th><th>Score</th><th>Latency</th><th>Cost</th><th>Status</th>")
        html.append("</tr></thead><tbody>")

        for test_result in result.results[:50]:  # Limit to 50
            status_class = "pass" if test_result.passed else "fail"
            status_text = "✓" if test_result.passed else "✗"

            html.append("<tr>")
            html.append(f"<td>{test_result.test_id}</td>")
            html.append(f'<td><span class="model-badge">{test_result.model}</span></td>')
            html.append(f"<td>{test_result.overall_score:.3f}</td>")
            html.append(f"<td>{test_result.latency_ms:.1f}ms</td>")
            html.append(f"<td>${test_result.cost_usd:.6f}</td>")
            html.append(f'<td class="{status_class}">{status_text}</td>')
            html.append("</tr>")

        if len(result.results) > 50:
            html.append(f'<tr><td colspan="6"><em>... and {len(result.results) - 50} more results</em></td></tr>')

        html.append("</tbody></table>")
        return "\n".join(html)

    # ========================================================================
    # Charts
    # ========================================================================

    def create_comparison_chart(
        self,
        result: EvaluationResult,
    ) -> str:
        """
        Create a comparison chart across models.

        Returns:
            HTML div with Plotly chart
        """
        model_summaries = result.summary.get("model_summaries", {})
        if not model_summaries:
            return "<p>No data for comparison chart.</p>"

        models = list(model_summaries.keys())
        metrics = result.config.metrics

        # Create subplot for each metric
        fig = make_subplots(
            rows=1,
            cols=len(metrics),
            subplot_titles=metrics,
            horizontal_spacing=0.15,
        )

        colors = px.colors.qualitative.Set2

        for i, metric in enumerate(metrics, start=1):
            values = []
            for model in models:
                metric_stats = model_summaries[model].get("metric_averages", {}).get(metric, {})
                values.append(metric_stats.get("mean", 0))

            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric,
                    marker_color=colors[i % len(colors)],
                    showlegend=False,
                ),
                row=1,
                col=i,
            )

        fig.update_layout(
            title="Model Comparison by Metric",
            height=400,
            margin=dict(t=80, b=80),
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def create_latency_distribution(
        self,
        result: EvaluationResult,
    ) -> str:
        """
        Create latency distribution chart.

        Returns:
            HTML div with Plotly chart
        """
        # Group latencies by model
        by_model: dict[str, list[float]] = defaultdict(list)
        for test_result in result.results:
            if not test_result.error:
                by_model[test_result.model].append(test_result.latency_ms)

        if not by_model:
            return "<p>No latency data available.</p>"

        fig = go.Figure()

        for model, latencies in by_model.items():
            fig.add_trace(
                go.Box(
                    y=latencies,
                    name=model,
                    boxmean=True,
                )
            )

        fig.update_layout(
            title="Latency Distribution by Model",
            yaxis_title="Latency (ms)",
            xaxis_title="Model",
            height=400,
            margin=dict(t=80, b=80),
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def create_cost_breakdown(
        self,
        result: EvaluationResult,
    ) -> str:
        """
        Create cost breakdown chart by model.

        Returns:
            HTML div with Plotly chart
        """
        model_summaries = result.summary.get("model_summaries", {})
        if not model_summaries:
            return "<p>No cost data available.</p>"

        models = list(model_summaries.keys())
        costs = [model_summaries[m]["total_cost_usd"] for m in models]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=models,
                    values=costs,
                    hole=0.3,
                    textinfo="label+percent",
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title="Cost Breakdown by Model",
            height=400,
            margin=dict(t=80, b=80),
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def create_metric_heatmap(
        self,
        result: EvaluationResult,
    ) -> str:
        """
        Create a heatmap of metric scores across models and test cases.

        Returns:
            HTML div with Plotly chart
        """
        # Build matrix
        models = list(set(r.model for r in result.results))
        test_ids = list(set(r.test_id for r in result.results))

        z_matrix = []
        for model in models:
            row = []
            for test_id in test_ids:
                result = next(
                    (r for r in result.results if r.model == model and r.test_id == test_id),
                    None,
                )
                row.append(result.overall_score if result else 0)
            z_matrix.append(row)

        fig = go.Figure(
            data=go.Heatmap(
                z=z_matrix,
                x=test_ids,
                y=models,
                colorscale="RdYlGn",
                zmin=0,
                zmax=1,
            )
        )

        fig.update_layout(
            title="Score Heatmap",
            xaxis_title="Test Case",
            yaxis_title="Model",
            height=400,
            margin=dict(t=80, b=80, l=100),
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    # ========================================================================
    # Statistical Analysis
    # ========================================================================

    def compare_models(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        """
        Statistical comparison of models across multiple evaluation runs.

        Args:
            results: List of evaluation results to compare

        Returns:
            Dictionary with statistical analysis
        """
        if not results:
            return {}

        # Collect scores by model
        model_scores: dict[str, list[float]] = defaultdict(list)
        model_latencies: dict[str, list[float]] = defaultdict(list)
        model_costs: dict[str, list[float]] = defaultdict(list)

        for result in results:
            for test_result in result.results:
                if not test_result.error:
                    model_scores[test_result.model].append(test_result.overall_score)
                    model_latencies[test_result.model].append(test_result.latency_ms)
                    model_costs[test_result.model].append(test_result.cost_usd)

        # Calculate statistics
        comparison = {}
        for model in model_scores:
            scores = model_scores[model]
            latencies = model_latencies[model]
            costs = model_costs[model]

            comparison[model] = {
                "score": {
                    "mean": statistics.mean(scores) if scores else 0,
                    "median": statistics.median(scores) if scores else 0,
                    "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "min": min(scores) if scores else 0,
                    "max": max(scores) if scores else 0,
                },
                "latency_ms": {
                    "mean": statistics.mean(latencies) if latencies else 0,
                    "median": statistics.median(latencies) if latencies else 0,
                    "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                },
                "cost_usd": {
                    "mean": statistics.mean(costs) if costs else 0,
                    "total": sum(costs),
                },
            }

        # Determine winner
        if comparison:
            winner = max(comparison.items(), key=lambda x: x[1]["score"]["mean"])
            comparison["winner"] = {
                "model": winner[0],
                "mean_score": winner[1]["score"]["mean"],
            }

        return comparison

    def calculate_rankings(
        self,
        result: EvaluationResult,
    ) -> list[dict[str, Any]]:
        """
        Calculate rankings of models based on multiple criteria.

        Args:
            result: Evaluation result

        Returns:
            List of model rankings
        """
        model_summaries = result.summary.get("model_summaries", {})
        if not model_summaries:
            return []

        rankings = []
        for model, stats in model_summaries.items():
            score = (
                stats["avg_score"] * 0.4 +  # 40% weight on score
                (stats["success_rate"] / 100) * 0.3 +  # 30% weight on success rate
                min(1, 5000 / stats["avg_latency_ms"]) * 0.2 +  # 20% weight on latency
                min(1, 0.01 / stats["total_cost_usd"]) * 0.1  # 10% weight on cost
            )
            rankings.append({
                "model": model,
                "overall_score": score,
                "avg_score": stats["avg_score"],
                "success_rate": stats["success_rate"],
                "avg_latency_ms": stats["avg_latency_ms"],
                "total_cost_usd": stats["total_cost_usd"],
            })

        # Sort by overall score
        rankings.sort(key=lambda x: x["overall_score"], reverse=True)

        # Add ranks
        for i, ranking in enumerate(rankings, start=1):
            ranking["rank"] = i

        return rankings

    # ========================================================================
    # Export Functions
    # ========================================================================

    def export_to_csv(
        self,
        result: EvaluationResult,
        path: Optional[Path] = None,
    ) -> Path:
        """
        Export results to CSV format.

        Args:
            result: Evaluation result
            path: Optional path to save CSV

        Returns:
            Path to saved CSV file
        """
        # Prepare data
        rows = []
        for test_result in result.results:
            row = {
                "test_id": test_result.test_id,
                "model": test_result.model,
                "provider": test_result.provider,
                "prompt": test_result.prompt,
                "response": test_result.response[:500],  # Truncate long responses
                "expected": test_result.expected,
                "overall_score": test_result.overall_score,
                "passed": test_result.passed,
                "latency_ms": test_result.latency_ms,
                "cost_usd": test_result.cost_usd,
                "input_tokens": test_result.input_tokens,
                "output_tokens": test_result.output_tokens,
                "total_tokens": test_result.total_tokens,
                "error": test_result.error,
                "timestamp": test_result.timestamp,
            }

            # Add metric scores
            for metric_name, metric_result in test_result.metrics.items():
                row[f"metric_{metric_name}"] = metric_result.value
                row[f"metric_{metric_name}_passed"] = metric_result.passed

            rows.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(rows)

        if path is None:
            timestamp = datetime.fromisoformat(result.start_time).strftime("%Y%m%d_%H%M%S")
            path = self.output_dir / f"{result.config.name}_{timestamp}.csv"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

        logger.info(f"Exported results to {path}")
        return path

    def export_to_json(
        self,
        result: EvaluationResult,
        path: Optional[Path] = None,
    ) -> Path:
        """
        Export results to JSON format.

        Args:
            result: Evaluation result
            path: Optional path to save JSON

        Returns:
            Path to saved JSON file
        """
        if path is None:
            timestamp = datetime.fromisoformat(result.start_time).strftime("%Y%m%d_%H%M%S")
            path = self.output_dir / f"{result.config.name}_{timestamp}.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        logger.info(f"Exported results to {path}")
        return path


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_report(
    result: EvaluationResult,
    format: str = "html",
    save: bool = True,
    output_dir: str | Path = "./reports",
) -> str:
    """
    Generate an evaluation report.

    Args:
        result: Evaluation result
        format: Report format (markdown, html)
        save: Whether to save the report
        output_dir: Output directory

    Returns:
        Report content

    Examples:
        >>> html = generate_report(result, format="html")
        >>> md = generate_report(result, format="markdown")
    """
    generator = ReportGenerator(output_dir=output_dir)

    timestamp = datetime.fromisoformat(result.start_time).strftime("%Y%m%d_%H%M%S")
    filename = f"{result.config.name}_{timestamp}"

    if format.lower() == "markdown" or format.lower() == "md":
        path = generator.output_dir / f"{filename}.md" if save else None
        return generator.generate_markdown_report(result, save_path=path)
    elif format.lower() == "html":
        path = generator.output_dir / f"{filename}.html" if save else None
        return generator.generate_html_report(result, save_path=path)
    else:
        raise ValueError(f"Unsupported format: {format}")


# Export main classes and functions
__all__ = [
    "ReportGenerator",
    "generate_report",
]
