"""
Streamlit dashboard for LLMOps-Eval.

Provides interactive UI for running evaluations, viewing results,
and comparing models.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import asyncio
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# Configure page
st.set_page_config(
    page_title="LLMOps-Eval Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.875rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .success { color: #48bb78; }
    .warning { color: #ed8936; }
    .error { color: #f56565; }
    .info { color: #4299e1; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State
# ============================================================================

if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

if "current_evaluation" not in st.session_state:
    st.session_state.current_evaluation = None

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"


# ============================================================================
# API Client
# ============================================================================

import requests


class APIClient:
    """Simple API client for interacting with the FastAPI backend."""

    def __init__(self, base_url: str = st.session_state.api_base_url):
        self.base_url = base_url

    def get(self, endpoint: str) -> Dict[str, Any]:
        """GET request."""
        try:
            response = requests.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {e}")
            return {}

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST request."""
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=data,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {e}")
            return {}

    def delete(self, endpoint: str) -> Dict[str, Any]:
        """DELETE request."""
        try:
            response = requests.delete(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {e}")
            return {}


api = APIClient()


# ============================================================================
# Utility Functions
# ============================================================================

def format_cost(cost_usd: float) -> str:
    """Format cost in USD."""
    return f"${cost_usd:.6f}"


def format_latency(ms: float) -> str:
    """Format latency."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    else:
        return f"{ms/1000:.2f}s"


def format_score(score: float) -> str:
    """Format score with color."""
    if score >= 0.8:
        return f'<span class="success">{score:.3f}</span>'
    elif score >= 0.5:
        return f'<span class="warning">{score:.3f}</span>'
    else:
        return f'<span class="error">{score:.3f}</span>'


def load_datasets() -> List[Dict[str, Any]]:
    """Load available datasets."""
    data = api.get("/datasets")
    return data.get("datasets", [])


def load_metrics() -> List[str]:
    """Load available metrics."""
    data = api.get("/metrics")
    return list(data.get("metrics", {}).keys())


def load_evaluations() -> List[Dict[str, Any]]:
    """Load all evaluations."""
    data = api.get("/evaluations")
    return data.get("evaluations", [])


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")

    # API URL
    st.session_state.api_base_url = st.sidebar.text_input(
        "API URL",
        value=st.session_state.api_base_url,
        help="Base URL of the FastAPI backend",
    )
    global api
    api = APIClient(st.session_state.api_base_url)

    st.sidebar.markdown("---")

    # New Evaluation Section
    st.sidebar.subheader("🚀 New Evaluation")

    # Evaluation name
    eval_name = st.sidebar.text_input(
        "Evaluation Name",
        value=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Unique name for this evaluation",
    )

    # Dataset selection
    datasets = load_datasets()
    dataset_names = [d["name"] for d in datasets]
    selected_dataset = st.sidebar.selectbox(
        "Dataset",
        options=dataset_names,
        help="Select dataset to evaluate",
    )

    # Model selection
    st.sidebar.markdown("**Models**")
    use_default_models = st.sidebar.checkbox("Use default models", value=True)

    if use_default_models:
        selected_models = [
            {"provider": "openai", "model": "gpt-4o-mini"},
            {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
        ]
    else:
        provider = st.sidebar.selectbox(
            "Provider",
            options=["openai", "anthropic", "ollama"],
        )
        model = st.sidebar.text_input("Model", value="gpt-4o-mini")
        selected_models = [{"provider": provider, "model": model}]

    # Metrics selection
    available_metrics = load_metrics()
    selected_metrics = st.sidebar.multiselect(
        "Metrics",
        options=available_metrics,
        default=["exact_match", "semantic_similarity", "latency", "cost"],
        help="Select metrics to evaluate",
    )

    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        parallel = st.slider(
            "Parallel Evaluations",
            min_value=1,
            max_value=20,
            value=5,
        )
        sample_size = st.number_input(
            "Sample Size",
            min_value=1,
            max_value=1000,
            value=None,
            step=1,
            help="Number of test cases to evaluate (None = all)",
        )

        dataset_version = st.text_input(
            "Dataset Version",
            value="latest",
            help="Specific version to use",
        )

    # Start evaluation button
    if st.sidebar.button(
        "▶️ Start Evaluation",
        type="primary",
        use_container_width=True,
    ):
        if not selected_dataset:
            st.sidebar.error("Please select a dataset")
        elif not selected_models:
            st.sidebar.error("Please select at least one model")
        elif not selected_metrics:
            st.sidebar.error("Please select at least one metric")
        else:
            start_evaluation(eval_name, selected_dataset, selected_models, selected_metrics, {
                "parallel": parallel,
                "sample_size": sample_size,
                "dataset_version": dataset_version,
            })

    st.sidebar.markdown("---")

    # Quick Actions
    st.sidebar.subheader("⚡ Quick Actions")

    if st.sidebar.button("🔄 Refresh Evaluations", use_container_width=True):
        st.session_state.evaluations = load_evaluations()
        st.sidebar.success("Refreshed!")
        st.rerun()

    if st.sidebar.button("🗑️ Clear Completed", use_container_width=True):
        clear_completed_evaluations()


# ============================================================================
# Evaluation Functions
# ============================================================================

def start_evaluation(
    name: str,
    dataset: str,
    models: List[Dict[str, str]],
    metrics: List[str],
    options: Dict[str, Any],
):
    """Start a new evaluation."""
    with st.sidebar:
        with st.spinner("Starting evaluation..."):
            payload = {
                "name": name,
                "dataset": dataset,
                "models": models,
                "metrics": metrics,
                "parallel": options.get("parallel", 5),
                "sample_size": options.get("sample_size"),
                "dataset_version": options.get("dataset_version", "latest"),
            }

            result = api.post("/evaluate", payload)

            if result.get("evaluation_id"):
                st.success(f"Evaluation started: {result['evaluation_id']}")
                st.session_state.evaluations = load_evaluations()
                st.session_state.current_evaluation = result["evaluation_id"]
            else:
                st.error("Failed to start evaluation")


def clear_completed_evaluations():
    """Clear completed evaluations from session state."""
    st.session_state.evaluations = [
        e for e in st.session_state.evaluations
        if e.get("status") not in ["completed", "failed"]
    ]
    st.rerun()


# ============================================================================
# Main Content
# ============================================================================

def render_overview_tab():
    """Render overview tab with summary statistics."""
    st.markdown('<h1 class="main-header">📊 Overview</h1>', unsafe_allow_html=True)

    evaluations = load_evaluations()

    if not evaluations:
        st.info("No evaluations found. Start a new evaluation from the sidebar.")
        return

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total = len(evaluations)
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Total Evaluations</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{total}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        completed = sum(1 for e in evaluations if e.get("status") == "completed")
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Completed</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value success">{completed}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        running = sum(1 for e in evaluations if e.get("status") == "running")
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Running</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value info">{running}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        failed = sum(1 for e in evaluations if e.get("status") == "failed")
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Failed</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value error">{failed}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Recent evaluations table
    st.markdown("### Recent Evaluations")

    df_data = []
    for eval_data in evaluations[:20]:
        df_data.append({
            "ID": eval_data.get("id", "")[:8] + "...",
            "Name": eval_data.get("config", {}).get("name", "N/A"),
            "Dataset": eval_data.get("config", {}).get("dataset", "N/A"),
            "Status": eval_data.get("status", "unknown"),
            "Created": eval_data.get("created_at", "N/A")[:19],
        })

    if df_data:
        df = pd.DataFrame(df_data)

        # Color status column
        def color_status(val):
            if val == "completed":
                return "background-color: #d4edda"
            elif val == "running":
                return "background-color: #cce5ff"
            elif val == "failed":
                return "background-color: #f8d7da"
            return ""

        styled_df = df.style.applymap(color_status, subset=["Status"])
        st.dataframe(styled_df, use_container_width=True)


def render_results_tab():
    """Render results tab with detailed evaluation results."""
    st.markdown('<h1 class="main-header">📈 Results</h1>', unsafe_allow_html=True)

    evaluations = load_evaluations()
    completed_evals = [e for e in evaluations if e.get("status") == "completed"]

    if not completed_evals:
        st.info("No completed evaluations found.")
        return

    # Select evaluation
    eval_options = {f"{e.get('config', {}).get('name', e['id'][:8])}": e["id"] for e in completed_evals}
    selected_name = st.selectbox("Select Evaluation", options=list(eval_options.keys()))
    eval_id = eval_options[selected_name]

    # Get results
    with st.spinner("Loading results..."):
        result = api.get(f"/evaluate/{eval_id}")

    if not result or "config" not in result:
        st.error("Failed to load results")
        return

    # Summary cards
    summary = result.get("summary", {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tests", summary.get("total_tests", 0))
    with col2:
        st.metric("Success Rate", f"{summary.get('success_rate', 0):.1f}%")
    with col3:
        st.metric("Avg Latency", f"{summary.get('avg_latency_ms', 0):.0f}ms")
    with col4:
        st.metric("Total Cost", f"${summary.get('total_cost_usd', 0):.6f}")

    # Model comparison
    st.markdown("### Model Comparison")

    model_summaries = summary.get("model_summaries", {})
    if model_summaries:
        # Create comparison chart
        models = list(model_summaries.keys())
        metrics_data = result.get("config", {}).get("metrics", [])

        fig = make_subplots(
            rows=1,
            cols=len(metrics_data),
            subplot_titles=metrics_data,
            horizontal_spacing=0.15,
        )

        colors = px.colors.qualitative.Set2

        for i, metric in enumerate(metrics_data, start=1):
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
            height=400,
            margin=dict(t=80, b=80),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Model comparison table
        st.markdown("#### Detailed Comparison")

        comparison_data = []
        for model, stats in model_summaries.items():
            comparison_data.append({
                "Model": model,
                "Tests": stats.get("total_tests", 0),
                "Success Rate": f"{stats.get('success_rate', 0):.1f}%",
                "Avg Score": f"{stats.get('avg_score', 0):.3f}",
                "Avg Latency": f"{stats.get('avg_latency_ms', 0):.1f}ms",
                "Total Cost": f"${stats.get('total_cost_usd', 0):.6f}",
            })

        st.dataframe(
            pd.DataFrame(comparison_data),
            use_container_width=True,
        )


def render_comparison_tab():
    """Render comparison tab for comparing multiple evaluations."""
    st.markdown('<h1 class="main-header">⚖️ Comparison</h1>', unsafe_allow_html=True)

    evaluations = load_evaluations()
    completed_evals = [e for e in evaluations if e.get("status") == "completed"]

    if len(completed_evals) < 2:
        st.info("Need at least 2 completed evaluations to compare.")
        return

    # Select evaluations to compare
    eval_options = {
        f"{e.get('config', {}).get('name', e['id'][:8])}": e["id"]
        for e in completed_evals
    }

    selected_names = st.multiselect(
        "Select Evaluations to Compare",
        options=list(eval_options.keys()),
        default=list(eval_options.keys())[:3],
    )

    if len(selected_names) < 2:
        st.warning("Please select at least 2 evaluations to compare")
        return

    # Get comparison
    eval_ids = [eval_options[name] for name in selected_names]

    with st.spinner("Generating comparison..."):
        comparison = api.post("/compare", {"evaluation_ids": eval_ids})

    if not comparison:
        st.error("Failed to generate comparison")
        return

    comp_data = comparison.get("comparison", {})

    # Show winner
    if "winner" in comp_data:
        winner = comp_data["winner"]
        st.success(f"🏆 Best Model: {winner.get('model', 'N/A')} (score: {winner.get('mean_score', 0):.3f})")

    # Comparison table
    st.markdown("### Statistical Comparison")

    table_data = []
    for model, stats in comp_data.items():
        if model != "winner":
            table_data.append({
                "Model": model,
                "Score (Mean)": f"{stats.get('score', {}).get('mean', 0):.3f}",
                "Score (StdDev)": f"{stats.get('score', {}).get('stdev', 0):.3f}",
                "Latency (Mean)": f"{stats.get('latency_ms', {}).get('mean', 0):.1f}ms",
                "Cost (Total)": f"${stats.get('cost_usd', {}).get('total', 0):.6f}",
            })

    if table_data:
        st.dataframe(
            pd.DataFrame(table_data),
            use_container_width=True,
        )

    # Charts
    if len(table_data) > 1:
        df = pd.DataFrame(table_data)

        # Score comparison
        fig_scores = px.bar(
            df,
            x="Model",
            y=[float(s.split()[0]) for s in df["Score (Mean)"]],
            title="Score Comparison",
            labels={"y": "Score"},
            color="Model",
        )
        st.plotly_chart(fig_scores, use_container_width=True)

        # Latency comparison
        fig_latency = px.bar(
            df,
            x="Model",
            y=[float(s.split()[0]) for s in df["Latency (Mean)"]],
            title="Latency Comparison",
            labels={"y": "Latency (ms)"},
            color="Model",
        )
        st.plotly_chart(fig_latency, use_container_width=True)


def render_history_tab():
    """Render history tab with historical evaluations."""
    st.markdown('<h1 class="main-header">📜 History</h1>', unsafe_allow_html=True)

    evaluations = load_evaluations()

    if not evaluations:
        st.info("No evaluations found.")
        return

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["All", "completed", "running", "pending", "failed"],
        )

    with col2:
        sort_order = st.selectbox(
            "Sort By",
            options=["Newest First", "Oldest First"],
        )

    # Apply filters
    filtered_evals = evaluations
    if status_filter != "All":
        filtered_evals = [e for e in evaluations if e.get("status") == status_filter]

    # Sort
    if sort_order == "Newest First":
        filtered_evals.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    else:
        filtered_evals.sort(key=lambda x: x.get("created_at", ""))

    # Display
    for eval_data in filtered_evals:
        with st.expander(
            f"{eval_data.get('config', {}).get('name', eval_data['id'][:8])} - "
            f"{eval_data.get('status', 'unknown').upper()}"
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**ID:** {eval_data.get('id', '')}")
                st.write(f"**Status:** {eval_data.get('status', 'unknown')}")
                st.write(f"**Created:** {eval_data.get('created_at', '')[:19]}")

            with col2:
                config = eval_data.get("config", {})
                st.write(f"**Dataset:** {config.get('dataset', 'N/A')}")
                models = config.get("models", [])
                st.write(f"**Models:** {len(models)}")
                st.write(f"**Metrics:** {len(config.get('metrics', []))}")

            with col3:
                if eval_data.get("status") == "completed":
                    st.write(f"**Total Tests:** {eval_data.get('total_tests', 'N/A')}")
                    st.write(f"**Success Rate:** {eval_data.get('success_rate', 'N/A')}")

                # View details button
                if st.button("View Details", key=f"view_{eval_data.get('id')}"):
                    st.session_state.selected_evaluation = eval_data.get("id")
                    st.rerun()

                # Download report button
                if eval_data.get("status") == "completed":
                    if st.button("📥 Report", key=f"report_{eval_data.get('id')}"):
                        # Generate and download report
                        report_html = api.get(
                            f"/reports/{eval_data.get('id')}?format=html"
                        )
                        st.download_button(
                            label="Download HTML",
                            data=report_html,
                            file_name=f"report_{eval_data.get('id')[:8]}.html",
                            mime="text/html",
                        )


def render_progress_tab():
    """Render progress tab for monitoring running evaluations."""
    st.markdown('<h1 class="main-header">⏳ Progress</h1>', unsafe_allow_html=True)

    evaluations = load_evaluations()
    running_evals = [e for e in evaluations if e.get("status") in ["running", "pending"]]

    if not running_evals:
        st.info("No evaluations currently running.")
        return

    for eval_data in running_evals:
        eval_id = eval_data.get("id")

        st.markdown(f"### {eval_data.get('config', {}).get('name', eval_id[:8])}")

        # Get current status
        status = api.get(f"/evaluate/{eval_id}/status")

        if status:
            # Progress bar
            progress = status.get("progress", 0)
            st.progress(progress / 100)

            # Stats
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Status", status.get("status", "unknown").capitalize())
            with col2:
                st.metric("Progress", f"{progress:.1f}%")
            with col3:
                st.metric("Completed", f"{status.get('completed_tests', 0)}/{status.get('total_tests', 0)}")
            with col4:
                eta = status.get("estimated_time_remaining")
                if eta:
                    st.metric("ETA", f"{eta:.0f}s")
                else:
                    st.metric("ETA", "Calculating...")

            # Current test
            if status.get("current_test"):
                st.caption(f"Currently running: {status.get('current_test')}")

        st.markdown("---")

    # Auto-refresh
    if st.button("🔄 Refresh"):
        st.rerun()

    # Auto-refresh option
    if st.checkbox("Auto-refresh (5s)"):
        st.autorun = True
        import time
        time.sleep(5)
        st.rerun()


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main application."""

    # Render sidebar
    render_sidebar()

    # Main title
    st.markdown('<h1 class="main-header">🤖 LLMOps-Eval Dashboard</h1>', unsafe_allow_html=True)

    # Check API connection
    try:
        health = api.get("/health")
        if health.get("status") == "healthy":
            st.success(f"✅ Connected to API (v{health.get('version', 'unknown')})")
        else:
            st.warning("⚠️ API connection issues")
    except:
        st.error("❌ Cannot connect to API. Please check the API URL in the sidebar.")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Results",
        "⚖️ Comparison",
        "📜 History",
        "⏳ Progress",
    ])

    with tab1:
        render_overview_tab()

    with tab2:
        render_results_tab()

    with tab3:
        render_comparison_tab()

    with tab4:
        render_history_tab()

    with tab5:
        render_progress_tab()


if __name__ == "__main__":
    main()
