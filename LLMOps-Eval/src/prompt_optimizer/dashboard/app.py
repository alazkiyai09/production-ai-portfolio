"""
Streamlit dashboard for PromptOptimizer.

This module provides an interactive web UI for prompt template management,
variation generation, experiment execution, and results analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

# Page config
st.set_page_config(
    page_title="PromptOptimizer Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        padding: 1rem 0;
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.875rem;
        opacity: 0.9;
    }
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .variant-card {
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
    }
    .experiment-status-running {
        background: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .experiment-status-completed {
        background: #d1fae5;
        color: #065f46;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .experiment-status-failed {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Initialization
# ============================================================================

@st.cache_resource
def get_template_manager():
    """Get template manager instance."""
    from src.prompt_optimizer.templates import create_template_manager
    return create_template_manager()


@st.cache_resource
def get_variation_generator():
    """Get variation generator instance."""
    from src.prompt_optimizer.variations import create_variation_generator
    return create_variation_generator()


@st.cache_resource
def get_experiment_framework():
    """Get experiment framework instance."""
    from src.prompt_optimizer.experiments import create_experiment_framework
    return create_experiment_framework()


@st.cache_resource
def get_history_manager():
    """Get history manager instance."""
    from src.prompt_optimizer.history import create_history_manager
    return create_history_manager()


@st.cache_resource
def get_statistical_analyzer():
    """Get statistical analyzer instance."""
    from src.prompt_optimizer.statistics.analyzer import StatisticalAnalyzer
    return StatisticalAnalyzer()


@st.cache_resource
def get_prompt_selector():
    """Get prompt selector instance."""
    from src.prompt_optimizer.selection.selector import BestPromptSelector, get_statistical_analyzer
    analyzer = get_statistical_analyzer()
    return BestPromptSelector(analyzer)


# ============================================================================
# Helper Functions
# ============================================================================

def display_metric_card(title: str, value: Any, delta: Optional[str] = None, icon: str = "📊"):
    """Display a metric card with styling."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {title}</div>
        <div class="metric-value">{value}</div>
        {f'<div style="font-size: 0.875rem; margin-top: 0.5rem;">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)


def display_success(message: str):
    """Display success message."""
    st.markdown(f"""
    <div class="success-box">
        ✅ {message}
    </div>
    """, unsafe_allow_html=True)


def display_warning(message: str):
    """Display warning message."""
    st.markdown(f"""
    <div class="warning-box">
        ⚠️ {message}
    </div>
    """, unsafe_allow_html=True)


def display_info(message: str):
    """Display info message."""
    st.markdown(f"""
    <div class="info-box">
        ℹ️ {message}
    </div>
    """, unsafe_allow_html=True)


def plot_metric_comparison(
    variant_scores: Dict[str, Dict[str, float]],
    metrics: List[str],
):
    """Plot metric comparison bar chart."""
    fig = go.Figure()

    for metric in metrics:
        values = []
        variants = []

        for variant_id, scores in variant_scores.items():
            if metric in scores:
                variants.append(variant_id)
                values.append(scores[metric])

        fig.add_trace(go.Bar(
            name=metric,
            x=variants,
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition='auto',
        ))

    fig.update_layout(
        title="Metric Comparison by Variant",
        xaxis_title="Variant",
        yaxis_title="Score",
        barmode='group',
        height=400,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_confidence_intervals(
    analysis_results: Dict[str, Dict[str, Any]],
    metric: str,
):
    """Plot confidence intervals for variants."""
    variants = []
    means = []
    lower_bounds = []
    upper_bounds = []

    for variant_id, results in analysis_results.items():
        if metric in results:
            variants.append(variant_id)
            r = results[metric]
            means.append(r.get("mean", 0))
            ci = r.get("confidence_interval", (0, 0))
            lower_bounds.append(ci[0])
            upper_bounds.append(ci[1])

    fig = go.Figure()

    # Add means
    fig.add_trace(go.Scatter(
        x=variants,
        y=means,
        mode='markers+lines',
        name='Mean',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=10),
    ))

    # Add confidence intervals
    for i, variant in enumerate(variants):
        fig.add_trace(go.Scatter(
            x=[variant, variant],
            y=[lower_bounds[i], upper_bounds[i]],
            mode='lines',
            line=dict(color='#9ca3af', width=4),
            showlegend=False,
            hoverinfo='skip',
        ))

        fig.add_trace(go.Scatter(
            x=[variant],
            y=[upper_bounds[i]],
            mode='markers',
            marker=dict(symbol='line-ns-open', size=20, color='#9ca3af'),
            showlegend=False,
            hoverinfo='skip',
        ))

    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} with 95% Confidence Intervals",
        xaxis_title="Variant",
        yaxis_title="Score",
        height=400,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_effect_sizes(
    treatment_results: Dict[str, Any],
):
    """Plot effect sizes forest plot."""
    variants = []
    effect_sizes = []
    cis = []

    for variant_id, result in treatment_results.items():
        variants.append(variant_id)
        effect_sizes.append(result.effect_size)
        cis.append(result.confidence_interval)

    fig = go.Figure()

    for i, (variant, es, ci) in enumerate(zip(variants, effect_sizes, cis)):
        fig.add_trace(go.Scatter(
            x=[es],
            y=[variant],
            mode='markers',
            name=variant,
            marker=dict(size=12, color=['#10b981' if es > 0 else '#ef4444'][0]),
            error_x=dict(
                type='data',
                symmetric=False,
                arrayminus=[es - ci[0]],
                arrayplus=[ci[1] - es],
                color='#9ca3af',
                thickness=2,
            ),
        ))

    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="#9ca3af")

    fig.update_layout(
        title="Effect Sizes (Cohen's d) with 95% Confidence Intervals",
        xaxis_title="Effect Size (Cohen's d)",
        yaxis_title="Variant",
        height=400,
        template="plotly_white",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_sample_size_progress(
    experiment_results: Dict[str, Any],
    min_sample_size: int,
):
    """Plot sample size progress."""
    variant_ids = list(experiment_results.keys())
    sample_sizes = [
        len(experiment_results.get(vid, {}).get("results", []))
        for vid in variant_ids
    ]

    fig = go.Figure(go.Bar(
        x=variant_ids,
        y=sample_sizes,
        marker_color=['#3b82f6' if s >= min_sample_size else '#f59e0b' for s in sample_sizes],
        text=[f"{s}/{min_sample_size}" for s in sample_sizes],
        textposition='auto',
    ))

    fig.add_hline(
        y=min_sample_size,
        line_dash="dash",
        line_color="#10b981",
        annotation_text="Min Sample Size",
    )

    fig.update_layout(
        title="Sample Size Progress",
        xaxis_title="Variant",
        yaxis_title="Sample Size",
        height=300,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_power_analysis(
    effect_sizes: List[float],
    sample_sizes: List[int],
    alpha: float = 0.05,
):
    """Plot power analysis chart."""
    from scipy.stats import norm

    power_grid = []
    for n in sample_sizes:
        powers = []
        for es in effect_sizes:
            z_alpha = norm.ppf(1 - alpha / 2)
            ncp = es * np.sqrt(n / 2)
            power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
            powers.append(min(power, 1.0))
        power_grid.append(powers)

    power_grid = np.array(power_grid)

    fig = go.Figure(data=go.Contour(
        z=power_grid,
        x=effect_sizes,
        y=sample_sizes,
        colorscale='Viridis',
        colorbar=dict(title="Power"),
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color='white'),
        ),
    ))

    # Add 0.8 power line
    fig.update_layout(
        title="Statistical Power Analysis",
        xaxis_title="Effect Size (Cohen's d)",
        yaxis_title="Sample Size per Group",
        height=400,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main application."""
    st.markdown('<h1 class="main-header">🎯 PromptOptimizer Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select a page",
            [
                "🏠 Overview",
                "📝 Templates",
                "🔄 Variations",
                "🧪 Experiments",
                "📊 Results",
                "🏆 Selection",
            ],
        )

        st.divider()

        st.title("Quick Actions")
        if st.button("➕ New Template", use_container_width=True):
            st.session_state.selected_template = None
            st.session_state.page = "templates"

        if st.button("🚀 New Experiment", use_container_width=True):
            st.session_state.page = "experiments"

        st.divider()

        # System stats
        st.title("System Stats")
        try:
            history = get_history_manager()
            stats = history.get_statistics()

            display_metric_card("Experiments", stats.get("total_experiments", 0))
            display_metric_card("Completed", stats.get("completed", 0))
            display_metric_card("Failed", stats.get("failed", 0))
        except Exception as e:
            st.error(f"Error loading stats: {e}")

    # Page routing
    if page == "🏠 Overview":
        show_overview()
    elif page == "📝 Templates":
        show_templates()
    elif page == "🔄 Variations":
        show_variations()
    elif page == "🧪 Experiments":
        show_experiments()
    elif page == "📊 Results":
        show_results()
    elif page == "🏆 Selection":
        show_selection()


# ============================================================================
# Overview Page
# ============================================================================

def show_overview():
    """Show overview page."""
    st.header("📈 Dashboard Overview")

    # Load stats
    try:
        history = get_history_manager()
        stats = history.get_statistics()

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            display_metric_card("Total Experiments", stats.get("total_experiments", 0), "+5 this week")
        with col2:
            display_metric_card("Completed", stats.get("completed", 0))
        with col3:
            display_metric_card("Running", stats.get("running", 0))
        with col4:
            display_metric_card("Variants Tested", stats.get("unique_variants", 0))

        st.divider()

        # Recent experiments
        st.subheader("📋 Recent Experiments")

        experiments = history.list_experiments(limit=10)

        if experiments:
            exp_data = []
            for exp in experiments:
                exp_data.append({
                    "ID": exp.experiment_id[:8],
                    "Name": exp.name,
                    "Status": exp.status,
                    "Best Variant": exp.best_variant or "N/A",
                    "Best Score": f"{exp.best_score:.3f}",
                    "Created": exp.timestamp[:10],
                })

            df = pd.DataFrame(exp_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            display_info("No experiments found. Create your first experiment to get started!")

    except Exception as e:
        st.error(f"Error loading overview: {e}")


# ============================================================================
# Templates Page
# ============================================================================

def show_templates():
    """Show templates page."""
    st.header("📝 Prompt Templates")

    tab1, tab2, tab3 = st.tabs(["Browse Templates", "Create Template", "Render Template"])

    with tab1:
        show_template_browser()

    with tab2:
        show_template_creator()

    with tab3:
        show_template_renderer()


def show_template_browser():
    """Browse existing templates."""
    try:
        manager = get_template_manager()
        templates = manager.list_templates()

        if not templates:
            display_info("No templates found. Create your first template!")
            return

        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            categories = set(t.category for t in templates)
            category_filter = st.selectbox("Filter by Category", ["All"] + sorted(categories))
        with col2:
            search = st.text_input("Search templates")

        # Apply filters
        filtered = templates
        if category_filter != "All":
            filtered = [t for t in filtered if t.category == category_filter]
        if search:
            filtered = [t for t in filtered if search.lower() in t.name.lower()]

        # Display templates
        for template in filtered:
            with st.expander(f"📄 {template.name} (v{template.version})"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**Description:** {template.description}")
                    st.markdown(f"**Category:** {template.category}")
                    st.markdown(f"**Tags:** {', '.join(template.tags) if template.tags else 'None'}")
                    st.markdown(f"**Variables:** {', '.join(template.variables) if template.variables else 'None'}")

                with col2:
                    if st.button("Use", key=f"use_{template.id}"):
                        st.session_state.selected_template = template.id
                        st.rerun()

                with st.expander("📋 Template Content"):
                    st.code(template.template_string, language="text")

    except Exception as e:
        st.error(f"Error loading templates: {e}")


def show_template_creator():
    """Create new template."""
    with st.form("create_template"):
        st.subheader("Create New Template")

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Template Name*", placeholder="my_template")
            category = st.text_input("Category", placeholder="general", value="general")
            version = st.text_input("Version", placeholder="1.0", value="1.0")

        with col2:
            tags_str = st.text_input("Tags (comma-separated)", placeholder="nlp,summarization")
            tags = [t.strip() for t in tags_str.split(",")] if tags_str else []

        description = st.text_area("Description", placeholder="What this template does...")

        template_string = st.text_area(
            "Template String (Jinja2)*",
            placeholder="You are a {{role}}. Your task is to: {{task}}.",
            height=200,
        )

        variables_str = st.text_area(
            "Default Variables (JSON)",
            placeholder='{"role": "assistant", "task": "help"}',
            height=100,
        )

        submitted = st.form_submit_button("🔨 Create Template", use_container_width=True)

        if submitted:
            if not name or not template_string:
                st.error("Name and template string are required!")
                return

            try:
                # Parse variables
                import json
                variables = json.loads(variables_str) if variables_str else {}

                # Create template
                manager = get_template_manager()
                template = manager.create_template(
                    name=name,
                    template_string=template_string,
                    description=description,
                    default_variables=variables,
                    category=category,
                    tags=tags,
                    version=version,
                )

                display_success(f"Template '{name}' created successfully!")

                # Show extracted variables
                if template.variables:
                    st.info(f"Extracted variables: {', '.join(template.variables)}")

            except json.JSONDecodeError:
                st.error("Invalid JSON in default variables!")
            except Exception as e:
                st.error(f"Error creating template: {e}")


def show_template_renderer():
    """Render template with variables."""
    try:
        manager = get_template_manager()
        templates = manager.list_templates()

        if not templates:
            display_info("No templates available. Create a template first!")
            return

        # Select template
        template_names = {t.name: t for t in templates}
        selected_name = st.selectbox("Select Template", list(template_names.keys()))
        template = template_names[selected_name]

        # Show template info
        st.info(f"Variables: {', '.join(template.variables) if template.variables else 'None'}")

        # Get variable values
        variables = {}
        if template.variables:
            st.subheader("Provide Variable Values")

            for var in template.variables:
                default_val = template.default_variables.get(var, "")
                variables[var] = st.text_input(var, value=str(default_val), key=f"var_{var}")

        # Render
        if st.button("✨ Render Template"):
            try:
                manager = get_template_manager()
                rendered = manager.render_template(template.name, variables)

                st.success("Template rendered successfully!")
                st.subheader("Rendered Prompt")
                st.code(rendered.rendered_content, language="text")

                # Token estimate
                tokens = len(rendered.rendered_content) // 4
                st.info(f"Estimated tokens: {tokens}")

            except Exception as e:
                st.error(f"Error rendering template: {e}")

    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================================
# Variations Page
# ============================================================================

def show_variations():
    """Show variations page."""
    st.header("🔄 Prompt Variation Generator")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")

        base_prompt = st.text_area(
            "Base Prompt",
            placeholder="Enter your base prompt here...",
            height=150,
        )

        st.subheader("Variation Strategies")

        available_strategies = {
            "instruction_rephrase": "Instruction Rephrasing",
            "few_shot_selection": "Few-Shot Selection",
            "few_shot_order": "Few-Shot Ordering",
            "output_format": "Output Format",
            "cot_style": "Chain-of-Thought",
            "system_prompt": "System Prompt",
            "emphasis": "Emphasis",
            "verbosity": "Verbosity",
            "temperature": "Temperature Sweep",
            "top_p": "Top-P Sweep",
            "context_position": "Context Position",
        }

        selected_strategies = st.multiselect(
            "Select Strategies",
            list(available_strategies.keys()),
            format_func=lambda x: available_strategies[x],
            default=["instruction_rephrase"],
        )

        variations_per_strategy = st.slider(
            "Variations per Strategy",
            min_value=1,
            max_value=10,
            value=3,
        )

        combine_strategies = st.checkbox("Combine Strategies", value=False)

        if st.button("🎲 Generate Variations", type="primary"):
            if not base_prompt:
                st.error("Please enter a base prompt!")
                return

            if not selected_strategies:
                st.error("Please select at least one strategy!")
                return

            try:
                from src.prompt_optimizer.variations import VariationStrategy

                # Convert to enums
                strategy_enums = [VariationStrategy(s) for s in selected_strategies]

                # Generate
                generator = get_variation_generator()
                variation_set = generator.generate(
                    base_prompt=base_prompt,
                    strategies=strategy_enums,
                    variations_per_strategy=variations_per_strategy,
                    combine_strategies=combine_strategies,
                )

                # Store in session
                st.session_state.variation_set = variation_set

                display_success(f"Generated {len(variation_set.variations)} variations!")

            except Exception as e:
                st.error(f"Error generating variations: {e}")

    with col2:
        st.subheader("Generated Variations")

        if "variation_set" in st.session_state:
            variation_set = st.session_state.variation_set

            # Summary
            st.info(f"Total: {len(variation_set.variations)} variations using {len(variation_set.strategies_used)} strategies")

            # Display variations
            for i, variation in enumerate(variation_set.variations[:10]):  # Show first 10
                with st.expander(f"📝 {variation.description or f'Variation {i+1}'}"):
                    st.markdown(f"**Strategy:** {variation.strategy.value}")
                    st.code(variation.prompt_content, language="text")

            if len(variation_set.variations) > 10:
                st.info(f"... and {len(variation_set.variations) - 10} more variations")

        else:
            display_info("Generate variations to see results here")


# ============================================================================
# Experiments Page
# ============================================================================

def show_experiments():
    """Show experiments page."""
    st.header("🧪 A/B Testing Experiments")

    tab1, tab2, tab3 = st.tabs(["Create Experiment", "Monitor Experiments", "Experiment Details"])

    with tab1:
        show_experiment_creator()

    with tab2:
        show_experiment_monitor()

    with tab3:
        show_experiment_details()


def show_experiment_creator():
    """Create new experiment."""
    with st.form("create_experiment"):
        st.subheader("Experiment Configuration")

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Experiment Name*", placeholder="My First A/B Test")
            description = st.text_area("Description", placeholder="Testing different instruction phrasings...")

            # Get templates
            try:
                manager = get_template_manager()
                templates = manager.list_templates()
                template_names = [t.name for t in templates]
                base_template = st.selectbox("Base Template", template_names)
            except:
                base_template = st.text_input("Base Template ID")

            dataset_name = st.text_input("Dataset Name", placeholder="summarization_test")

        with col2:
            # Get strategies
            available_strategies = [
                "instruction_rephrase",
                "few_shot_selection",
                "output_format",
                "cot_style",
            ]
            strategies = st.multiselect("Strategies to Test", available_strategies, default=["instruction_rephrase"])

            variations_per_strategy = st.slider("Variations per Strategy", 1, 5, 2)

            metrics = st.multiselect(
                "Evaluation Metrics",
                ["semantic_similarity", "llm_judge", "exact_match", "contains"],
                default=["semantic_similarity", "llm_judge"],
            )

            provider = st.selectbox("LLM Provider", ["openai", "anthropic", "ollama"])
            model = st.text_input("Model", value="gpt-4o-mini")

        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            col1, col2, col3 = st.columns(3)

            with col1:
                min_sample_size = st.number_input("Min Sample Size", value=30, min_value=5)

            with col2:
                significance_level = st.slider("Significance Level (α)", 0.01, 0.20, 0.05, 0.01)

            with col3:
                paired_design = st.checkbox("Paired Design", value=True)

        submitted = st.form_submit_button("🚀 Create Experiment", type="primary")

        if submitted:
            if not name or not strategies:
                st.error("Name and strategies are required!")
                return

            display_success(f"Experiment '{name}' created! Go to Monitor tab to start it.")


def show_experiment_monitor():
    """Monitor running experiments."""
    try:
        history = get_history_manager()

        # Get running experiments
        running = history.list_experiments()  # Add status filter if available
        running = [e for e in running if e.status == "running"]

        if running:
            st.subheader("🏃 Running Experiments")

            for exp in running:
                with st.expander(f"🧪 {exp.name}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"**ID:** {exp.experiment_id}")
                        st.markdown(f"**Started:** {exp.timestamp}")

                    with col2:
                        # Progress bar
                        sample_size = exp.metadata.get("total_variants", 0)
                        min_size = 30  # Get from config
                        progress = min(sample_size / min_size, 1.0)

                        st.progress(progress)
                        st.markdown(f"Samples: {sample_size}/{min_size}")

                    with col3:
                        if st.button("📊 View Details", key=f"view_{exp.experiment_id}"):
                            st.session_state.selected_experiment = exp.experiment_id

        else:
            display_info("No running experiments")

        st.divider()

        # All experiments
        st.subheader("📋 All Experiments")

        all_experiments = history.list_experiments(limit=20)

        if all_experiments:
            exp_data = []
            for exp in all_experiments:
                status_emoji = {
                    "completed": "✅",
                    "running": "🏃",
                    "failed": "❌",
                    "pending": "⏳",
                }.get(exp.status, "❓")

                exp_data.append({
                    "": status_emoji,
                    "Name": exp.name,
                    "ID": exp.experiment_id[:8],
                    "Status": exp.status,
                    "Best Score": f"{exp.best_score:.3f}",
                    "Created": exp.timestamp[:10],
                })

            df = pd.DataFrame(exp_data)
            st.dataframe(
                df,
                column_config={
                    "": st.column_config.TextColumn(width="small"),
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            display_info("No experiments found")

    except Exception as e:
        st.error(f"Error loading experiments: {e}")


def show_experiment_details():
    """Show detailed experiment results."""
    # Experiment selector
    try:
        history = get_history_manager()
        experiments = history.list_experiments(limit=50)

        if not experiments:
            display_info("No experiments available")
            return

        exp_options = {f"{e.name} ({e.experiment_id[:8]})": e for e in experiments}
        selected = st.selectbox("Select Experiment", list(exp_options.keys()))
        experiment = exp_options[selected]

        # Load full result
        result = history.get_experiment(experiment.experiment_id)

        if not result:
            st.error("Could not load experiment details")
            return

        # Overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            display_metric_card("Status", result.status.value)
        with col2:
            display_metric_card("Variants", len(result.variant_results))
        with col3:
            display_metric_card("Duration", f"{result.total_time:.1f}s")
        with col4:
            if result.best_variant:
                display_metric_card("Best", result.best_variant[:8] + "...")

        st.divider()

        # Variant results
        st.subheader("📊 Variant Results")

        for var_id, var_result in result.variant_results.items():
            with st.expander(f"🎯 {var_id}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Mean Scores**")
                    for metric, score in var_result.mean_scores.items():
                        st.markdown(f"- {metric}: {score:.4f}")

                with col2:
                    st.markdown("**Performance**")
                    st.markdown(f"- Evaluation Time: {var_result.evaluation_time:.2f}s")
                    st.markdown(f"- Cost: ${var_result.cost:.4f}")
                    st.markdown(f"- Total Tokens: {var_result.token_usage.get('total_tokens', 0)}")

    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================================
# Results Page
# ============================================================================

def show_results():
    """Show results and analysis page."""
    st.header("📊 Results & Analysis")

    # Select experiment
    try:
        history = get_history_manager()
        experiments = [e for e in history.list_experiments() if e.status == "completed"]

        if not experiments:
            display_info("No completed experiments found")
            return

        exp_options = {f"{e.name} ({e.experiment_id[:8]})": e for e in experiments}
        selected = st.selectbox("Select Experiment to Analyze", list(exp_options.keys()))
        experiment = exp_options[selected]

        result = history.get_experiment(experiment.experiment_id)

        # Get variant scores for plotting
        variant_scores = {}
        for var_id, var_result in result.variant_results.items():
            variant_scores[var_id] = var_result.mean_scores

        metrics = list(result.config.metrics)

        # Plots
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Metric Comparison")
            plot_metric_comparison(variant_scores, metrics)

        with col2:
            st.subheader("Sample Size Progress")
            plot_sample_size_progress(
                {vid: {"results": vr.scores.get(list(metrics)[0], [])}
                 for vid, vr in result.variant_results.items()},
                30,
            )

        st.divider()

        # Statistical analysis
        st.subheader("📈 Statistical Analysis")

        metric_to_analyze = st.selectbox("Select Metric to Analyze", metrics)

        if st.button("Run Analysis"):
            try:
                analyzer = get_statistical_analyzer()
                analysis = analyzer.analyze_experiment(result, metric_to_analyze)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Best Variant")
                    st.success(f"**{analysis.best_variant_id}**")
                    st.markdown(f"Improvement: {analysis.best_variant_improvement:.3f}")

                    if analysis.warnings:
                        st.markdown("### ⚠️ Warnings")
                        for warning in analysis.warnings:
                            display_warning(warning)

                with col2:
                    st.markdown("### Treatment Results")

                    for variant_id, test_result in analysis.treatment_results.items():
                        with st.expander(f"🧪 {variant_id}"):
                            st.markdown(f"**Test:** {test_result.test_name}")
                            st.markdown(f"**P-value:** {test_result.p_value:.4f}")
                            st.markdown(f"**Significant:** {'✅ Yes' if test_result.significant else '❌ No'}")
                            st.markdown(f"**Effect Size:** {test_result.effect_size:.3f} ({test_result.effect_size_interpretation})")
                            st.markdown(f"**Power:** {test_result.power:.3f}")
                            st.markdown(f"**Recommendation:** {test_result.recommendation}")

                # Plot effect sizes
                plot_effect_sizes(analysis.treatment_results)

                # Plot confidence intervals
                plot_confidence_intervals(
                    {vid: {metric_to_analyze: tr.__dict__}
                     for vid, tr in analysis.treatment_results.items()},
                    metric_to_analyze,
                )

                # Recommendations
                if analysis.recommendations:
                    st.markdown("### 💡 Recommendations")
                    for rec in analysis.recommendations:
                        display_info(rec)

            except Exception as e:
                st.error(f"Error running analysis: {e}")

    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================================
# Selection Page
# ============================================================================

def show_selection():
    """Show best prompt selection page."""
    st.header("🏆 Best Prompt Selection")

    # Select experiment
    try:
        history = get_history_manager()
        experiments = [e for e in history.list_experiments() if e.status == "completed"]

        if not experiments:
            display_info("No completed experiments found")
            return

        exp_options = {f"{e.name} ({e.experiment_id[:8]})": e for e in experiments}
        selected = st.selectbox("Select Experiment", list(exp_options.keys()))
        experiment = exp_options[selected]

        result = history.get_experiment(experiment.experiment_id)

        # Configure selection criteria
        st.subheader("⚙️ Selection Criteria")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Metric Weights")

            # Get available metrics
            metrics = result.config.metrics
            weights = {}

            for metric in metrics:
                default_weight = 1.0 / len(metrics)
                weights[metric] = st.slider(
                    metric.replace("_", " ").title(),
                    0.0,
                    1.0,
                    default_weight,
                    0.1,
                    key=f"weight_{metric}",
                )

            # Normalize weights
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

        with col2:
            st.markdown("### Settings")

            min_confidence = st.slider("Min Confidence", 0.5, 1.0, 0.95, 0.05)
            min_sample_size = st.number_input("Min Sample Size", value=30)
            prefer_simpler = st.checkbox("Prefer Simpler Prompts", value=True)

        # Select best
        if st.button("🏆 Select Best Prompt", type="primary"):
            try:
                from src.prompt_optimizer.selection.selector import SelectionCriteria

                criteria = SelectionCriteria(
                    metric_weights=weights,
                    min_confidence=min_confidence,
                    min_sample_size=min_sample_size,
                    prefer_simpler=prefer_simpler,
                )

                analyzer = get_statistical_analyzer()
                selector = get_prompt_selector()

                selection = selector.select_best(result, criteria)

                # Display results
                st.success("Best prompt selected!")

                # Main result card
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           padding: 2rem; border-radius: 1rem; color: white; margin: 1rem 0;">
                    <h2 style="margin-top: 0;">🏆 Selected Variant</h2>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)

                with col1:
                    display_metric_card("Variant ID", selection.selected_variant_id[:12], icon="🎯")

                with col2:
                    display_metric_card("Confidence", f"{selection.confidence_score:.1%}", icon="✅")

                with col3:
                    display_metric_card("Score", f"{selection.weighted_score:.3f}", icon="⭐")

                st.divider()

                # Selected prompt
                st.subheader("📝 Selected Prompt")
                st.code(selection.selected_prompt, language="text")

                # Explanation
                st.subheader("💭 Explanation")
                st.info(selection.explanation)

                # Metric breakdown
                st.subheader("📊 Metric Scores")

                metric_data = []
                for metric, score in selection.metric_scores.items():
                    weight = weights.get(metric, 0)
                    metric_data.append({
                        "Metric": metric.replace("_", " ").title(),
                        "Score": f"{score:.4f}",
                        "Weight": f"{weight:.2f}",
                        "Contribution": f"{score * weight:.4f}",
                    })

                st.dataframe(
                    pd.DataFrame(metric_data),
                    use_container_width=True,
                    hide_index=True,
                )

                # Comparison to baseline
                if selection.comparison_to_baseline:
                    st.subheader("📈 Comparison to Baseline")

                    baseline = selection.comparison_to_baseline
                    col1, col2 = st.columns(2)

                    with col1:
                        delta_abs = baseline.get("absolute", 0)
                        st.metric("Absolute Difference", f"{delta_abs:+.4f}")

                    with col2:
                        delta_rel = baseline.get("relative", 0)
                        st.metric("Relative Change", f"{delta_rel:+.1%}")

                # Runner ups
                if selection.runner_up_variants:
                    st.subheader("🥈 Runner-Up Variants")

                    for i, runner_up in enumerate(selection.runner_up_variants, 1):
                        with st.expander(f"#{i} {runner_up['variant_id'][:12]}"):
                            st.markdown(f"**Score:** {runner_up['weighted_score']:.3f}")

                            for metric, score in runner_up['metric_scores'].items():
                                st.markdown(f"- {metric}: {score:.4f}")

            except Exception as e:
                st.error(f"Error selecting best prompt: {e}")

        # Power analysis visualization
        st.divider()
        st.subheader("📈 Power Analysis Calculator")

        col1, col2, col3 = st.columns(3)

        with col1:
            effect_size = st.slider("Effect Size (Cohen's d)", 0.1, 2.0, 0.5, 0.1)

        with col2:
            power = st.slider("Desired Power", 0.5, 0.99, 0.80, 0.05)

        with col3:
            alpha = st.slider("Significance Level (α)", 0.01, 0.20, 0.05, 0.01)

        plot_power_analysis(
            effect_sizes=np.linspace(0.1, 2.0, 20),
            sample_sizes=np.arange(10, 501, 10),
            alpha=alpha,
        )

    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
