"""
Streamlit demo interface for AgenticFlow.

This module provides a user-friendly web interface for the multi-agent
workflow system with real-time progress tracking and visualized outputs.
"""

import asyncio
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import streamlit as st

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="AgenticFlow",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS Styling
# =============================================================================

CUSTOM_CSS = """
<style>
    /* Main container */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Agent cards */
    .agent-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        transition: all 0.3s ease;
    }

    .agent-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }

    .agent-card.active {
        border-left: 4px solid #667eea;
        background: #f8f9ff;
    }

    .agent-card.complete {
        border-left: 4px solid #4caf50;
        background: #f0fff4;
    }

    .agent-card.error {
        border-left: 4px solid #f44336;
        background: #fff5f5;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .status-pending { background: #e0e0e0; color: #616161; }
    .status-running { background: #fff3cd; color: #856404; }
    .status-complete { background: #d4edda; color: #155724; }
    .status-error { background: #f8d7da; color: #721c24; }

    /* Timeline */
    .timeline {
        position: relative;
        padding-left: 2rem;
    }

    .timeline-item {
        position: relative;
        padding-bottom: 1.5rem;
    }

    .timeline-item::before {
        content: '';
        position: absolute;
        left: -2rem;
        top: 0.5rem;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #667eea;
        border: 2px solid white;
        box-shadow: 0 0 0 3px #667eea;
    }

    .timeline-item::after {
        content: '';
        position: absolute;
        left: -1.55rem;
        top: 1.5rem;
        width: 2px;
        height: calc(100% - 0.5rem);
        background: #e0e0e0;
    }

    .timeline-item:last-child::after {
        display: none;
    }

    /* Progress bar */
    .progress-container {
        background: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
        height: 24px;
    }

    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }

    .metric-card .label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }

    /* Code block styling */
    .stCode {
        border-radius: 8px;
    }

    /* Info box */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .error-box {
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# API Configuration
# =============================================================================

API_BASE_URL = "http://localhost:8000/api/v1"

# =============================================================================
# Session State Initialization
# =============================================================================

if "workflow_id" not in st.session_state:
    st.session_state.workflow_id = None

if "workflow_status" not in st.session_state:
    st.session_state.workflow_status = None

if "workflow_result" not in st.session_state:
    st.session_state.workflow_result = None

if "agent_outputs" not in st.session_state:
    st.session_state.agent_outputs = {}

if "show_results" not in st.session_state:
    st.session_state.show_results = False


# =============================================================================
# Helper Functions
# =============================================================================

def api_request(
    endpoint: str,
    method: str = "GET",
    json_data: Optional[dict] = None,
) -> Optional[dict]:
    """
    Make an API request to the AgenticFlow backend.

    Args:
        endpoint: API endpoint path
        method: HTTP method (GET, POST, DELETE)
        json_data: JSON payload for POST requests

    Returns:
        Response JSON or None if error
    """
    url = f"{API_BASE_URL}{endpoint}"

    try:
        if method == "GET":
            response = httpx.get(url, timeout=30.0)
        elif method == "POST":
            response = httpx.post(url, json=json_data, timeout=30.0)
        elif method == "DELETE":
            response = httpx.delete(url, timeout=30.0)
        else:
            st.error(f"Unsupported method: {method}")
            return None

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None

    except httpx.ConnectError:
        st.error("❌ Cannot connect to API. Make sure the backend is running on http://localhost:8000")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AgenticFlow</h1>
        <p>Multi-Agent Workflow System with LangGraph</p>
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(status: str) -> str:
    """Render a status badge HTML."""
    badge_class = f"status-{status.lower()}"
    return f'<span class="status-badge {badge_class}">{status.upper()}</span>'


def render_progress_bar(progress_percent: float) -> str:
    """Render a custom progress bar."""
    return f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {progress_percent}%">
            {progress_percent:.0f}%
        </div>
    </div>
    """


def render_agent_card(
    agent_name: str,
    status: str,
    output: Optional[str] = None,
    duration: Optional[float] = None,
) -> None:
    """Render an agent status card."""
    status_class = "active" if status == "running" else ("complete" if status == "complete" else "error")

    icons = {
        "Planner": "📋",
        "Researcher": "🔍",
        "Analyzer": "📊",
        "Writer": "✍️",
        "Reviewer": "👁️",
    }

    icon = icons.get(agent_name, "🤖")

    st.markdown(f"""
    <div class="agent-card {status_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 1.5rem;">{icon}</span>
                <strong style="margin-left: 0.5rem;">{agent_name}</strong>
            </div>
            <div>{render_status_badge(status)}</div>
        </div>
        {f'<p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">⏱️ {duration:.2f}s</p>' if duration else ''}
    </div>
    """, unsafe_allow_html=True)


def render_timeline(events: list[dict]) -> None:
    """Render a timeline of events."""
    st.markdown('<div class="timeline">', unsafe_allow_html=True)

    for event in events:
        st.markdown(f"""
        <div class="timeline-item">
            <strong>{event.get('node', 'Unknown')}</strong>
            <span style="color: #666; font-size: 0.9rem;"> - {event.get('timestamp', '')}</span>
            {f"<p>{event.get('message', '')}</p>" if event.get('message') else ''}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


# =============================================================================
# Main App Layout
# =============================================================================

def main():
    """Main Streamlit application."""
    render_header()

    # Create columns for main layout
    col1, col2 = st.columns([1, 3])

    # =============================================================================
    # Sidebar - Task Input
    # =============================================================================

    with col1:
        st.markdown("### 📝 New Task")

        with st.form("task_form"):
            task = st.text_area(
                "Task Description",
                placeholder="Enter your task here...",
                height=100,
                help="Describe what you want the multi-agent system to accomplish",
            )

            task_type = st.selectbox(
                "Task Type",
                options=["general", "research", "analysis", "content_creation"],
                index=0,
                help="Select the type of task for optimal agent routing",
            )

            with st.expander("Advanced Settings"):
                model_name = st.selectbox(
                    "Model",
                    options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
                    index=0,
                )

                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.1,
                    help="Higher values make output more random",
                )

                max_iterations = st.number_input(
                    "Max Revisions",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Maximum number of revision cycles",
                )

                task_context = st.text_area(
                    "Additional Context",
                    placeholder="Any additional context or constraints...",
                    height=80,
                )

            submitted = st.form_submit_button(
                "🚀 Start Workflow",
                use_container_width=True,
                type="primary",
            )

        if submitted and task:
            # Start workflow
            with st.spinner("Starting workflow..."):
                request_data = {
                    "task": task,
                    "task_type": task_type,
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_iterations": max_iterations,
                }

                if task_context:
                    request_data["task_context"] = task_context

                result = api_request("/workflow/start", method="POST", json_data=request_data)

                if result:
                    st.session_state.workflow_id = result["workflow_id"]
                    st.session_state.show_results = False
                    st.success(f"✅ Workflow started! ID: {result['workflow_id'][:8]}...")
                    st.rerun()

        # Health check
        st.markdown("---")
        st.markdown("### 🔌 API Status")

        health = api_request("/health")
        if health:
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ API Online</strong><br>
                Version: {health.get('version', 'N/A')}<br>
                Active Workflows: {health.get('active_workflows', 0)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="error-box">
                <strong>❌ API Offline</strong><br>
                Start the backend: <code>python src/api/main.py</code>
            </div>
            """, unsafe_allow_html=True)

        # List workflows
        if st.button("📋 Refresh Workflows", use_container_width=True):
            workflows = api_request("/workflows")
            if workflows:
                st.session_state.all_workflows = workflows["workflows"]

        if "all_workflows" in st.session_state and st.session_state.all_workflows:
            st.markdown("### 📁 Recent Workflows")
            for wf in st.session_state.all_workflows[:5]:
                with st.expander(f"{wf['task'][:50]}... ({wf['status']})"):
                    st.code(f"ID: {wf['id']}\nType: {wf['task_type']}\nCreated: {wf['created_at']}")

    # =============================================================================
    # Main Content Area
    # =============================================================================

    with col2:
        # No active workflow
        if not st.session_state.workflow_id:
            st.markdown("""
            <div class="info-box">
                <h3>👋 Welcome to AgenticFlow!</h3>
                <p>
                This is a multi-agent AI system that uses LangGraph to orchestrate
                specialized agents through complex workflows.
                </p>
                <p>
                <strong>Agents:</strong>
                📋 Planner → 🔍 Researcher → 📊 Analyzer → ✍️ Writer → 👁️ Reviewer
                </p>
                <p>
                Enter a task in the sidebar to get started!
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🎯 Example Tasks")
            example_tasks = [
                ("Research Task", "Research the latest developments in quantum computing and explain their potential applications"),
                ("Analysis Task", "Analyze the benefits and drawbacks of remote work for tech companies"),
                ("Content Creation", "Write a blog post about the future of artificial intelligence in healthcare"),
            ]

            for title, task_text in example_tasks:
                with st.expander(f"📌 {title}"):
                    st.code(task_text, language="text")
                    if st.button(f"Use This Task", key=f"example_{title}"):
                        st.session_state.example_task = task_text
                        st.info("✨ Task loaded! Click the sidebar to adjust settings, then Start Workflow.")

        # Active workflow - show progress/results
        else:
            workflow_id = st.session_state.workflow_id

            # Status bar
            status_data = api_request(f"/workflow/{workflow_id}/status")

            if status_data:
                progress = status_data.get("progress", {})
                progress_percent = progress.get("progress_percent", 0)
                wf_status = status_data.get("status", "pending")

                st.markdown(f"### 📊 Workflow Status: {render_status_badge(wf_status)}")
                st.markdown(render_progress_bar(progress_percent), unsafe_allow_html=True)

                # Metrics row
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Steps", f"{progress.get('steps_completed', 0)}/{progress.get('total_steps', 0)}")
                with m2:
                    st.metric("Iterations", progress.get("iteration_count", 0))
                with m3:
                    st.metric("Agents", progress.get('agents_executed', 0))
                with m4:
                    st.metric("Research", "✅" if progress.get('has_research') else "⏳")
                with m5:
                    st.metric("Elapsed", format_duration(progress.get('elapsed_time_seconds', 0)))

                st.markdown("---")

                # Agent execution timeline
                st.markdown("### 🕐 Execution Timeline")

                # Simulate agent statuses (in real implementation, get from API)
                agent_statuses = {
                    "Planner": "complete" if progress_percent >= 10 else "running" if progress_percent < 10 else "pending",
                    "Researcher": "complete" if progress_percent >= 35 else "running" if 10 <= progress_percent < 35 else "pending",
                    "Analyzer": "complete" if progress_percent >= 60 else "running" if 35 <= progress_percent < 60 else "pending",
                    "Writer": "complete" if progress_percent >= 85 else "running" if 60 <= progress_percent < 85 else "pending",
                    "Reviewer": "complete" if wf_status == "complete" else "running" if progress_percent >= 85 else "pending",
                }

                for agent, status in agent_statuses.items():
                    if status != "pending":
                        render_agent_card(agent, status)

                # Detailed outputs (expandable)
                st.markdown("---")
                st.markdown("### 📦 Agent Outputs")

                # Planner output
                if progress_percent >= 10:
                    with st.expander("📋 Planner - Execution Plan", expanded=False):
                        st.info("The Planner agent has broken down your task into actionable steps.")
                        st.markdown("""
                        1. **Research Phase** - Gather information from multiple sources
                        2. **Analysis Phase** - Identify patterns and key insights
                        3. **Writing Phase** - Create comprehensive content
                        4. **Review Phase** - Evaluate quality and completeness
                        """)

                # Researcher output
                if progress.get('has_research'):
                    with st.expander("🔍 Researcher - Gathered Information", expanded=False):
                        st.success(f"✅ Found {progress.get('research_count', 5)} relevant sources")
                        st.markdown("""
                        - Web search completed successfully
                        - Multiple sources analyzed
                        - Key information extracted
                        """)

                # Analyzer output
                if progress.get('has_analysis'):
                    with st.expander("📊 Analyzer - Insights & Findings", expanded=False):
                        st.success("✅ Key findings identified")
                        st.markdown("""
                        - **Pattern Recognition**: Common themes identified across sources
                        - **Data Analysis**: Quantitative insights extracted
                        - **Trend Analysis**: Emerging patterns highlighted
                        """)

                # Writer output
                if progress.get('has_draft'):
                    with st.expander("✍️ Writer - Content Draft", expanded=True):
                        st.info(f"📝 Draft v{progress.get('revision_count', 1)} created")
                        if progress.get('draft_preview'):
                            st.markdown(progress['draft_preview'])
                        else:
                            st.markdown("""
                            *Draft content is being generated...*
                            """)
                            st.spinner("Writer is creating content...")

                # Reviewer output
                if progress.get('is_reviewed'):
                    with st.expander("👁️ Reviewer - Evaluation", expanded=False):
                        approval = status_data.get('approval', 'pending')
                        if approval == 'approved':
                            st.success("✅ Content Approved - Ready for delivery!")
                        elif approval == 'needs_revision':
                            st.warning("🔄 Revision Requested - Refining content...")
                        else:
                            st.error("❌ Content Rejected - Significant revisions needed")

                # Final result
                if wf_status == "complete":
                    result = api_request(f"/workflow/{workflow_id}/result")

                    if result:
                        st.markdown("---")
                        st.markdown("### 🎉 Final Result")

                        # Success box
                        st.markdown(f"""
                        <div class="success-box">
                            <strong>✅ Workflow Complete!</strong><br>
                            Duration: {format_duration(result.get('duration_seconds', 0))}<br>
                            Revisions: {result.get('revision_count', 0)}<br>
                            Status: {result.get('approval_status', 'unknown').title()}
                        </div>
                        """, unsafe_allow_html=True)

                        # Final output
                        st.markdown("#### 📄 Final Output")
                        final_output = result.get('final_output', '')

                        if final_output:
                            st.markdown(final_output)

                            # Download button
                            st.download_button(
                                label="📥 Download Result",
                                data=final_output,
                                file_name=f"agenticflow_result_{workflow_id[:8]}.md",
                                mime="text/markdown",
                                use_container_width=True,
                            )
                        else:
                            st.warning("No final output available")

                        # Feedback section
                        st.markdown("---")
                        st.markdown("### 💬 Feedback")

                        feedback_list = result.get('feedback', [])
                        if feedback_list:
                            for i, item in enumerate(feedback_list, 1):
                                st.markdown(f"{i}. {item}")
                        else:
                            st.info("No feedback items - content was approved!")

                # Action buttons
                st.markdown("---")
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    if st.button("🔄 Refresh", use_container_width=True):
                        st.rerun()

                with col_b:
                    if st.button("📋 Copy ID", use_container_width=True):
                        st.clipboard_copy(workflow_id)
                        st.success("✅ Copied to clipboard!")

                with col_c:
                    if st.button("🆕 New Task", use_container_width=True):
                        # Reset session state
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        st.rerun()

            else:
                st.error("❌ Could not fetch workflow status")
                if st.button("🔄 Reset", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()


# =============================================================================
# Footer
# =============================================================================

def render_footer():
    """Render the application footer."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>
            <strong>AgenticFlow</strong> - Multi-Agent Workflow System with LangGraph<br>
            Built with Streamlit • LangGraph • LangChain • FastAPI
        </p>
        <p style="font-size: 0.85rem;">
            <a href="/docs" target="_blank">API Documentation</a> •
            <a href="https://github.com/langchain-ai/langgraph" target="_blank">LangGraph</a> •
            <a href="https://github.com/streamlit/streamlit" target="_blank">Streamlit</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
    render_footer()
