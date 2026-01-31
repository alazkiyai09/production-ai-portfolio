"""
Streamlit Chat Interface for DataChat-RAG

Professional chat UI for healthcare AdTech internal Q&A system.
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# =============================================================================
# Configuration
# =============================================================================

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = 120  # seconds

# Page Config
st.set_page_config(
    page_title="DataChat-RAG",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================

def load_custom_css(theme: str = "light") -> str:
    """Load custom CSS for the chat interface."""
    if theme == "dark":
        bg_color = "#1e1e1e"
        sidebar_bg = "#252526"
        text_color = "#e0e0e0"
        user_msg_bg = "#0e639c"
        assistant_msg_bg = "#3a3a3a"
        border_color = "#3e3e42"
    else:
        bg_color = "#ffffff"
        sidebar_bg = "#f8f9fa"
        text_color = "#212529"
        user_msg_bg = "#0d6efd"
        assistant_msg_bg = "#f1f3f5"
        border_color = "#dee2e6"

    return f"""
<style>
/* Main container */
.stApp {{
    background-color: {bg_color};
}}

/* Sidebar */
.css-1d391kg {{
    background-color: {sidebar_bg} !important;
}}

/* Chat messages */
.chat-message {{
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    animation: fadeIn 0.3s ease-in;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.user-message {{
    background-color: {user_msg_bg};
    color: white;
    margin-left: 2rem;
    border-radius: 1rem 1rem 0 1rem;
}}

.assistant-message {{
    background-color: {assistant_msg_bg};
    color: {text_color};
    margin-right: 2rem;
    border-radius: 1rem 1rem 1rem 0;
}}

.message-header {{
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
    opacity: 0.8;
}}

.message-content {{
    line-height: 1.6;
}}

/* SQL Query Box */
.sql-query-box {{
    background-color: #1e1e1e;
    color: #d4d4d4;
    padding: 1rem;
    border-radius: 0.5rem;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.85rem;
    overflow-x: auto;
    margin: 0.5rem 0;
    border-left: 3px solid #007acc;
}}

/* Source Card */
.source-card {{
    background-color: {assistant_msg_bg};
    border: 1px solid {border_color};
    border-radius: 0.5rem;
    padding: 0.75rem;
    margin: 0.5rem 0;
}}

.source-header {{
    font-weight: 600;
    color: #007acc;
    font-size: 0.9rem;
}}

.source-content {{
    font-size: 0.85rem;
    color: {text_color};
    opacity: 0.9;
    margin: 0.5rem 0;
}}

.source-meta {{
    font-size: 0.75rem;
    color: #6c757d;
    font-style: italic;
}}

/* Metrics Cards */
.metric-card {{
    background-color: {assistant_msg_bg};
    border: 1px solid {border_color};
    border-radius: 0.5rem;
    padding: 1rem;
    text-align: center;
}}

.metric-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #007acc;
}}

.metric-label {{
    font-size: 0.85rem;
    color: #6c757d;
}}

/* Follow-up Suggestions */
.suggestion-chip {{
    display: inline-block;
    padding: 0.5rem 1rem;
    background-color: #e7f5ff;
    border: 1px solid #74c0fc;
    border-radius: 2rem;
    margin: 0.25rem;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s;
}}

.suggestion-chip:hover {{
    background-color: #339af0;
    color: white;
}}

/* Status Indicator */
.status-dot {{
    height: 10px;
    width: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 0.5rem;
}}

.status-connected {{
    background-color: #51cf66;
    box-shadow: 0 0 8px #51cf66;
}}

.status-disconnected {{
    background-color: #ff6b6b;
}}

.status-loading {{
    background-color: #ffd43b;
    animation: pulse 1.5s infinite;
}}

@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}

/* Copy Button */
.copy-button {{
    background: none;
    border: none;
    color: #007acc;
    cursor: pointer;
    font-size: 0.85rem;
    padding: 0.25rem 0.5rem;
}}

.copy-button:hover {{
    text-decoration: underline;
}}

/* Query Type Badge */
.query-type-badge {{
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.query-type-sql {{
    background-color: #e7f5ff;
    color: #1971c2;
}}

.query-type-doc {{
    background-color: #fff3bf;
    color: #f59f00;
}}

.query-type-hybrid {{
    background-color: #d3f9d8;
    color: #2f9e44;
}}

/* Confidence Score */
.confidence-bar {{
    width: 100px;
    height: 6px;
    background-color: #e9ecef;
    border-radius: 3px;
    overflow: hidden;
    display: inline-block;
    margin-left: 0.5rem;
}}

.confidence-fill {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
}}
</style>
"""


# =============================================================================
# Session State Management
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    if "api_connected" not in st.session_state:
        st.session_state.api_connected = False

    if "show_sql" not in st.session_state:
        st.session_state.show_sql = {}

    if "show_sources" not in st.session_state:
        st.session_state.show_sources = {}

    if "show_reasoning" not in st.session_state:
        st.session_state.show_reasoning = {}


# =============================================================================
# API Communication
# =============================================================================

def check_api_connection() -> bool:
    """Check if API is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def send_question(
    question: str,
    conversation_id: Optional[str] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """Send a question to the API."""
    url = f"{API_BASE_URL}/chat"

    payload = {
        "question": question,
        "conversation_id": conversation_id,
    }

    try:
        response = requests.post(url, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except requests.exceptions.RequestException as e:
        return {"error": f"API error: {str(e)}"}


def get_health_status() -> Dict[str, Any]:
    """Get API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {"status": "unhealthy", "components": []}


# =============================================================================
# UI Components
# =============================================================================

def render_status_indicator():
    """Render connection status indicator."""
    if not st.session_state.api_connected:
        st.markdown(
            f"""
            <span class="status-dot status-disconnected"></span>
            <span style="color: #ff6b6b;">API Disconnected</span>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <span class="status-dot status-connected"></span>
            <span style="color: #51cf66;">API Connected</span>
            """,
            unsafe_allow_html=True,
        )


def render_message(message_idx: int, message: Dict[str, Any]):
    """Render a single chat message."""
    role = message["role"]
    content = message["content"]

    if role == "user":
        st.markdown(
            f"""
            <div class="chat-message user-message">
                <div class="message-header">👤 You</div>
                <div class="message-content">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Assistant message with metadata
        query_type = message.get("query_type", "UNKNOWN")
        confidence = message.get("confidence", 0.0)
        sql_query = message.get("sql_query")
        doc_sources = message.get("doc_sources", [])
        reasoning = message.get("reasoning", "")
        suggested_followup = message.get("suggested_followup", [])
        processing_time = message.get("processing_time_seconds", 0)

        # Query type badge color
        badge_class = {
            "SQL_QUERY": "query-type-sql",
            "DOC_SEARCH": "query-type-doc",
            "HYBRID": "query-type-hybrid",
        }.get(query_type, "")

        # Confidence color
        confidence_color = "#51cf66" if confidence > 0.8 else "#ffd43b" if confidence > 0.5 else "#ff6b6b"

        # Build message HTML
        html_parts = ["""
        <div class="chat-message assistant-message">
            <div class="message-header">
                🤖 DataChat &nbsp;
        """]

        # Query type badge
        html_parts.append(f"""
                <span class="query-type-badge {badge_class}">{query_type.replace('_', ' ')}</span>
                &nbsp;
                <span style="font-size: 0.75rem; opacity: 0.7;">
                    {confidence:.0%} confidence
                    <span class="confidence-bar">
                        <span class="confidence-fill" style="width: {confidence * 100}%; background-color: {confidence_color};"></span>
                    </span>
                </span>
                <span style="font-size: 0.75rem; opacity: 0.7; float: right;">
                    ⏱ {processing_time:.2f}s
                </span>
            </div>
            <div class="message-content">
        """)

        # Main answer
        # Format markdown-style code blocks
        formatted_content = content
        if "```" in content:
            # Replace code blocks with styled divs
            import re
            formatted_content = re.sub(
                r'```(\w+)?\n(.*?)```',
                r'<div class="sql-query-box">\2</div>',
                formatted_content,
                flags=re.DOTALL
            )

        html_parts.append(f"{formatted_content}")

        # SQL Query (expandable)
        if sql_query:
            key = f"sql_{message_idx}"
            if st.session_state.get("show_sql", {}).get(key, False):
                sql_icon = "▼"
            else:
                sql_icon = "▶"

            html_parts.append(f"""
                <details {"open" if st.session_state.get("show_sql", {}).get(key, False) else ""}>
                    <summary style="cursor: pointer; padding: 0.5rem 0; font-weight: 600;">
                        {sql_icon} SQL Query
                    </summary>
                    <div class="sql-query-box">
                        <code>{sql_query}</code>
                    </div>
                    <button class="copy-button" onclick="navigator.clipboard.writeText(`{sql_query.replace('`', '\\`')}`); this.textContent='✓ Copied!'; setTimeout(() => this.textContent='📋 Copy', 2000);">
                        📋 Copy
                    </button>
                </details>
            """)

        # Document Sources (expandable)
        if doc_sources:
            key = f"sources_{message_idx}"
            html_parts.append(f"""
                <details {"open" if st.session_state.get("show_sources", {}).get(key, False) else ""} style="margin-top: 1rem;">
                    <summary style="cursor: pointer; padding: 0.5rem 0; font-weight: 600;">
                        📚 {len(doc_sources)} Source{'' if len(doc_sources) == 1 else 's'}
                    </summary>
            """)

            for i, source in enumerate(doc_sources, 1):
                html_parts.append(f"""
                    <div class="source-card">
                        <div class="source-header">[{i}] {source.get('source', 'Unknown')}</div>
                        <div class="source-content">{source.get('content', '')[:200]}...</div>
                        <div class="source-meta">
                            Type: {source.get('doc_type', 'unknown')} |
                            Relevance: {source.get('relevance', 0):.2%}
                        </div>
                    </div>
                """)

            html_parts.append("</details>")

        # Reasoning (expandable)
        if reasoning:
            key = f"reasoning_{message_idx}"
            html_parts.append(f"""
                <details style="margin-top: 1rem;">
                    <summary style="cursor: pointer; padding: 0.5rem 0; font-weight: 600;">
                        🧠 Query Routing Reasoning
                    </summary>
                    <p style="font-size: 0.9rem; opacity: 0.8;">{reasoning}</p>
                </details>
            """)

        # Suggested follow-up questions
        if suggested_followup:
            html_parts.append("""
                <div style="margin-top: 1rem;">
                    <div style="font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem;">💡 Follow-up:</div>
            """)
            for suggestion in suggested_followup:
                # Remove emoji for cleaner display
                clean_suggestion = suggestion.replace("💡 ", "").strip()
                html_parts.append(f'<span class="suggestion-chip">{clean_suggestion}</span>')
            html_parts.append("</div>")

        html_parts.append("</div></div>")

        st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with controls and sample questions."""
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 1.8rem; margin-bottom: 0.5rem;">💬 DataChat-RAG</h1>
            <p style="font-size: 0.85rem; opacity: 0.7;">Healthcare AdTech Intelligence</p>
        </div>
        """, unsafe_allow_html=True)

        # Connection status
        st.markdown("---")
        st.markdown("### Connection")
        render_status_indicator()

        # Theme toggle
        theme = st.selectbox(
            "🎨 Theme",
            ["light", "dark"],
            index=0 if st.session_state.theme == "light" else 1,
        )
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()

        st.markdown("---")

        # Conversation management
        st.markdown("### 💬 Conversation")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_id = None
                st.session_state.show_sql = {}
                st.session_state.show_sources = {}
                st.session_state.show_reasoning = {}
                st.rerun()

        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                if st.session_state.conversation_id:
                    try:
                        requests.delete(f"{API_BASE_URL}/conversations/{st.session_state.conversation_id}", timeout=5)
                    except Exception:
                        pass
                st.session_state.messages = []
                st.rerun()

        # Query type filter
        st.markdown("---")
        st.markdown("### 🔍 Query Type")
        query_filter = st.radio(
            "Filter responses by:",
            ["All", "SQL Only", "Documents Only", "Hybrid Only"],
            label_visibility="collapsed",
        )

        # Stats
        if st.session_state.messages:
            st.markdown("---")
            st.markdown("### 📊 Session Stats")
            message_count = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.metric("Questions Asked", message_count)

            # Count query types
            sql_count = sum(1 for m in st.session_state.messages if m.get("query_type") == "SQL_QUERY")
            doc_count = sum(1 for m in st.session_state.messages if m.get("query_type") == "DOC_SEARCH")
            hybrid_count = sum(1 for m in st.session_state.messages if m.get("query_type") == "HYBRID")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("SQL Queries", sql_count)
                st.metric("Doc Searches", doc_count)
            with col2:
                st.metric("Hybrid", hybrid_count)

        st.markdown("---")

        # Sample questions
        st.markdown("### 🎯 Try These")

        sample_questions = [
            "What was our total ad spend last month?",
            "Which campaigns have the highest CTR?",
            "What are our HIPAA compliance requirements for healthcare ads?",
            "Compare performance of pharma vs medical device campaigns",
            "What's the process for getting an ad creative approved?",
            "Why is campaign X underperforming compared to benchmarks?",
            "Top 5 campaigns by conversions this quarter",
            "What are our targeting policies for pharmaceutical ads?",
        ]

        for question in sample_questions:
            if st.button(question, key=f"sample_{question[:20]}", use_container_width=True):
                st.session_state.input_question = question
                st.rerun()

        st.markdown("---")

        # Health status
        st.markdown("### 🏥 System Health")
        try:
            health = get_health_status()
            for component in health.get("components", []):
                status_color = {
                    "healthy": "🟢",
                    "degraded": "🟡",
                    "unhealthy": "🔴",
                }.get(component.get("status", "unknown"), "⚪")

                st.markdown(f"{status_color} **{component.get('name', 'Unknown')}**")
                st.caption(component.get("message", ""))
        except Exception:
            st.caption("Unable to fetch health status")

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.75rem; opacity: 0.6;">
            DataChat-RAG v1.0.0<br>
            Healthcare AdTech Intelligence System
        </div>
        """, unsafe_allow_html=True)


def render_chat_input():
    """Render chat input area."""
    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)

    # Check for pre-filled question from sidebar
    default_value = st.session_state.pop("input_question", "")

    with st.container():
        col1, col2 = st.columns([6, 1])

        with col1:
            user_input = st.text_input(
                "Ask a question about campaigns, metrics, policies, or anything...",
                value=default_value,
                placeholder="e.g., What was our average CTR last week?",
                label_visibility="collapsed",
                key="chat_input",
            )

        with col2:
            send_button = st.button("Send 📤", use_container_width=True, type="primary")

        if send_button and user_input:
            return user_input
        elif user_input and st.session_state.get("submit_on_enter", False):
            return user_input

    return None


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main Streamlit application."""
    # Initialize session state
    init_session_state()

    # Apply custom CSS
    st.markdown(load_custom_css(st.session_state.theme), unsafe_allow_html=True)

    # Check API connection
    if not st.session_state.api_connected:
        st.session_state.api_connected = check_api_connection()

    # Render sidebar
    render_sidebar()

    # Main content area
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem;">
        <h1 style="font-size: 2rem; margin-bottom: 0.5rem;">💬 DataChat-RAG</h1>
        <p style="font-size: 1rem; opacity: 0.7;">
            Healthcare AdTech Internal Q&A System
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Connection warning
    if not st.session_state.api_connected:
        st.warning(
            "⚠️ **API Not Connected** - Unable to reach the backend. "
            f"Make sure the API is running at `{API_BASE_URL}`"
        )
        st.info("💡 Start the API with: `uvicorn src.api.main:app --reload`")

    # Render message history
    if st.session_state.messages:
        for idx, message in enumerate(st.session_state.messages):
            render_message(idx, message)
    else:
        # Welcome message
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; opacity: 0.6;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">👋</div>
            <h3>Welcome to DataChat-RAG!</h3>
            <p>I can help you with:</p>
            <ul style="list-style: none; padding: 0;">
                <li>📊 Campaign performance metrics and analysis</li>
                <li>📋 Company policies, procedures, and guidelines</li>
                <li>🔍 Hybrid insights combining data and knowledge</li>
            </ul>
            <p style="margin-top: 1rem;">Try a sample question from the sidebar or ask your own!</p>
        </div>
        """, unsafe_allow_html=True)

    # Chat input
    user_input = render_chat_input()

    # Process user input
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })

        # Display user message
        render_message(len(st.session_state.messages) - 1, st.session_state.messages[-1])

        # Show loading indicator
        with st.spinner("🤔 Thinking..."):
            # Send to API
            response = send_question(
                user_input,
                st.session_state.conversation_id,
            )

        if "error" in response:
            # Error response
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ {response['error']}",
                "query_type": "ERROR",
                "confidence": 0.0,
            })
        else:
            # Successful response
            assistant_message = {
                "role": "assistant",
                "content": response.get("answer", ""),
                "query_type": response.get("query_type", "UNKNOWN"),
                "confidence": response.get("confidence", 0.0),
                "sql_query": response.get("sql_query"),
                "sql_results": response.get("sql_results"),
                "doc_sources": response.get("doc_sources", []),
                "reasoning": response.get("reasoning", ""),
                "suggested_followup": response.get("suggested_followup", []),
                "processing_time_seconds": response.get("processing_time_seconds", 0.0),
            }

            # Download SQL results if available
            if assistant_message["sql_results"]:
                results = assistant_message["sql_results"]
                if results.get("results"):
                    df = pd.DataFrame(results["results"])
                    csv = df.to_csv(index=False)

                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"datachat_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key=f"download_{len(st.session_state.messages)}",
                    )

            st.session_state.messages.append(assistant_message)

            # Update conversation ID
            st.session_state.conversation_id = response.get("conversation_id")

        # Rerun to display the response
        st.rerun()


if __name__ == "__main__":
    main()
