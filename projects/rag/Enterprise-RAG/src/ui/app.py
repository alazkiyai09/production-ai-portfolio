# ============================================================
# Enterprise-RAG: Streamlit Demo Interface
# ============================================================
"""
Professional Streamlit interface for the Enterprise-RAG system.

Features:
- Chat interface with message history
- Source display with relevance scores
- Document upload and management
- Configurable settings
- RAGAS evaluation dashboard
"""

import os
import time
from typing import Any, Optional

import requests
import streamlit as st
from typing import List

# ============================================================
# Configuration
# ============================================================

# API Configuration
DEFAULT_API_URL = "http://localhost:8000"
API_URL = os.getenv("RAG_API_URL", DEFAULT_API_URL)

# Page Configuration
st.set_page_config(
    page_title="Enterprise-RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
<style>
    /* Main container */
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    /* Chat messages */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Source cards */
    .source-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
        transition: all 0.3s ease;
    }

    .source-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Score bars */
    .score-container {
        margin: 10px 0;
    }

    .score-label {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 5px;
    }

    .score-bar-bg {
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
    }

    .score-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }

    /* Headers */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px 15px;
    }

    /* File uploader */
    .uploadedFile {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Helper Functions
# ============================================================


def api_request(
    endpoint: str,
    method: str = "GET",
    json_data: Optional[dict] = None,
    files: Optional[dict] = None,
) -> dict:
    """
    Make API request with error handling.

    Args:
        endpoint: API endpoint path
        method: HTTP method
        json_data: JSON request body
        files: Files for upload

    Returns:
        JSON response
    """
    url = f"{API_URL}{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, timeout=60)
            else:
                response = requests.post(url, json=json_data, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return {}
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_URL}")
        st.info("💡 Make sure the API is running: `uvicorn src.api.main:app`")
        return {}
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API error: {e.response.status_code}")
        try:
            error_detail = e.response.json()
            st.error(f"Detail: {error_detail.get('detail', 'Unknown error')}")
        except:
            pass
        return {}
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return {}


def render_score_bar(label: str, score: float, max_score: float = 1.0) -> None:
    """
    Render a score bar with label.

    Args:
        label: Score label
        score: Score value
        max_score: Maximum score
    """
    percentage = min(score / max_score * 100, 100)

    st.markdown(
        f"""
    <div class="score-container">
        <div class="score-label">{label}: {score:.3f}</div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width: {percentage}%"></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_source_card(
    index: int,
    citation: dict,
) -> None:
    """
    Render a source card with preview.

    Args:
        index: Source number
        citation: Citation data
    """
    st.markdown(
        f"""
    <div class="source-card">
        <strong>[{index}] {citation.get('source', 'Unknown')}</strong>
        <p style="font-size: 0.9em; margin: 10px 0;">
            {citation.get('content_preview', '')[:200]}...
        </p>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <small style="color: #666;">Relevance</small>
            <small style="color: #4CAF50; font-weight: bold;">
                {citation.get('relevance_score', 0):.1%}
            </small>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def process_question(prompt: str) -> bool:
    """
    Process a user question through the RAG system.

    This function handles the common logic for processing questions,
    including API request, response display, and source rendering.

    Args:
        prompt: The user's question

    Returns:
        True if successful, False otherwise
    """
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the question
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching and generating..."):
            start_time = time.time()

            # API request
            result = api_request(
                "/api/v1/query",
                method="POST",
                json_data={
                    "question": prompt,
                    "top_k": st.session_state.settings["top_k"],
                    "use_reranking": st.session_state.settings["use_reranking"],
                    "include_history": st.session_state.settings["include_history"],
                },
            )

            elapsed = time.time() - start_time

            if result:
                # Display answer
                st.markdown(result["answer"])

                # Store sources for this message
                msg_id = id(st.session_state.messages[-1])
                st.session_state.sources[msg_id] = result.get("citations", [])

                # Show sources expander
                if result.get("citations"):
                    with st.expander("📚 View Sources", expanded=False):
                        for idx, citation in enumerate(result["citations"], 1):
                            render_source_card(idx, citation)

                # Display timing and model info
                caption = f"⏱️ {result.get('processing_time', elapsed):.2f}s"
                caption += f" | 🤖 {result.get('model_used', 'unknown')}"

                if result.get("rerank_time", 0) > 0:
                    caption += f" | 🔄 Rerank: {result['rerank_time']:.2f}s"

                st.caption(caption)

                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                })
                return True
            else:
                st.error("Failed to get response")
                return False


# ============================================================
# Session State Initialization
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = {}

if "settings" not in st.session_state:
    st.session_state.settings = {
        "top_k": 5,
        "use_reranking": True,
        "include_history": False,
    }

if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    # Header
    st.markdown(
        """
    <div class="sidebar-header">
        <h1 style="margin: 0; font-size: 1.8rem;">⚙️ Settings</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">Configure your RAG experience</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # API Status
    st.subheader("🔌 API Connection")
    try:
        health = api_request("/health")
        if health:
            if health.get("status") == "healthy":
                st.success("✅ API Connected")
            else:
                st.warning("⚠️ API Degraded")
        else:
            st.error("❌ API Unreachable")
    except:
        st.error("❌ API Unreachable")

    st.markdown(f"**URL:** `{API_URL}`")

    st.divider()

    # Retrieval Settings
    st.subheader("🎯 Retrieval Settings")

    top_k = st.slider(
        "Number of Sources",
        min_value=1,
        max_value=10,
        value=st.session_state.settings["top_k"],
        help="Number of document chunks to retrieve and use",
    )
    st.session_state.settings["top_k"] = top_k

    use_reranking = st.checkbox(
        "Use Cross-Encoder Reranking",
        value=st.session_state.settings["use_reranking"],
        help="Apply cross-encoder reranking for better accuracy (slower)",
    )
    st.session_state.settings["use_reranking"] = use_reranking

    include_history = st.checkbox(
        "Include Conversation History",
        value=st.session_state.settings["include_history"],
        help="Include previous messages in context for follow-up questions",
    )
    st.session_state.settings["include_history"] = include_history

    st.divider()

    # Document Upload
    st.subheader("📄 Upload Documents")

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "md", "txt"],
        help="Supported formats: PDF, DOCX, Markdown, Text",
        label_visibility="visible",
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Selected:** `{uploaded_file.name}`")
            st.caption(f"Size: {len(uploaded_file) / 1024:.1f} KB")
        with col2:
            if st.button("Upload", type="primary", use_container_width=True):
                with st.spinner(f"Processing `{uploaded_file.name}`..."):
                    files = {"file": uploaded_file}
                    result = api_request("/api/v1/documents/ingest", method="POST", files=files)

                    if result:
                        st.success(f"✅ Created {result.get('chunks_created', 0)} chunks")
                        st.balloons()
                    else:
                        st.error("❌ Upload failed")

    # Batch Upload Info
    with st.expander("📚 Batch Upload"):
        st.info(
            """
            For batch uploads, use the API directly:

            ```bash
            curl -X POST http://localhost:8000/api/v1/documents/batch-ingest \\
              -F "files=@doc1.pdf" \\
              -F "files=@doc2.docx" \\
              -F "files=@doc3.txt"
            ```
            """
        )

    st.divider()

    # Sample Questions
    st.subheader("💬 Sample Questions")

    sample_questions = [
        "What is the refund policy?",
        "How do I cancel my subscription?",
        "What security measures are in place?",
        "What are the pricing tiers?",
        "How is user data handled?",
    ]

    for question in sample_questions:
        if st.button(question, key=f"sample_{question}", use_container_width=True):
            st.session_state.pending_question = question
            st.rerun()

    # Clear Conversation
    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = {}
        st.success("Conversation cleared")

    # System Info
    st.divider()
    st.subheader("ℹ️ System Info")

    try:
        stats = api_request("/stats")
        if stats:
            vector_stats = stats.get("stats", {}).get("vector_store", {})
            st.metric("Documents", vector_stats.get("total_documents", 0))
            st.metric("Chunks", vector_stats.get("total_chunks", 0))
    except:
        st.info("Stats unavailable")

# ============================================================
# Main Area
# ============================================================

# Header
st.title("🔍 Enterprise-RAG")
st.caption(
    "Production-grade RAG system with hybrid retrieval (dense + sparse) and cross-encoder reranking"
)

# Evaluation Section (show if requested)
if st.session_state.get("show_evaluation"):
    st.subheader("📊 RAGAS Evaluation Results")

    with st.spinner("Running evaluation..."):
        eval_result = api_request(
            "/api/v1/evaluation/run",
            method="POST",
            json_data={"num_samples": 5},
        )

        if eval_result:
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    "Faithfulness",
                    f"{eval_result.get('faithfulness', 0):.3f}",
                    help="Factual consistency with context",
                )

            with col2:
                st.metric(
                    "Answer Relevancy",
                    f"{eval_result.get('answer_relevancy', 0):.3f}",
                    help="Relevance to question",
                )

            with col3:
                st.metric(
                    "Context Precision",
                    f"{eval_result.get('context_precision', 0):.3f}",
                    help="Retrieved context relevance",
                )

            with col4:
                st.metric(
                    "Context Recall",
                    f"{eval_result.get('context_recall', 0):.3f}",
                    help="Ground truth coverage",
                )

            with col5:
                st.metric(
                    "Overall Score",
                    f"{eval_result.get('overall_score', 0):.3f}",
                    help="Average of all metrics",
                )

            # Score bars
            st.subheader("Metric Breakdown")
            render_score_bar("Faithfulness", eval_result.get("faithfulness", 0))
            render_score_bar("Answer Relevancy", eval_result.get("answer_relevancy", 0))
            render_score_bar("Context Precision", eval_result.get("context_precision", 0))
            render_score_bar("Context Recall", eval_result.get("context_recall", 0))

            st.info(f"⏱️ Evaluated {eval_result.get('num_samples', 0)} samples in {eval_result.get('evaluation_time', 0):.2f}s")

    if st.button("← Back to Chat"):
        st.session_state.show_evaluation = False
        st.rerun()

# Main Chat Interface
else:
    # Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant":
                msg_id = id(message)
                if msg_id in st.session_state.sources:
                    with st.expander("📚 View Sources", expanded=False):
                        for idx, citation in st.session_state.sources[msg_id]:
                            render_source_card(idx, citation)

    # Handle pending question from sample questions
    if "pending_question" in st.session_state:
        prompt = st.session_state.pending_question
        del st.session_state.pending_question
        process_question(prompt)

    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        process_question(prompt)

# Bottom Toolbar
st.divider()

col1, col2, col3, col4 = st.columns([2, 2, 2, 4])

with col1:
    if st.button("📊 Run Evaluation", use_container_width=True):
        st.session_state.show_evaluation = True
        st.rerun()

with col2:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = {}
        st.success("History cleared")

with col3:
    # Health check indicator
    try:
        health = api_request("/health")
        if health:
            if health.get("status") == "healthy":
                st.success("✅ API Healthy")
            else:
                st.warning("⚠️ API Degraded")
        else:
            st.error("❌ API Down")
    except:
        st.error("❌ API Down")

with col4:
    st.caption(f"API: `{API_URL}`")

# Footer
st.markdown(
    """
---

<div style="text-align: center; color: #999; font-size: 0.8rem;">
    Enterprise-RAG v1.0.0 | Built with ❤️ using FastAPI, LangChain, and Streamlit
</div>
""",
    unsafe_allow_html=True,
)
