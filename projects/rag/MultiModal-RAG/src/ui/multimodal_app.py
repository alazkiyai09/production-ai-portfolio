"""
Multi-Modal Streamlit UI for RAG System.

Features:
- Multi-modal query interface
- Content type filters
- Image and table previews
- Visual results display
- Document upload with extraction
"""

import io
import base64
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

import streamlit as st
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Multi-Modal RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .citation-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .image-preview {
        max-width: 100%;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .table-container {
        overflow-x: auto;
        border-radius: 8px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# ==================== Session State ====================

def init_session_state():
    """Initialize session state variables."""
    if 'api_url' not in st.session_state:
        st.session_state.api_url = "http://localhost:8000"
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'current_response' not in st.session_state:
        st.session_state.current_response = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []


init_session_state()


# ==================== API Client ====================

import requests


class MultiModalRAGClient:
    """Client for multi-modal RAG API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def query(
        self,
        question: str,
        content_types: List[str] = None,
        include_images: bool = True,
        include_tables: bool = True,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Send multi-modal query."""
        url = f"{self.base_url}/api/v1/multimodal/query"

        payload = {
            "question": question,
            "content_types": content_types or ["text", "image", "table"],
            "include_images": include_images,
            "include_tables": include_tables,
            "top_k": top_k,
        }

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def query_with_image(
        self,
        question: str,
        image_bytes: bytes,
        include_text: bool = True,
        include_tables: bool = True,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Query with image upload."""
        url = f"{self.base_url}/api/v1/multimodal/query/image"

        files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
        data = {
            "question": question,
            "include_text": include_text,
            "include_tables": include_tables,
            "top_k": top_k,
        }

        response = requests.post(url, files=files, data=data, timeout=60)
        response.raise_for_status()
        return response.json()

    def ingest_document(
        self,
        file_path: str,
        extract_images: bool = True,
        extract_tables: bool = True,
        generate_captions: bool = True,
        extract_ocr: bool = True,
    ) -> Dict[str, Any]:
        """Ingest a document."""
        url = f"{self.base_url}/api/v1/multimodal/ingest"

        with open(file_path, 'rb') as f:
            files = {"file": (Path(file_path).name, f, "application/octet-stream")}
            data = {
                "extract_images": extract_images,
                "extract_tables": extract_tables,
                "generate_captions": generate_captions,
                "extract_ocr": extract_ocr,
            }

            response = requests.post(url, files=files, data=data, timeout=300)
            response.raise_for_status()
            return response.json()

    def get_image(self, image_id: str) -> Dict[str, Any]:
        """Get image data."""
        url = f"{self.base_url}/api/v1/multimodal/images/{image_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def get_table(self, table_id: str) -> Dict[str, Any]:
        """Get table data."""
        url = f"{self.base_url}/api/v1/multimodal/tables/{table_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def list_images(self, skip: int = 0, limit: int = 50) -> Dict[str, Any]:
        """List all images."""
        url = f"{self.base_url}/api/v1/multimodal/images"
        params = {"skip": skip, "limit": limit}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def list_tables(self, skip: int = 0, limit: int = 50) -> Dict[str, Any]:
        """List all tables."""
        url = f"{self.base_url}/api/v1/multimodal/tables"
        params = {"skip": skip, "limit": limit}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        url = f"{self.base_url}/api/v1/multimodal/stats"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


# ==================== Helper Functions ====================

def render_image_citation(citation: Dict[str, Any], idx: int):
    """Render an image citation with preview."""
    st.markdown(f"""
    <div class="citation-card">
        <strong>📖 Image Source {idx}</strong><br/>
        <strong>File:</strong> {Path(citation.get('source', '')).name} |
        <strong>Page:</strong> {citation.get('page_number', 'N/A')} |
        <strong>Score:</strong> {citation.get('score', 0):.2f}
    </div>
    """, unsafe_allow_html=True)

    # Show caption
    if citation.get('caption'):
        st.caption(f"💭 {citation['caption']}")

    # Show OCR text
    if citation.get('ocr_text'):
        with st.expander("📝 Extracted Text"):
            st.text(citation['ocr_text'][:500] + "..." if len(citation['ocr_text']) > 500 else citation['ocr_text'])

    # Show image if base64 available
    if citation.get('base64'):
        try:
            image_data = base64.b64decode(citation['base64'])
            st.image(image_data, use_column_width=True)
        except Exception as e:
            st.warning(f"Could not display image: {e}")


def render_table_citation(citation: Dict[str, Any], idx: int):
    """Render a table citation with preview."""
    st.markdown(f"""
    <div class="citation-card">
        <strong>📊 Table Source {idx}</strong><br/>
        <strong>File:</strong> {Path(citation.get('source', '')).name} |
        <strong>Page:</strong> {citation.get('page_number', 'N/A')} |
        <strong>Score:</strong> {citation.get('score', 0):.2f}
    </div>
    """, unsafe_allow_html=True)

    # Show description
    if citation.get('description'):
        st.caption(f"📋 {citation['description']}")

    # Show table as DataFrame
    try:
        if citation.get('markdown'):
            st.markdown("#### Table Preview")
            st.markdown(citation['markdown'])

        # Show in expander with CSV
        if citation.get('csv'):
            with st.expander("📄 View as CSV"):
                st.code(citation['csv'], language='csv')
    except Exception as e:
        st.warning(f"Could not display table: {e}")


def render_text_citation(citation: Dict[str, Any], idx: int):
    """Render a text citation."""
    st.markdown(f"""
    <div class="citation-card">
        <strong>📄 Text Source {idx}</strong><br/>
        <strong>File:</strong> {Path(citation.get('source', '')).name} |
        <strong>Page:</strong> {citation.get('page_number', 'N/A')} |
        <strong>Score:</strong> {citation.get('score', 0):.2f}
    </div>
    """, unsafe_allow_html=True)

    # Show content
    content = citation.get('content', '')
    if len(content) > 300:
        content = content[:300] + "..."
    st.info(content)


def display_response(response: Dict[str, Any]):
    """Display multi-modal response."""
    # Answer
    st.markdown("### 💡 Answer")
    st.write(response.get('answer', 'No answer generated.'))

    # Sources used
    sources_used = response.get('sources_used', [])
    if sources_used:
        st.markdown(f"**Sources:** {', '.join(sources_used).title()}")

    # Citations
    text_citations = response.get('text_citations', [])
    image_citations = response.get('image_citations', [])
    table_citations = response.get('table_citations', [])

    if not any([text_citations, image_citations, table_citations]):
        st.info("No citations found.")
        return

    # Tabs for different citation types
    tabs = []
    if text_citations:
        tabs.append(f"📄 Text ({len(text_citations)})")
    if image_citations:
        tabs.append(f"🖼️  Images ({len(image_citations)})")
    if table_citations:
        tabs.append(f"📊 Tables ({len(table_citations)})")

    if tabs:
        tab1, tab2, tab3 = st.tabs(tabs)

        # Text citations
        if text_citations:
            with tab1:
                for idx, citation in enumerate(text_citations, 1):
                    render_text_citation(citation, idx)

        # Image citations
        if image_citations:
            with (tab2 if text_citations else tab1):
                for idx, citation in enumerate(image_citations, 1):
                    render_image_citation(citation, idx)

        # Table citations
        if table_citations:
            with (tab3 if text_citations or image_citations else tab1):
                for idx, citation in enumerate(table_citations, 1):
                    render_table_citation(citation, idx)


# ==================== Main Layout ====================

def main():
    """Main application."""

    # Header
    st.markdown('<h1 class="main-header">🔍 Multi-Modal RAG System</h1>', unsafe_allow_html=True)
    st.markdown("*Search across text, images, and tables in your documents*")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API URL
        api_url = st.text_input("API URL", value=st.session_state.api_url, key="api_url_input")
        st.session_state.api_url = api_url

        # Initialize client
        client = MultiModalRAGClient(api_url)

        # Content type filters
        st.subheader("Content Filters")
        include_text = st.checkbox("📄 Text", value=True, key="filter_text")
        include_images = st.checkbox("🖼️  Images", value=True, key="filter_images")
        include_tables = st.checkbox("📊 Tables", value=True, key="filter_tables")

        # Query settings
        st.subheader("Query Settings")
        top_k = st.slider("Results", min_value=5, max_value=50, value=10, key="top_k")
        min_score = st.slider("Min Score", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key="min_score")

        st.divider()

        # Stats
        try:
            stats = client.get_stats()
            st.markdown("### 📊 Index Stats")
            st.metric("Total Nodes", stats.get('total_nodes', 0))
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Text", stats.get('text_nodes', 0))
                st.metric("Images", stats.get('image_nodes', 0))
            with col2:
                st.metric("Tables", stats.get('table_nodes', 0))
        except Exception as e:
            st.warning(f"Could not fetch stats: {e}")

        st.divider()

        # Quick links
        st.markdown("### 🔗 Quick Links")
        if st.button("📋 Browse Images"):
            st.session_state.page = "browse_images"
        if st.button("📊 Browse Tables"):
            st.session_state.page = "browse_tables"

    # Main content area
    page = st.session_state.get('page', 'query')

    if page == 'query':
        query_page(client, include_text, include_images, include_tables, top_k, min_score)
    elif page == 'browse_images':
        browse_images_page(client)
    elif page == 'browse_tables':
        browse_tables_page(client)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Multi-Modal RAG System | Built with Streamlit, FastAPI, and CLIP
    </div>
    """, unsafe_allow_html=True)


def query_page(client, include_text, include_images, include_tables, top_k, min_score):
    """Query page."""

    # Query tabs
    query_tab, upload_tab, history_tab = st.tabs(["💬 Query", "📤 Upload", "📜 History"])

    with query_tab:
        st.markdown('<h2 class="sub-header">Ask a Question</h2>', unsafe_allow_html=True)

        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="e.g., What does the revenue chart show about Q3 performance?",
            height=100,
            key="question_input"
        )

        # Image query option
        with st.expander("🖼️  Query with Image"):
            query_image = st.file_uploader(
                "Upload an image to find similar content",
                type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
                key="query_image_upload"
            )

        # Query button
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🔍 Search", type="primary", use_container_width=True):
                if not question and not query_image:
                    st.warning("Please enter a question or upload an image.")
                    st.stop()

                with st.spinner("Searching..."):
                    try:
                        # Build content types
                        content_types = []
                        if include_text:
                            content_types.append("text")
                        if include_images:
                            content_types.append("image")
                        if include_tables:
                            content_types.append("table")

                        if query_image:
                            # Image query
                            response = client.query_with_image(
                                question=question or "Describe similar content",
                                image_bytes=query_image.read(),
                                include_text=include_text,
                                include_tables=include_tables,
                                top_k=top_k,
                            )
                        else:
                            # Text query
                            response = client.query(
                                question=question,
                                content_types=content_types,
                                include_images=include_images,
                                include_tables=include_tables,
                                top_k=top_k,
                            )

                        # Store response
                        st.session_state.current_response = response
                        st.session_state.query_history.insert(0, {
                            'question': question or "[Image Query]",
                            'response': response,
                            'timestamp': pd.Timestamp.now().isoformat()
                        })

                        # Display response
                        st.success("✅ Search complete!")
                        display_response(response)

                    except requests.exceptions.ConnectionError:
                        st.error("❌ Could not connect to API. Make sure the server is running.")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        with col2:
            st.write("")

        # Display current response if available
        if st.session_state.get('current_response'):
            st.divider()
            display_response(st.session_state.current_response)

    with upload_tab:
        st.markdown('<h2 class="sub-header">Upload Document</h2>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a document (PDF, DOCX)",
            type=['pdf', 'docx', 'doc'],
            key="doc_upload"
        )

        if uploaded_file:
            # File info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Filename", uploaded_file.name)
            with col2:
                st.metric("Size", f"{len(uploaded_file.getvalue()) / 1024:.1f} KB")
            with col3:
                st.metric("Type", uploaded_file.type)

            st.divider()

            # Extraction options
            col1, col2 = st.columns(2)
            with col1:
                extract_images = st.checkbox("Extract Images", value=True)
                generate_captions = st.checkbox("Generate Captions", value=True)
            with col2:
                extract_tables = st.checkbox("Extract Tables", value=True)
                extract_ocr = st.checkbox("Extract OCR Text", value=True)

            # Upload button
            if st.button("📤 Process Document", type="primary", use_container_width=True):
                with st.spinner("Processing document..."):
                    try:
                        # Save to temp file
                        temp_path = f"/tmp/{uploaded_file.name}"
                        with open(temp_path, 'wb') as f:
                            f.write(uploaded_file.getvalue())

                        # Ingest
                        result = client.ingest_document(
                            file_path=temp_path,
                            extract_images=extract_images,
                            extract_tables=extract_tables,
                            generate_captions=generate_captions,
                            extract_ocr=extract_ocr,
                        )

                        # Display results
                        st.success("✅ Document processed successfully!")
                        st.json(result)

                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    with history_tab:
        st.markdown('<h2 class="sub-header">Query History</h2>', unsafe_allow_html=True)

        if not st.session_state.query_history:
            st.info("No queries yet.")
        else:
            for i, item in enumerate(st.session_state.query_history[:10], 1):
                with st.expander(f"Q{i}: {item['question'][:80]}..."):
                    st.write(f"**Time:** {item['timestamp']}")
                    display_response(item['response'])


def browse_images_page(client):
    """Browse images page."""
    st.markdown('<h2 class="sub-header">🖼️ Browse Images</h2>', unsafe_allow_html=True)

    try:
        data = client.list_images(limit=50)

        # Summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Images", data['total'])
        with col2:
            st.metric("Showing", f"{len(data['images'])} images")

        # Display images in grid
        if data['images']:
            # Filter controls
            search = st.text_input("Filter by source...", key="image_search")

            filtered = [
                img for img in data['images']
                if search.lower() in img['source_doc'].lower()
            ]

            st.write(f"Showing {len(filtered)} images")

            # Grid layout
            cols = st.columns(3)
            for i, img in enumerate(filtered):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="citation-card">
                        <strong>{Path(img['source_doc']).name}</strong><br/>
                        Page {img['page_number']} | Format: {img['format']}
                    </div>
                    """, unsafe_allow_html=True)

                    if img.get('caption'):
                        st.caption(f"💭 {img['caption'][:100]}...")

                    if st.button(f"View", key=f"view_img_{i}"):
                        try:
                            full_data = client.get_image(img['id'])
                            st.json(full_data)
                        except:
                            st.warning("Could not fetch full image data")
        else:
            st.info("No images found.")

    except Exception as e:
        st.error(f"Error: {e}")


def browse_tables_page(client):
    """Browse tables page."""
    st.markdown('<h2 class="sub-header">📊 Browse Tables</h2>', unsafe_allow_html=True)

    try:
        data = client.list_tables(limit=50)

        # Summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Tables", data['total'])
        with col2:
            st.metric("Showing", f"{len(data['tables'])} tables")

        # Display tables
        if data['tables']:
            # Filter controls
            search = st.text_input("Filter by source...", key="table_search")

            filtered = [
                tbl for tbl in data['tables']
                if search.lower() in tbl['source_doc'].lower()
            ]

            for i, tbl in enumerate(filtered):
                with st.expander(
                    f"📊 {Path(tbl['source_doc']).name} "
                    f"(Page {tbl['page_number']}) | "
                    f"{tbl['row_count']}×{tbl['col_count']}"
                ):
                    st.caption(f"Description: {tbl.get('description', 'N/A')}")

                    if st.button(f"View Full Table", key=f"view_tbl_{i}"):
                        try:
                            full_data = client.get_table(tbl['id'])
                            st.markdown("#### Table Data")
                            st.markdown(full_data['markdown'])

                            with st.expander("JSON"):
                                st.code(full_data['json'], language='json')
                        except:
                            st.warning("Could not fetch full table data")
        else:
            st.info("No tables found.")

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
