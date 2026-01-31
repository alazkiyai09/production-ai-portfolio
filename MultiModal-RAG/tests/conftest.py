# ============================================================
# Enterprise-RAG: Test Fixtures
# ============================================================
"""
Pytest configuration and shared fixtures for Enterprise-RAG tests.

This module provides:
- Sample document fixtures
- Mock services (embedding, LLM, vector store)
- Test utilities
- Database fixtures
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ============================================================
# Sample Documents
# ============================================================

SAMPLE_TEXT_MD = """
# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data.

## Key Concepts

### Supervised Learning
In supervised learning, models are trained on labeled data. Common algorithms include linear regression, decision trees, and neural networks.

### Unsupervised Learning
Unsupervised learning deals with unlabeled data. Techniques include clustering and dimensionality reduction.

## Applications

Machine learning is used in various fields:
- Image recognition
- Natural language processing
- Recommendation systems
- Autonomous vehicles
"""

SAMPLE_TEXT_PDF = """
Company Privacy Policy
=====================

Effective Date: January 1, 2024

1. Data Collection
We collect information you provide directly, including:
- Name and contact information
- Account credentials
- Payment information

2. Data Usage
We use your data to:
- Provide our services
- Process transactions
- Send important updates
- Improve our offerings

3. Data Sharing
We do not sell your personal information. We may share data with:
- Service providers who assist our operations
- Legal authorities when required
- Affiliates with your consent

4. Data Security
We implement industry-standard security measures including:
- Encryption (AES-256)
- Secure sockets layer (SSL)
- Regular security audits

5. Your Rights
You have the right to:
- Access your data
- Correct inaccurate information
- Delete your account
- Opt-out of marketing communications

Contact us at privacy@company.com with questions.
"""

SAMPLE_TEXT_DOCX = """
Quarterly Report Q1 2024
======================

Financial Highlights
-------------------

Revenue Growth
Our revenue increased by 25% year-over-year to reach $10.5 million in Q1 2024.

Key Metrics
- Revenue: $10.5M (+25% YoY)
- Gross Margin: 68%
- Operating Income: $2.1M
- Active Users: 500,000

Business Segments
-----------------

Enterprise: Contributed $6.3M in revenue, representing 60% of total revenue.

Consumer: Generated $4.2M in revenue, with strong growth in mobile applications.

Outlook
-------
Based on current trends, we expect full-year revenue to reach $45-48M.
"""

SAMPLE_TEXT_TXT = """
Refund Policy
=============

Standard Refunds
-----------------
Digital Products
- Full refund within 7 days of purchase
- Partial refund (50%) within 30 days

Physical Products
- Full refund within 14 days
- Customer pays return shipping

Subscription Services
-------------------------
Monthly Plans: Cancel anytime, prorated refund for unused days.

Annual Plans: Cancel at end of billing cycle, or receive 50% refund for unused months.

Process
------
To request a refund:
1. Log into your account
2. Go to Settings > Billing
3. Click "Request Refund"
4. Select the item and reason
5. Submit the request

Refunds are typically processed within 5-7 business days.

Exceptions
----------
- No refunds on gift cards
- Custom orders are non-refundable
- Sale items may be exchanged only
"""


# ============================================================
# Pytest Configuration
# ============================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_openai: marks tests that require OPENAI_API_KEY"
    )


# ============================================================
# Document Fixtures
# ============================================================

@pytest.fixture
def sample_md_content():
    """Sample markdown content for testing."""
    return SAMPLE_TEXT_MD


@pytest.fixture
def sample_pdf_content():
    """Sample PDF-like text content for testing."""
    return SAMPLE_TEXT_PDF


@pytest.fixture
def sample_docx_content():
    """Sample DOCX-like text content for testing."""
    return SAMPLE_TEXT_DOCX


@pytest.fixture
def sample_txt_content():
    """Sample plain text content for testing."""
    return SAMPLE_TEXT_TXT


@pytest.fixture
def temp_pdf_file(sample_pdf_content):
    """Create a temporary PDF-like file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
        f.write(sample_pdf_content)
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_docx_file(sample_docx_content):
    """Create a temporary DOCX-like file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".docx", delete=False) as f:
        f.write(sample_docx_content)
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_md_file(sample_md_content):
    """Create a temporary markdown file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(sample_md_content)
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_txt_file(sample_txt_content):
    """Create a temporary text file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(sample_txt_content)
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_documents_dir():
    """Create a temporary directory with sample documents."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create sample files
    (temp_dir / "doc1.md").write_text(SAMPLE_TEXT_MD)
    (temp_dir / "doc2.txt").write_text(SAMPLE_TEXT_PDF)
    (temp_dir / "doc3.txt").write_text(SAMPLE_TEXT_DOCX)
    (temp_dir / "doc4.txt").write_text(SAMPLE_TEXT_TXT)

    yield temp_dir

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# Mock Services
# ============================================================

@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = MagicMock()

    # Mock embed_single to return random embeddings
    def mock_embed_single(text: str) -> np.ndarray:
        # Generate consistent embeddings based on text hash
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(384).astype(np.float32)

    def mock_embed_texts(texts: List[str]) -> np.ndarray:
        return np.array([mock_embed_single(t) for t in texts])

    service.embed_single = Mock(side_effect=mock_embed_single)
    service.embed_texts = Mock(side_effect=mock_embed_texts)
    service.embedding_dimension = 384
    service.device = "cpu"

    return service


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()

    # Mock chat completions
    class MockChoice:
        def __init__(self, message):
            self.message = message

    class MockMessage:
        content = "This is a test answer based on the provided context."

    class MockUsage:
        prompt_tokens = 100
        completion_tokens = 50
        total_tokens = 150

    class MockResponse:
        choices = [MockChoice]
        usage = MockUsage()

    client.chat.completions.create = Mock(return_value=MockResponse())

    return client


@pytest.fixture
def mock_llm_client_multi():
    """Create a mock LLM client that returns different answers."""
    client = MagicMock()

    call_count = [0]

    def mock_create(*args, **kwargs):
        call_count[0] += 1
        answers = [
            "First answer about refunds.",
            "Second answer about privacy.",
            "Third answer about pricing.",
        ]

        class MockChoice:
            pass

        class MockMessage:
            content = answers[call_count[0] % len(answers)]

        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

        class MockResponse:
            choices = [MockChoice]
            message = MockMessage()
            usage = MockUsage()

        return MockResponse()

    client.chat.completions.create = Mock(side_effect=mock_create)
    return client


# ============================================================
# Test Data
# ============================================================

@pytest.fixture
def sample_documents():
    """Create sample Document objects for testing."""
    from src.ingestion.document_processor import Document

    return [
        Document(
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "test1.md", "page": 1, "file_type": "md"},
            doc_id="doc_001",
            chunk_id="doc_001_chunk_0",
        ),
        Document(
            content="Refunds are processed within 5-7 business days.",
            metadata={"source": "policy.pdf", "page": 2, "file_type": "pdf"},
            doc_id="doc_002",
            chunk_id="doc_002_chunk_0",
        ),
        Document(
            content="We use AES-256 encryption for data security.",
            metadata={"source": "privacy.txt", "file_type": "txt"},
            doc_id="doc_003",
            chunk_id="doc_003_chunk_0",
        ),
    ]


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return np.random.rand(3, 384).astype(np.float32)


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What is the refund policy?"


@pytest.fixture
def sample_questions():
    """Sample questions for testing."""
    return [
        "What is the refund policy?",
        "How does the company handle data privacy?",
        "What are the pricing tiers?",
        "How do I cancel my subscription?",
    ]


# ============================================================
# Vector Store Fixtures
# ============================================================

@pytest.fixture
def test_vector_store(sample_documents, sample_embeddings):
    """Create an in-memory test vector store."""
    from src.retrieval.vector_store import VectorStoreBase, SearchResult

    class TestVectorStore(VectorStoreBase):
        """Simple in-memory vector store for testing."""

        def __init__(self):
            self.documents = {}
            self.embeddings = {}

        def add_documents(self, documents, embeddings):
            for doc, emb in zip(documents, embeddings):
                self.documents[doc.chunk_id] = doc
                self.embeddings[doc.chunk_id] = emb
            return len(documents)

        def search(self, query_embedding, top_k=10, filters=None):
            # Simple cosine similarity search
            similarities = []
            for chunk_id, emb in self.embeddings.items():
                similarity = np.dot(query_embedding, emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                )
                similarities.append((similarity, chunk_id))

            # Sort and filter
            similarities.sort(reverse=True)
            results = []
            for sim, chunk_id in similarities[:top_k]:
                doc = self.documents[chunk_id]
                result = SearchResult(
                    doc_id=doc.doc_id,
                    chunk_id=chunk_id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=sim,
                )
                results.append(result)

            return results

        def delete(self, doc_ids):
            count = 0
            for doc_id in doc_ids:
                to_delete = [
                    chunk_id for chunk_id, doc in self.documents.items()
                    if doc.doc_id == doc_id
                ]
                for chunk_id in to_delete:
                    del self.documents[chunk_id]
                    del self.embeddings[chunk_id]
                    count += 1
            return count

        def get_stats(self):
            from src.retrieval.vector_store import VectorStoreStats
            return VectorStoreStats(
                total_documents=len(set(doc.doc_id for doc in self.documents.values())),
                total_chunks=len(self.documents),
            )

        def clear(self):
            self.documents.clear()
            self.embeddings.clear()

    store = TestVectorStore()
    store.add_documents(sample_documents, sample_embeddings)
    return store


# ============================================================
# BM25 Retriever Fixtures
# ============================================================

@pytest.fixture
def test_bm25_retriever(sample_documents):
    """Create a test BM25 retriever."""
    from src.retrieval.sparse_retriever import BM25Retriever

    retriever = BM25Retriever(
        index_path=None,  # In-memory
        k1=1.5,
        b=0.75,
        normalize_scores=True,
    )

    # Build index with sample documents
    retriever.build_index(sample_documents)

    return retriever


# ============================================================
# Hybrid Retriever Fixtures
# ============================================================

@pytest.fixture
def test_hybrid_retriever(test_vector_store, mock_embedding_service, test_bm25_retriever):
    """Create a test hybrid retriever."""
    from src.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(
        vector_store=test_vector_store,
        embedding_service=mock_embedding_service,
        bm25_retriever=test_bm25_retriever,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
    )

    return retriever


# ============================================================
# Reranker Fixtures
# ============================================================

@pytest.fixture
def mock_reranker():
    """Create a mock reranker."""
    from unittest.mock import MagicMock
    from src.retrieval.reranker import RerankedSearchResult, HybridSearchResult
    from src.retrieval.hybrid_retriever import HybridSearchResult

    reranker = MagicMock()

    # Mock rerank method
    def mock_rerank_func(query, results, top_k):
        # Return results with modified scores
        reranked = []
        for i, result in enumerate(results[:top_k]):
            # Create reranked result with new score
            reranked.append(
                RerankedSearchResult(
                    base_result=result,
                    rerank_score=0.9 - (i * 0.1),  # Descending scores
                    original_rank=i + 1,
                    rank_change=i - i,  # No change
                )
            )
        return reranked

    reranker.rerank = Mock(side_effect=mock_rerank_func)
    reranker._is_loaded = True
    reranker.model_name = "test-model"

    return reranker


# ============================================================
# RAG Chain Fixtures
# ============================================================

@pytest.fixture
def mock_rag_chain(test_hybrid_retriever, mock_reranker):
    """Create a mock RAG chain."""
    from src.generation.rag_chain import RAGChain, LLMProvider
    from unittest.mock import MagicMock, patch

    chain = MagicMock(spec=RAGChain)

    # Set up retriever and reranker
    chain.retriever = test_hybrid_retriever
    chain.reranker = mock_reranker

    # Mock query method
    from src.generation.rag_chain import RAGResponse, Citation, Message

    test_answer = (
        "According to the policy [1], refunds are processed within 5-7 business days. "
        "For digital products, full refunds are available within 7 days [2]."
    )

    test_response = RAGResponse(
        answer=test_answer,
        citations=[
            Citation(
                source="policy.pdf",
                chunk_id="doc_002_chunk_0",
                content_preview="Refunds are processed within 5-7 business days...",
                relevance_score=0.92,
            )
        ],
        retrieval_results=test_hybrid_retriever.retrieve("test", top_k=5)[:5],
        model_used="test-model",
        provider_used=LLMProvider.OPENAI,
        processing_time=1.5,
        retrieval_time=0.2,
        rerank_time=0.3,
        generation_time=1.0,
        token_usage={"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
    )

    chain.query = Mock(return_value=test_response)
    chain.conversation_history = []

    return chain


# ============================================================
# API Test Fixtures
# ============================================================

@pytest.fixture
def test_client():
    """Create a test FastAPI client."""
    from fastapi.testclient import TestClient

    # Import app here to avoid import errors
    from src.api.main import app

    return TestClient(app)


@pytest.fixture
def mock_api_dependencies():
    """Mock API dependencies for testing."""
    from unittest.mock import MagicMock, patch

    # Mock all the components
    mock_components = {
        "rag_chain": MagicMock(),
        "document_processor": MagicMock(),
        "vector_store": MagicMock(),
        "rag_evaluator": MagicMock(),
    }

    return mock_components


# ============================================================
# Async Support
# ============================================================

@pytest.fixture
def event_loop_policy():
    """Set event loop policy for async tests."""
    import sys

    if sys.version_info >= (3, 8):
        import asyncio

        try:
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        except AttributeError:
            pass  # Windows doesn't have DefaultEventLoopPolicy


# ============================================================
# Environment Setup
# ============================================================

@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")  # Reduce log noise
    monkeypatch.setenv("ENABLE_EVALUATION", "false")  # Disable eval by default


# ============================================================
# Cleanup
# ============================================================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Cleanup temporary files after tests."""
    yield

    # Cleanup happens in fixtures themselves
    pass
