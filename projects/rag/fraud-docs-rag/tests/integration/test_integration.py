"""
Integration Tests for FraudDocs-RAG.

This test suite covers end-to-end testing of the RAG system including:
- Document ingestion pipeline
- Query processing and retrieval
- API endpoints
- Multi-component workflows

Run with:
    pytest tests/integration/test_integration.py -v
    pytest tests/integration/test_integration.py::test_e2e_query -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fraud_docs_rag.config import settings
from fraud_docs_rag.generation.rag_chain import RAGChain
from fraud_docs_rag.ingestion.document_processor import DocumentProcessor, DocumentCategory
from fraud_docs_rag.retrieval.hybrid_retriever import HybridRetriever


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_documents_dir(tmp_path):
    """Create a temporary directory with sample documents."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    # AML document
    aml_doc = docs_dir / "aml_policy.txt"
    aml_doc.write_text(
        "Anti-Money Laundering Policy\n\n"
        "Suspicious Activity Reports (SAR) must be filed within 30 days "
        "of detecting suspicious transaction activity. Financial institutions "
        "are required to report any transactions that appear suspicious, "
        "involve illegal funds, or have no business purpose. "
        "The Bank Secrecy Act (BSA) requires all financial institutions to "
        "maintain proper records and report suspicious activities to FinCEN."
    )

    # KYC document
    kyc_doc = docs_dir / "kyc_procedures.txt"
    kyc_doc.write_text(
        "Know Your Customer (KYC) Procedures\n\n"
        "Customer Due Diligence (CDD) requires verification of customer "
        "identity using reliable, independent source documents. "
        "Enhanced Due Diligence (EDD) must be applied for high-risk customers, "
        "including politically exposed persons (PEPs). "
        "Beneficial ownership information must be collected for all entity accounts."
    )

    # Fraud document
    fraud_doc = docs_dir / "fraud_detection.txt"
    fraud_doc.write_text(
        "Fraud Detection Guide\n\n"
        "Fraud detection systems must monitor for unusual transaction patterns, "
        "including rapid movement of funds, structuring to avoid reporting "
        "thresholds (smurfing), and transactions with high-risk jurisdictions. "
        "All fraud alerts must be investigated by trained personnel within "
        "24 hours of generation. Common fraud indicators include "
        "unauthorized transactions and account takeover attempts."
    )

    return docs_dir


@pytest.fixture
def test_chroma_path(tmp_path):
    """Create a temporary path for ChromaDB."""
    chroma_path = tmp_path / "chroma_db"
    chroma_path.mkdir()
    return str(chroma_path)


@pytest.fixture
def sample_nodes(sample_documents_dir):
    """Create pre-built sample nodes with embeddings."""
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu",
    )

    nodes = [
        TextNode(
            text=(
                "Suspicious Activity Reports (SAR) must be filed within 30 days "
                "of detecting suspicious transaction activity. Financial institutions "
                "are required to report any transactions that appear suspicious."
            ),
            metadata={
                "file_name": "aml_sar.pdf",
                "category": "aml",
                "title": "SAR Requirements",
            }
        ),
        TextNode(
            text=(
                "Customer Due Diligence (CDD) requires verification of customer "
                "identity using reliable, independent source documents. "
                "Enhanced Due Diligence (EDD) must be applied for high-risk customers."
            ),
            metadata={
                "file_name": "kyc_cdd.pdf",
                "category": "kyc",
                "title": "CDD Procedures",
            }
        ),
    ]

    # Add embeddings
    for node in nodes:
        node.embedding = embed_model.get_text_embedding(node.text)

    return nodes


# ============================================================================
# Document Processor Tests
# ============================================================================


@pytest.mark.integration
class TestDocumentProcessorIntegration:
    """Integration tests for document processing."""

    def test_process_directory(self, sample_documents_dir, test_chroma_path):
        """Test processing a directory of documents."""
        processor = DocumentProcessor()

        nodes = processor.process_directory(sample_documents_dir)

        assert nodes is not None
        assert len(nodes) > 0
        assert all(node.metadata.get("category") for node in nodes)

    def test_document_classification(self, sample_documents_dir):
        """Test document classification."""
        processor = DocumentProcessor()

        # Test AML document
        nodes = processor.process_document(sample_documents_dir / "aml_policy.txt")
        assert nodes is not None
        assert nodes[0].metadata.get("category") == "aml"

        # Test KYC document
        nodes = processor.process_document(sample_documents_dir / "kyc_procedures.txt")
        assert nodes is not None
        assert nodes[0].metadata.get("category") == "kyc"

    def test_content_deduplication(self, sample_documents_dir, test_chroma_path):
        """Test that duplicate documents are detected."""
        processor = DocumentProcessor()

        # Process same document twice
        nodes1 = processor.process_document(sample_documents_dir / "aml_policy.txt")
        nodes2 = processor.process_document(sample_documents_dir / "aml_policy.txt")

        assert len(nodes1) > 0
        # Second call should return None or empty due to deduplication
        assert nodes2 is None or len(nodes2) == 0

    def test_metadata_extraction(self, sample_documents_dir):
        """Test metadata extraction from documents."""
        processor = DocumentProcessor()

        nodes = processor.process_document(sample_documents_dir / "aml_policy.txt")

        assert nodes is not None
        metadata = nodes[0].metadata

        # Check required metadata fields
        assert "file_name" in metadata
        assert "category" in metadata
        assert "ingestion_date" in metadata
        assert "content_hash" in metadata
        assert "word_count" in metadata

        assert metadata["file_name"] == "aml_policy.txt"
        assert metadata["category"] == "aml"


# ============================================================================
# Retriever Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.llm
class TestHybridRetrieverIntegration:
    """Integration tests for the hybrid retriever."""

    def test_build_and_load_index(self, sample_nodes, test_chroma_path):
        """Test building and loading a vector index."""
        retriever = HybridRetriever(
            chroma_path=test_chroma_path,
            collection_name="test_collection",
        )

        # Build index
        retriever.build_index(sample_nodes)

        # Load index
        success = retriever.load_index()
        assert success is True
        assert retriever.index is not None

    def test_retrieve_without_filter(self, sample_nodes, test_chroma_path):
        """Test retrieval without document type filtering."""
        retriever = HybridRetriever(
            chroma_path=test_chroma_path,
            collection_name="test_retrieve",
            top_k=2,
            rerank_top_n=2,  # Must be <= top_k
        )

        retriever.build_index(sample_nodes)
        retriever.load_index()

        nodes = retriever.retrieve("customer verification requirements")

        assert nodes is not None
        assert len(nodes) > 0
        assert nodes[0].score is not None

    def test_retrieve_with_filter(self, sample_nodes, test_chroma_path):
        """Test retrieval with document type filtering."""
        retriever = HybridRetriever(
            chroma_path=test_chroma_path,
            collection_name="test_filter",
            top_k=5,
        )

        retriever.build_index(sample_nodes)
        retriever.load_index()

        # Filter by KYC
        nodes = retriever.retrieve(
            "customer requirements",
            doc_type_filter="kyc"
        )

        assert nodes is not None
        # Should only return KYC documents
        for node in nodes:
            assert node.metadata.get("category") == "kyc"

    def test_reranking(self, sample_nodes, test_chroma_path):
        """Test cross-encoder reranking."""
        retriever = HybridRetriever(
            chroma_path=test_chroma_path,
            collection_name="test_rerank",
            top_k=5,
            rerank_top_n=2,
        )

        retriever.build_index(sample_nodes)
        retriever.load_index()

        # Retrieve with reranking
        nodes_with_rerank = retriever.retrieve(
            "suspicious activity",
            use_rerank=True
        )

        # Retrieve without reranking
        nodes_without_rerank = retriever.retrieve(
            "suspicious activity",
            use_rerank=False
        )

        assert nodes_with_rerank is not None
        assert nodes_without_rerank is not None
        # Reranked should have fewer results
        assert len(nodes_with_rerank) <= len(nodes_without_rerank)

    def test_format_context(self, sample_nodes, test_chroma_path):
        """Test context formatting with citations."""
        from llama_index.core.schema import NodeWithScore

        retriever = HybridRetriever(
            chroma_path=test_chroma_path,
            collection_name="test_format",
        )

        retriever.build_index(sample_nodes)
        retriever.load_index()

        nodes = retriever.retrieve("customer due diligence")

        context = retriever.format_context(nodes, include_scores=True)

        assert context is not None
        # Check for numbered citation format like [1. filename | Category:
        assert "[" in context and "." in context
        assert "Category:" in context


# ============================================================================
# RAG Chain Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.llm
class TestRAGChainIntegration:
    """Integration tests for the RAG chain."""

    @pytest.fixture
    def rag_chain(self, sample_nodes, test_chroma_path):
        """Create a RAG chain instance for testing."""
        retriever = HybridRetriever(
            chroma_path=test_chroma_path,
            collection_name="test_rag",
            top_k=5,
            rerank_top_n=3,
        )

        retriever.build_index(sample_nodes)

        return RAGChain(
            retriever=retriever,
            environment="development",  # Uses Ollama
        )

    def test_query_generation(self, rag_chain):
        """Test end-to-end query processing."""
        answer, sources = rag_chain.query(
            "What are the customer verification requirements?"
        )

        assert answer is not None
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(sources, list)

        if sources:
            assert "file_name" in sources[0]
            assert "category" in sources[0]
            assert "score" in sources[0]

    def test_query_with_filter(self, rag_chain):
        """Test querying with document type filter."""
        answer, sources = rag_chain.query(
            "verification requirements",
            doc_type_filter="kyc"
        )

        assert answer is not None
        # Check that sources match the filter
        if sources:
            assert all(s["category"] == "kyc" for s in sources)

    def test_format_context_with_numbers(self, rag_chain):
        """Test context formatting with numbered citations."""
        context = "Test [Source: file.pdf | Category: aml]\nSome content here"

        formatted, sources = rag_chain._format_context_with_numbers(context)

        assert formatted is not None
        assert isinstance(sources, list)

    @pytest.mark.skipif(
        os.getenv("OLLAMA_BASE_URL") is None,
        reason="Ollama not available"
    )
    def test_query_stream(self, rag_chain):
        """Test streaming query response."""
        chunks = list(rag_chain.query_stream("What is SAR?"))

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.llm
class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_full_ingion_and_query_workflow(
        self,
        sample_documents_dir,
        test_chroma_path
    ):
        """Test complete workflow: ingest documents and query."""
        # Step 1: Ingest documents
        processor = DocumentProcessor()
        nodes = processor.process_directory(sample_documents_dir)

        assert nodes is not None
        assert len(nodes) > 0

        # Step 2: Build index
        retriever = HybridRetriever(
            chroma_path=test_chroma_path,
            collection_name="test_e2e",
        )
        retriever.build_index(nodes)

        # Step 3: Query
        rag_chain = RAGChain(
            retriever=retriever,
            environment="development",
        )

        answer, sources = rag_chain.query("What are SAR requirements?")

        assert answer is not None
        assert len(answer) > 0

    def test_multi_category_filtering(self, sample_nodes, test_chroma_path):
        """Test filtering by multiple document categories."""
        retriever = HybridRetriever(
            chroma_path=test_chroma_path,
            collection_name="test_multi",
        )

        retriever.build_index(sample_nodes)

        # Query with multiple category filters
        nodes = retriever.retrieve(
            "requirements",
            doc_type_filter=["aml", "kyc"]
        )

        assert nodes is not None
        # Verify results are from specified categories
        for node in nodes:
            assert node.metadata.get("category") in ["aml", "kyc"]

    def test_empty_query_handling(self, sample_nodes, test_chroma_path):
        """Test handling of queries with no results."""
        retriever = HybridRetriever(
            chroma_path=test_chroma_path,
            collection_name="test_empty",
        )

        retriever.build_index(sample_nodes)

        rag_chain = RAGChain(retriever=retriever, environment="development")

        # Query about something not in documents
        answer, sources = rag_chain.query("quantum physics applications")

        # Should return a helpful message
        assert answer is not None
        assert len(sources) == 0


# ============================================================================
# API Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.llm
class TestAPIIntegration:
    """Integration tests for the FastAPI endpoints."""

    @pytest.fixture
    def api_url(self):
        """Get the API URL for testing."""
        return os.getenv("API_TEST_URL", "http://localhost:8000")

    def test_health_endpoint(self, api_url):
        """Test the health check endpoint."""
        response = requests.get(f"{api_url}/health", timeout=10)

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_collections_endpoint(self, api_url):
        """Test the collections list endpoint."""
        response = requests.get(f"{api_url}/collections", timeout=10)

        assert response.status_code == 200
        data = response.json()
        assert "collections" in data
        assert "total" in data

    def test_query_endpoint(self, api_url):
        """Test the query endpoint."""
        payload = {
            "question": "What are SAR filing requirements?",
            "doc_type_filter": "aml",
            "use_rerank": True,
        }

        response = requests.post(
            f"{api_url}/query",
            json=payload,
            timeout=60,
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "query" in data

    def test_query_endpoint_validation(self, api_url):
        """Test query endpoint validation."""
        # Empty question
        payload = {
            "question": "",
        }

        response = requests.post(
            f"{api_url}/query",
            json=payload,
            timeout=10,
        )

        assert response.status_code == 422

    def test_ingest_endpoint(self, api_url, sample_documents_dir):
        """Test the document ingestion endpoint."""
        # Test with a sample file
        files = {
            "file": open(sample_documents_dir / "aml_policy.txt", "rb"),
        }
        data = {
            "doc_type": "aml",
        }

        response = requests.post(
            f"{api_url}/ingest",
            files=files,
            data=data,
            timeout=120,
        )

        assert response.status_code == 200
        result = response.json()
        assert "status" in result
        assert "documents_processed" in result


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.llm
class TestPerformance:
    """Performance and load tests."""

    def test_query_latency(self, sample_nodes, test_chroma_path):
        """Test query processing latency."""
        import time

        retriever = HybridRetriever(
            chroma_path=test_chroma_path,
            collection_name="test_perf",
        )

        retriever.build_index(sample_nodes)

        rag_chain = RAGChain(retriever=retriever, environment="development")

        start_time = time.time()
        answer, sources = rag_chain.query("What is SAR?")
        latency = time.time() - start_time

        # Query should complete in reasonable time
        assert latency < 30  # 30 seconds max

    def test_batch_queries(self, sample_nodes, test_chroma_path):
        """Test processing multiple queries."""
        retriever = HybridRetriever(
            chroma_path=test_chroma_path,
            collection_name="test_batch",
        )

        retriever.build_index(sample_nodes)

        rag_chain = RAGChain(retriever=retriever, environment="development")

        queries = [
            "What is SAR?",
            "Explain KYC",
            "Fraud detection methods",
        ]

        results = []
        for query in queries:
            answer, sources = rag_chain.query(query)
            results.append((answer, sources))

        assert len(results) == len(queries)
        assert all(answer for answer, _ in results)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
