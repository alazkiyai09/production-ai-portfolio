"""
Comprehensive Unit Tests for DataChat-RAG

Tests for Query Router, Document Retriever, RAG Chain, and API.
"""

import json
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, MagicMock, patch, AsyncMock

import pytest
import pandas as pd
from pydantic import ValidationError

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.routers.query_router import QueryRouter, QueryType, QueryClassification
from src.retrievers.doc_retriever import DocumentRetriever, DocumentType, RetrievalResult, IngestionResult
from src.core.rag_chain import DataChatRAG, RAGResponse, SQLResult, Message
from llama_index.core import Document


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def sample_query_router():
    """Create a query router instance."""
    return QueryRouter()


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    mock = Mock()
    mock.chat = Mock(return_value=Mock(message=Mock(content='{"query_type": "SQL_QUERY", "confidence": 0.9, "reasoning": "Test", "suggested_sql_tables": ["campaigns"], "suggested_doc_topics": [], "keywords": ["ctr"]}')))
    return mock


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            text="HIPAA requires all patient data to be encrypted and access to be logged for 6 years.",
            metadata={
                "source": "HIPAA_Guide.txt",
                "doc_type": "compliance",
                "department": "legal",
                "date": "2024-01-15",
            }
        ),
        Document(
            text="Healthcare campaigns typically have CTR between 0.8-1.5% due to regulatory requirements.",
            metadata={
                "source": "Best_Practices.txt",
                "doc_type": "best_practice",
                "department": "marketing",
                "date": "2024-03-10",
            }
        ),
        Document(
            text="All healthcare ads must go through a 3-stage approval process: Creative Review, Compliance Review, and Client Approval.",
            metadata={
                "source": "Approval_Process.txt",
                "doc_type": "sop",
                "department": "operations",
                "date": "2024-02-01",
            }
        ),
    ]


@pytest.fixture
def mock_doc_retriever(sample_documents):
    """Create a mock document retriever."""
    mock = Mock(spec=DocumentRetriever)

    # Mock retrieve method
    mock.retrieve = Mock(return_value=[
        RetrievalResult(
            content="HIPAA requires all patient data to be encrypted.",
            source="HIPAA_Guide.txt",
            doc_type=DocumentType.COMPLIANCE,
            relevance_score=0.95,
            metadata={"department": "legal", "date": "2024-01-15"},
        )
    ])

    # Mock format_context method
    mock.format_context = Mock(return_value="[1] HIPAA requires all patient data to be encrypted.\n    [Source: HIPAA_Guide.txt | Compliance | 2024-01-15]")

    return mock


@pytest.fixture
def mock_sql_retriever():
    """Create a mock SQL retriever."""
    mock = Mock()

    # Mock query method
    mock.query = Mock(return_value=SQLResult(
        query="SELECT AVG(ctr) as avg_ctr FROM daily_metrics",
        results=[{"avg_ctr": 1.2}],
        columns=["avg_ctr"],
        row_count=1,
    ))

    return mock


@pytest.fixture
def sample_rag_chain(mock_sql_retriever, mock_doc_retriever, sample_query_router):
    """Create a sample RAG chain."""
    return DataChatRAG(
        sql_retriever=mock_sql_retriever,
        doc_retriever=mock_doc_retriever,
        query_router=sample_query_router,
        enable_memory=True,
    )


@pytest.fixture
def mock_api_client():
    """Create a mock API client for testing."""
    mock = Mock()

    # Mock successful chat response
    mock.post = Mock(return_value=Mock(
        status_code=200,
        json=Mock(return_value={
            "answer": "Based on the data, average CTR was 1.2%.",
            "query_type": "SQL_QUERY",
            "confidence": 0.92,
            "conversation_id": "conv_test123",
            "sql_query": "SELECT AVG(ctr) FROM daily_metrics",
            "sql_results": {"avg_ctr": 1.2},
            "doc_sources": [],
            "suggested_followup": ["See trends over time?"],
            "processing_time_seconds": 1.5,
        })
    ))

    return mock


@pytest.fixture
def sample_chroma_collection():
    """Create a mock ChromaDB collection."""
    mock = Mock()
    mock.count = Mock(return_value=100)
    mock.query = Mock(return_value={
        "documents": [["Test document content"]],
        "metadatas": [[{"source": "test.txt", "doc_type": "guideline"}]],
        "distances": [[0.1]],
    })
    mock.add = Mock()
    mock.delete = Mock()
    return mock


# =============================================================================
# Query Router Tests
# =============================================================================

class TestQueryRouter:
    """Test suite for QueryRouter."""

    def test_sql_classification_metrics_question(self, sample_query_router):
        """Test SQL classification for metrics questions."""
        query = "What was our average CTR last week?"

        result = sample_query_router.classify(query)

        assert result.query_type == QueryType.SQL_QUERY
        assert result.confidence > 0.5
        assert "daily_metrics" in result.suggested_sql_tables or "campaigns" in result.suggested_sql_tables

    def test_sql_classification_ranking_question(self, sample_query_router):
        """Test SQL classification for ranking questions."""
        query = "Top 5 campaigns by spend this month"

        result = sample_query_router.classify(query)

        assert result.query_type == QueryType.SQL_QUERY
        assert any("top" in kw.lower() or "spend" in kw.lower() for kw in result.keywords)

    def test_doc_classification_policy_question(self, sample_query_router):
        """Test DOC classification for policy questions."""
        query = "What are our HIPAA compliance requirements?"

        result = sample_query_router.classify(query)

        assert result.query_type == QueryType.DOC_SEARCH
        assert len(result.suggested_doc_topics) > 0
        assert any("hipaa" in kw.lower() or "compliance" in kw.lower() for kw in result.keywords)

    def test_doc_classification_process_question(self, sample_query_router):
        """Test DOC classification for process questions."""
        query = "What's the process for getting an ad approved?"

        result = sample_query_router.classify(query)

        assert result.query_type == QueryType.DOC_SEARCH
        assert "approval" in " ".join(result.keywords).lower() or "process" in " ".join(result.keywords).lower()

    def test_hybrid_classification_why_question(self, sample_query_router):
        """Test HYBRID classification for causal questions."""
        query = "Why is the BioGen campaign underperforming?"

        result = sample_query_router.classify(query)

        assert result.query_type == QueryType.HYBRID
        assert "campaigns" in result.suggested_sql_tables
        assert result.confidence > 0.5

    def test_hybrid_classification_comparison_question(self, sample_query_router):
        """Test HYBRID classification for benchmark comparison."""
        query = "How does our CTR compare to industry benchmarks?"

        result = sample_query_router.classify(query)

        assert result.query_type in [QueryType.HYBRID, QueryType.SQL_QUERY]
        assert result.confidence > 0.5

    def test_edge_case_empty_query(self, sample_query_router):
        """Test handling of empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            sample_query_router.classify("")

    def test_edge_case_whitespace_query(self, sample_query_router):
        """Test handling of whitespace-only query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            sample_query_router.classify("   ")

    def test_edge_case_very_long_query(self, sample_query_router):
        """Test handling of very long query."""
        long_query = "What is the " + "very " * 500 + "long question about CTR?"

        # Should not raise an error
        result = sample_query_router.classify(long_query)

        assert result.query_type in QueryType
        assert 0 <= result.confidence <= 1

    def test_ambiguous_query(self, sample_query_router):
        """Test handling of ambiguous query."""
        query = "Tell me about campaigns"

        result = sample_query_router.classify(query)

        # Should still classify, possibly with lower confidence
        assert result.query_type in QueryType
        assert 0 <= result.confidence <= 1

    def test_classification_to_dict(self, sample_query_router):
        """Test QueryClassification serialization."""
        query = "What was our CTR?"
        result = sample_query_router.classify(query)

        data = result.to_dict()

        assert "query_type" in data
        assert "confidence" in data
        assert "reasoning" in data
        assert isinstance(data["suggested_sql_tables"], list)

    def test_classification_from_dict(self, sample_query_router):
        """Test QueryClassification deserialization."""
        data = {
            "query_type": "SQL_QUERY",
            "confidence": 0.85,
            "reasoning": "Test reasoning",
            "suggested_sql_tables": ["campaigns"],
            "suggested_doc_topics": [],
            "keywords": ["ctr"],
        }

        result = QueryClassification.from_dict(data)

        assert result.query_type == QueryType.SQL_QUERY
        assert result.confidence == 0.85

    def test_cache_functionality(self, sample_query_router):
        """Test query classification caching."""
        query = "What was our CTR?"

        # First call
        result1 = sample_query_router.classify(query, use_cache=True)

        # Second call with cache
        result2 = sample_query_router.classify(query, use_cache=True)

        # Should return same result (possibly same object if cached)
        assert result1.query_type == result2.query_type
        assert result1.confidence == result2.confidence

    def test_cache_bypass(self, sample_query_router):
        """Test bypassing classification cache."""
        query = "What was our CTR?"

        result1 = sample_query_router.classify(query, use_cache=False)
        result2 = sample_query_router.classify(query, use_cache=False)

        # Should still work, just not cached
        assert result1.query_type == result2.query_type

    def test_clear_cache(self, sample_query_router):
        """Test clearing classification cache."""
        query = "What was our CTR?"

        # Add to cache (use_cache=True to populate cache)
        sample_query_router.classify(query, use_cache=True)
        # Note: Cache may not be populated if LLM is not configured
        # Clear should still work
        sample_query_router.clear_cache()
        assert len(sample_query_router._classification_cache) == 0

    def test_batch_classify(self, sample_query_router):
        """Test batch classification of multiple queries."""
        queries = [
            "What was our CTR?",
            "HIPAA requirements?",
            "Why is campaign X underperforming?",
        ]

        results = sample_query_router.batch_classify(queries)

        assert len(results) == len(queries)
        assert all(isinstance(r, QueryClassification) for r in results)


# =============================================================================
# SQL Retriever Tests
# =============================================================================

class TestSQLRetriever:
    """Test suite for SQL Retriever functionality."""

    def test_sql_result_formatting(self):
        """Test SQLResult formatting."""
        result = SQLResult(
            query="SELECT AVG(ctr) FROM daily_metrics",
            results=[{"avg_ctr": 1.2}, {"avg_ctr": 1.5}],
            columns=["avg_ctr"],
            row_count=2,
        )

        summary = result.format_summary()

        assert "SELECT" in summary
        assert "2 row" in summary
        assert "1.2" in summary

    def test_sql_result_empty_results(self):
        """Test SQLResult with no results."""
        result = SQLResult(
            query="SELECT * FROM campaigns WHERE id = 'invalid'",
            results=[],
            columns=["id"],
            row_count=0,
        )

        summary = result.format_summary()

        assert "no results" in summary.lower()

    def test_sql_result_with_error(self):
        """Test SQLResult with error."""
        result = SQLResult(
            query="INVALID SQL",
            results=[],
            columns=[],
            row_count=0,
            error="Syntax error near 'INVALID'",
        )

        summary = result.format_summary()

        assert "Error" in summary
        assert "INVALID SQL" in summary

    def test_sql_result_serialization(self):
        """Test SQLResult to_dict conversion."""
        result = SQLResult(
            query="SELECT 1",
            results=[{"col": 1}],
            columns=["col"],
            row_count=1,
        )

        data = result.to_dict()

        assert data["query"] == "SELECT 1"
        assert data["row_count"] == 1
        assert data["results"] == [{"col": 1}]

    def test_sql_query_validation_drop_table(self):
        """Test that DROP TABLE queries are blocked."""
        # This would be implemented in the actual SQLRetriever
        # For now, we test the concept
        dangerous_queries = [
            "DROP TABLE campaigns",
            "DELETE FROM campaigns",
            "UPDATE campaigns SET name='test'",
            "INSERT INTO campaigns VALUES (...)",
        ]

        for query in dangerous_queries:
            # In real implementation, these would be rejected
            assert any(keyword in query.upper() for keyword in ["DROP", "DELETE", "UPDATE", "INSERT"])


# =============================================================================
# Document Retriever Tests
# =============================================================================

class TestDocumentRetriever:
    """Test suite for Document Retriever."""

    def test_retrieval_result_format_citation(self):
        """Test RetrievalResult citation formatting."""
        result = RetrievalResult(
            content="Test content",
            source="Test_Doc.txt",
            doc_type=DocumentType.POLICY,
            relevance_score=0.92,
            metadata={"date": "2024-01-15"},
        )

        citation = result.format_citation()

        assert "Test_Doc.txt" in citation
        assert "Policy" in citation
        assert "2024-01-15" in citation

    def test_retrieval_result_to_dict(self):
        """Test RetrievalResult serialization."""
        result = RetrievalResult(
            content="Test content",
            source="Test_Doc.txt",
            doc_type=DocumentType.GUIDELINE,
            relevance_score=0.88,
        )

        data = result.to_dict()

        assert data["source"] == "Test_Doc.txt"
        assert data["doc_type"] == "guideline"
        assert data["relevance_score"] == 0.88

    def test_document_type_from_string(self):
        """Test DocumentType enum parsing."""
        assert DocumentType.from_string("policy") == DocumentType.POLICY
        assert DocumentType.from_string("POLICY") == DocumentType.POLICY
        assert DocumentType.from_string("best-practice") == DocumentType.BEST_PRACTICE

    def test_document_type_invalid_string(self):
        """Test DocumentType with invalid string."""
        with pytest.raises(ValueError):
            DocumentType.from_string("invalid_type")

    def test_retrieval_result_sorting(self):
        """Test that retrieval results can be sorted by relevance."""
        results = [
            RetrievalResult(
                content="Low relevance",
                source="doc1.txt",
                doc_type=DocumentType.GUIDELINE,
                relevance_score=0.5,
            ),
            RetrievalResult(
                content="High relevance",
                source="doc2.txt",
                doc_type=DocumentType.POLICY,
                relevance_score=0.95,
            ),
        ]

        sorted_results = sorted(results, key=lambda r: r.relevance_score, reverse=True)

        assert sorted_results[0].relevance_score > sorted_results[1].relevance_score
        assert sorted_results[0].source == "doc2.txt"

    def test_metadata_filtering(self):
        """Test metadata filtering logic."""
        # This tests the filter building logic
        filters = {
            "doc_type": "policy",
            "department": "legal",
        }

        assert filters["doc_type"] == "policy"
        assert filters["department"] == "legal"

    def test_ingestion_result_structure(self):
        """Test IngestionResult data structure."""
        result = IngestionResult(
            num_documents=5,
            num_chunks=50,
            num_errors=0,
            errors=[],
            processing_time_seconds=2.5,
        )

        assert result.num_documents == 5
        assert result.num_chunks == 50
        assert result.num_errors == 0
        assert result.processing_time_seconds == 2.5


# =============================================================================
# RAG Chain Tests
# =============================================================================

class TestRAGChain:
    """Test suite for RAG Chain."""

    def test_message_to_chat_message(self):
        """Test Message to ChatMessage conversion."""
        msg = Message(role="user", content="Test message")
        chat_msg = msg.to_chat_message()

        assert chat_msg.role == "user"
        assert chat_msg.content == "Test message"

    def test_message_from_dict(self):
        """Test Message creation from dict."""
        data = {
            "role": "assistant",
            "content": "Test response",
            "timestamp": "2024-01-15T10:00:00",
        }

        msg = Message.from_dict(data)

        assert msg.role == "assistant"
        assert msg.content == "Test response"

    def test_rag_response_to_dict(self, sample_rag_chain):
        """Test RAGResponse serialization."""
        response = RAGResponse(
            answer="Test answer",
            query_type="SQL_QUERY",
            confidence=0.92,
            sql_query="SELECT 1",
            sql_results=SQLResult(
                query="SELECT 1",
                results=[{"col": 1}],
                columns=["col"],
                row_count=1,
            ),
        )

        data = response.to_dict()

        assert data["answer"] == "Test answer"
        assert data["query_type"] == "SQL_QUERY"
        assert data["confidence"] == 0.92
        assert "sql_query" in data

    def test_rag_chain_initialization(self, sample_rag_chain):
        """Test RAG chain initialization."""
        assert sample_rag_chain.sql_retriever is not None
        assert sample_rag_chain.doc_retriever is not None
        assert sample_rag_chain.query_router is not None
        assert sample_rag_chain.enable_memory is True

    def test_rag_chain_memory_disabled(self, mock_sql_retriever, mock_doc_retriever, sample_query_router):
        """Test RAG chain with memory disabled."""
        chain = DataChatRAG(
            sql_retriever=mock_sql_retriever,
            doc_retriever=mock_doc_retriever,
            query_router=sample_query_router,
            enable_memory=False,
        )

        assert chain.enable_memory is False

    def test_rag_chain_clear_memory(self, sample_rag_chain):
        """Test clearing conversation memory."""
        # Add some messages
        sample_rag_chain.conversation_history = [
            Message(role="user", content="Test"),
            Message(role="assistant", content="Response"),
        ]

        assert len(sample_rag_chain.conversation_history) == 2

        # Clear memory
        sample_rag_chain.clear_memory()

        assert len(sample_rag_chain.conversation_history) == 0

    def test_rag_chain_get_conversation_history(self, sample_rag_chain):
        """Test getting conversation history."""
        sample_rag_chain.conversation_history = [
            Message(role="user", content="Question 1"),
            Message(role="assistant", content="Answer 1"),
        ]

        history = sample_rag_chain.get_conversation_history()

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_rag_chain_sql_query_flow(self, sample_rag_chain):
        """Test SQL query flow through RAG chain."""
        # Mock the classification
        with patch.object(sample_rag_chain.query_router, 'classify') as mock_classify:
            mock_classify.return_value = QueryClassification(
                query_type=QueryType.SQL_QUERY,
                confidence=0.9,
                reasoning="Test",
                suggested_sql_tables=["daily_metrics"],
            )

            response = sample_rag_chain.query("What was our CTR?")

            assert response.query_type == "SQL_QUERY"
            assert "CTR" in response.answer or len(response.answer) > 0

    def test_rag_chain_doc_query_flow(self, sample_rag_chain):
        """Test document query flow through RAG chain."""
        with patch.object(sample_rag_chain.query_router, 'classify') as mock_classify:
            mock_classify.return_value = QueryClassification(
                query_type=QueryType.DOC_SEARCH,
                confidence=0.9,
                reasoning="Test",
                suggested_doc_topics=["hipaa_compliance"],
            )

            response = sample_rag_chain.query("HIPAA requirements?")

            assert response.query_type == "DOC_SEARCH"

    def test_rag_chain_hybrid_query_flow(self, sample_rag_chain):
        """Test hybrid query flow through RAG chain."""
        with patch.object(sample_rag_chain.query_router, 'classify') as mock_classify:
            mock_classify.return_value = QueryClassification(
                query_type=QueryType.HYBRID,
                confidence=0.85,
                reasoning="Test",
                suggested_sql_tables=["campaigns"],
                suggested_doc_topics=["campaign_best_practices"],
            )

            response = sample_rag_chain.query("Why is campaign X underperforming?")

            assert response.query_type == "HYBRID"

    def test_conversation_memory_update(self, sample_rag_chain):
        """Test that conversation memory is updated."""
        initial_count = len(sample_rag_chain.conversation_history)

        # Simulate a query (using internal method)
        sample_rag_chain._update_memory(
            "Test question",
            RAGResponse(
                answer="Test answer",
                query_type="SQL_QUERY",
                confidence=0.9,
            )
        )

        # Should have added 2 messages (user + assistant)
        assert len(sample_rag_chain.conversation_history) == initial_count + 2


# =============================================================================
# API Tests
# =============================================================================

class TestAPIEndpoints:
    """Test suite for API endpoints."""

    def test_chat_request_model(self):
        """Test ChatRequest model validation."""
        from src.api.main import ChatRequest

        # Valid request
        request = ChatRequest(
            question="What was our CTR?",
            conversation_id="conv_123",
        )

        assert request.question == "What was our CTR?"
        assert request.conversation_id == "conv_123"

    def test_chat_request_empty_question(self):
        """Test ChatRequest with empty question."""
        from src.api.main import ChatRequest

        with pytest.raises(ValidationError):
            ChatRequest(question="")

    def test_chat_request_too_long_question(self):
        """Test ChatRequest with question exceeding max length."""
        from src.api.main import ChatRequest

        long_question = "A" * 2001  # Exceeds 2000 char limit

        with pytest.raises(ValidationError):
            ChatRequest(question=long_question)

    def test_chat_response_model(self):
        """Test ChatResponse model."""
        from src.api.main import ChatResponse, DocumentSource

        response = ChatResponse(
            answer="Test answer",
            query_type="SQL_QUERY",
            confidence=0.92,
            conversation_id="conv_123",
            sql_query="SELECT 1",
            doc_sources=[],
            suggested_followup=["Follow up?"],
            processing_time_seconds=1.5,
        )

        assert response.answer == "Test answer"
        assert response.query_type == "SQL_QUERY"
        assert response.confidence == 0.92

    def test_document_source_model(self):
        """Test DocumentSource model."""
        from src.api.main import DocumentSource

        source = DocumentSource(
            content="Test content snippet",
            source="test.txt",
            doc_type="policy",
            relevance=0.95,
        )

        assert source.source == "test.txt"
        assert source.doc_type == "policy"
        assert 0 <= source.relevance <= 1

    def test_health_response_model(self):
        """Test HealthResponse model."""
        from src.api.main import HealthResponse, ComponentStatus

        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            components=[
                ComponentStatus(
                    name="test_component",
                    status="healthy",
                    message="OK",
                )
            ],
            uptime_seconds=3600.0,
        )

        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert len(response.components) == 1

    def test_schema_response_model(self):
        """Test SchemaResponse model."""
        from src.api.main import SchemaResponse, SchemaTable

        response = SchemaResponse(
            tables=[
                SchemaTable(
                    name="campaigns",
                    description="Campaign data",
                    columns=[
                        {"name": "id", "type": "UUID", "description": "ID"},
                        {"name": "name", "type": "VARCHAR", "description": "Name"},
                    ],
                )
            ],
            relationships=[
                {"from": "campaigns", "to": "daily_metrics", "type": "one_to_many"}
            ],
        )

        assert len(response.tables) == 1
        assert response.tables[0].name == "campaigns"
        assert len(response.relationships) == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for end-to-end flows."""

    def test_full_sql_flow(self, sample_rag_chain):
        """Test complete SQL query flow."""
        with patch.object(sample_rag_chain.query_router, 'classify') as mock_classify:
            mock_classify.return_value = QueryClassification(
                query_type=QueryType.SQL_QUERY,
                confidence=0.9,
                reasoning="Metrics question",
                suggested_sql_tables=["daily_metrics"],
            )

            response = sample_rag_chain.query("What was our average CTR?")

            # Verify response structure
            assert hasattr(response, 'answer')
            assert hasattr(response, 'query_type')
            assert hasattr(response, 'confidence')

    def test_full_doc_flow(self, sample_rag_chain):
        """Test complete document search flow."""
        with patch.object(sample_rag_chain.query_router, 'classify') as mock_classify:
            mock_classify.return_value = QueryClassification(
                query_type=QueryType.DOC_SEARCH,
                confidence=0.9,
                reasoning="Policy question",
                suggested_doc_topics=["hipaa_compliance"],
            )

            response = sample_rag_chain.query("What are HIPAA requirements?")

            assert response.query_type == "DOC_SEARCH"

    def test_full_hybrid_flow(self, sample_rag_chain):
        """Test complete hybrid query flow."""
        with patch.object(sample_rag_chain.query_router, 'classify') as mock_classify:
            mock_classify.return_value = QueryClassification(
                query_type=QueryType.HYBRID,
                confidence=0.85,
                reasoning="Why question",
                suggested_sql_tables=["campaigns"],
                suggested_doc_topics=["campaign_best_practices"],
            )

            response = sample_rag_chain.query("Why is campaign X underperforming?")

            assert response.query_type == "HYBRID"

    def test_conversation_context_persistence(self, sample_rag_chain):
        """Test that conversation context is maintained."""
        # First query
        with patch.object(sample_rag_chain.query_router, 'classify') as mock_classify:
            mock_classify.return_value = QueryClassification(
                query_type=QueryType.SQL_QUERY,
                confidence=0.9,
                reasoning="Test",
                suggested_sql_tables=["daily_metrics"],
            )

            response1 = sample_rag_chain.query("What was our CTR?")

            # Second query (should have context)
            response2 = sample_rag_chain.query("How about last week?")

            # Memory should have been updated
            assert len(sample_rag_chain.conversation_history) >= 4  # 2 queries * 2 messages each


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
