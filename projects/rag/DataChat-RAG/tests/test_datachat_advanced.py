"""
Advanced unit tests for DataChat RAG System
"""

import pytest
from typing import List, Dict, Any


class MockDataChatRAG:
    """Mock DataChat RAG for testing."""

    def __init__(self):
        self.knowledge_base = []
        self.conversation_history = []

    def add_document(self, content: str, metadata: Dict = None) -> str:
        """Add a document to the knowledge base."""
        doc_id = f"doc_{len(self.knowledge_base)}"
        self.knowledge_base.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata or {}
        })
        return doc_id

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        # Simple keyword matching for mock
        query_terms = set(query.lower().split())
        scored = []

        for doc in self.knowledge_base:
            content_lower = doc["content"].lower()
            score = sum(1 for term in query_terms if term in content_lower)
            if score > 0:
                scored.append({**doc, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate a response based on retrieved context."""
        if not context:
            return "I don't have relevant information to answer that question."

        # Simple response generation
        top_doc = context[0]["content"]
        return f"Based on the data, {top_doc[:100]}..."

    def chat(self, message: str) -> Dict[str, Any]:
        """Process a chat message."""
        # Retrieve relevant documents
        docs = self.retrieve_documents(message)

        # Generate response
        response = self.generate_response(message, docs)

        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return {
            "response": response,
            "sources": [d["id"] for d in docs],
            "confidence": len(docs) / 3.0
        }

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "total_documents": len(self.knowledge_base),
            "conversation_turns": len(self.conversation_history) // 2,
            "unique_sources": len(set(d["metadata"].get("source", "unknown") for d in self.knowledge_base))
        }


class TestDataChatRAG:
    """Test cases for DataChat RAG."""

    def setup_method(self):
        """Setup test fixtures."""
        self.rag = MockDataChatRAG()

    def test_add_document(self):
        """Test adding a document."""
        doc_id = self.rag.add_document(
            "Test content about fraud detection.",
            metadata={"source": "test"}
        )
        assert doc_id == "doc_0"
        assert len(self.rag.knowledge_base) == 1

    def test_add_multiple_documents(self):
        """Test adding multiple documents."""
        self.rag.add_document("Document 1")
        self.rag.add_document("Document 2")
        self.rag.add_document("Document 3")

        assert len(self.rag.knowledge_base) == 3

    def test_retrieve_documents(self):
        """Test document retrieval."""
        self.rag.add_document("Fraud detection using machine learning.")
        self.rag.add_document("Customer support best practices.")
        self.rag.add_document("Fraud patterns in financial transactions.")

        results = self.rag.retrieve_documents("fraud detection")

        assert len(results) > 0
        assert "fraud" in results[0]["content"].lower()

    def test_retrieve_with_top_k(self):
        """Test retrieval with top_k parameter."""
        self.rag.add_document("Document 1 about fraud")
        self.rag.add_document("Document 2 about fraud")
        self.rag.add_document("Document 3 about fraud")

        results = self.rag.retrieve_documents("fraud", top_k=2)

        assert len(results) <= 2

    def test_retrieve_empty_kb(self):
        """Test retrieval with empty knowledge base."""
        results = self.rag.retrieve_documents("test query")
        assert results == []

    def test_retrieve_no_matches(self):
        """Test retrieval with no matching documents."""
        self.rag.add_document("Content about cats")
        self.rag.add_document("Content about dogs")

        results = self.rag.retrieve_documents("quantum physics")
        assert results == []

    def test_generate_response_with_context(self):
        """Test response generation with context."""
        context = [{
            "id": "doc_1",
            "content": "Fraud detection models use feature engineering."
        }]

        response = self.rag.generate_response("What is fraud detection?", context)
        assert response is not None
        assert "Based on the data" in response

    def test_generate_response_no_context(self):
        """Test response generation without context."""
        response = self.rag.generate_response("Random question", [])
        assert "don't have relevant information" in response

    def test_chat_flow(self):
        """Test complete chat flow."""
        self.rag.add_document("Python is a programming language.")
        self.rag.add_document("Machine learning uses Python libraries.")

        result = self.rag.chat("Tell me about Python")

        assert "response" in result
        assert "sources" in result
        assert "confidence" in result
        assert len(self.rag.conversation_history) == 2

    def test_clear_history(self):
        """Test clearing conversation history."""
        self.rag.chat("Test message")
        assert len(self.rag.conversation_history) > 0

        self.rag.clear_history()
        assert len(self.rag.conversation_history) == 0

    def test_get_stats(self):
        """Test getting knowledge base statistics."""
        self.rag.add_document("Doc 1", metadata={"source": "A"})
        self.rag.add_document("Doc 2", metadata={"source": "B"})
        self.rag.chat("Test")

        stats = self.rag.get_stats()

        assert stats["total_documents"] == 2
        assert stats["conversation_turns"] == 1
        assert stats["unique_sources"] == 2

    def test_chat_confidence_score(self):
        """Test confidence score calculation."""
        self.rag.add_document("Relevant document")
        self.rag.add_document("Another relevant")
        self.rag.add_document("Third relevant")

        result = self.rag.chat("relevant query")

        assert 0 <= result["confidence"] <= 1

    def test_sources_returned(self):
        """Test that sources are properly returned."""
        doc_id = self.rag.add_document("Test content")
        result = self.rag.chat("test")

        assert isinstance(result["sources"], list)


class TestDataChatQueryHandling:
    """Test cases for query handling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.rag = MockDataChatRAG()
        self.rag.add_document("Sales increased by 20% in Q1.")
        self.rag.add_document("Customer churn rate is 5%.")
        self.rag.add_document("Revenue reached $1M in Q4.")

    def test_sales_query(self):
        """Test sales-related query."""
        result = self.rag.chat("What were the sales results?")
        assert result["response"] is not None

    def test_churn_query(self):
        """Test churn-related query."""
        result = self.rag.chat("What is the churn rate?")
        assert result["response"] is not None

    def test_revenue_query(self):
        """Test revenue-related query."""
        result = self.rag.chat("How much revenue was generated?")
        assert result["response"] is not None

    def test_irrelevant_query(self):
        """Test handling of irrelevant query."""
        result = self.rag.chat("What is the weather today?")
        assert "don't have relevant information" in result["response"]


class TestDataChatMetadataHandling:
    """Test cases for metadata handling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.rag = MockDataChatRAG()

    def test_document_with_metadata(self):
        """Test adding document with metadata."""
        doc_id = self.rag.add_document(
            "Content",
            metadata={"source": "report.pdf", "date": "2024-01-01"}
        )

        doc = self.rag.knowledge_base[0]
        assert doc["metadata"]["source"] == "report.pdf"
        assert doc["metadata"]["date"] == "2024-01-01"

    def test_document_without_metadata(self):
        """Test adding document without metadata."""
        self.rag.add_document("Content")

        doc = self.rag.knowledge_base[0]
        assert doc["metadata"] == {}

    def test_filter_by_source(self):
        """Test filtering documents by source."""
        self.rag.add_document("Doc 1", metadata={"source": "A"})
        self.rag.add_document("Doc 2", metadata={"source": "B"})
        self.rag.add_document("Doc 3", metadata={"source": "A"})

        source_a_docs = [d for d in self.rag.knowledge_base if d["metadata"].get("source") == "A"]
        assert len(source_a_docs) == 2
