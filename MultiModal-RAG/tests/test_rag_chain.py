# ============================================================
# Enterprise-RAG: RAG Chain Tests
# ============================================================
"""
Tests for RAG chain and generation components.
"""

from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from typing import List

import pytest

from src.generation.rag_chain import (
    Citation,
    LLMProvider,
    Message,
    RAGChain,
    RAGResponse,
)


# ============================================================
# Data Class Tests
# ============================================================

class TestCitation:
    """Tests for Citation dataclass."""

    def test_citation_creation(self):
        """Test creating a citation."""
        citation = Citation(
            source="policy.pdf",
            chunk_id="doc_123_chunk_0",
            content_preview="Refunds are processed...",
            relevance_score=0.92,
            page_number=5,
        )

        assert citation.source == "policy.pdf"
        assert citation.relevance_score == 0.92
        assert citation.page_number == 5

    def test_citation_to_dict(self):
        """Test converting citation to dictionary."""
        citation = Citation(
            source="test.pdf",
            chunk_id="doc_1",
            content_preview="Preview...",
            relevance_score=0.85,
        )

        citation_dict = citation.to_dict()

        assert citation_dict["source"] == "test.pdf"
        assert citation_dict["relevance_score"] == 0.85


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Test creating a message."""
        message = Message(
            role="user",
            content="Test message",
        )

        assert message.role == "user"
        assert message.content == "Test message"
        assert isinstance(message.timestamp, str)


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""

    def test_response_creation(self):
        """Test creating a RAG response."""
        response = RAGResponse(
            answer="This is the answer.",
            citations=[],
            retrieval_results=[],
            model_used="gpt-4o-mini",
            provider_used=LLMProvider.OPENAI,
            processing_time=2.5,
            retrieval_time=0.3,
            rerank_time=0.2,
            generation_time=2.0,
        )

        assert response.answer == "This is the answer."
        assert response.processing_time == 2.5
        assert response.provider_used == LLMProvider.OPENAI

    def test_response_to_dict(self):
        """Test converting response to dictionary."""
        from src.generation.rag_chain import Citation

        response = RAGResponse(
            answer="Test answer",
            citations=[Citation(
                source="test.pdf",
                chunk_id="doc_1",
                content_preview="...",
                relevance_score=0.9,
            )],
            retrieval_results=[],
            model_used="test-model",
            provider_used=LLMProvider.OPENAI,
            processing_time=1.0,
        )

        response_dict = response.to_dict()

        assert response_dict["answer"] == "Test answer"
        assert "citations" in response_dict
        assert "num_sources" in response_dict


# ============================================================
# RAG Chain Tests
# ============================================================

class TestRAGChain:
    """Tests for RAG chain functionality."""

    def test_initialization(self, test_hybrid_retriever, mock_reranker):
        """Test RAG chain initialization."""
        chain = RAGChain(
            retriever=test_hybrid_retriever,
            reranker=mock_reranker,
            llm_provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=0.1,
        )

        assert chain.llm_provider == LLMProvider.OPENAI
        assert chain.model_name == "gpt-4o-mini"
        assert chain.temperature == 0.1

    def test_query_returns_response(self, mock_rag_chain):
        """Test that query returns a RAGResponse."""
        response = mock_rag_chain.query(
            question="What is the refund policy?",
        )

        assert isinstance(response, RAGResponse)
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0

    def test_query_with_top_k(self, mock_rag_chain):
        """Test query with custom top_k."""
        response = mock_rag_chain.query(
            question="Test question",
            top_k_retrieve=10,
            top_k_rerank=3,
        )

        assert response is not None

    def test_query_with_filters(self, mock_rag_chain):
        """Test query with metadata filters."""
        filters = {"file_type": "pdf"}

        response = mock_rag_chain.query(
            question="Test question",
            filters=filters,
        )

        assert response is not None

    def test_query_without_reranking(self, mock_rag_chain):
        """Test query without reranking."""
        response = mock_rag_chain.query(
            question="Test question",
            use_reranking=False,
        )

        assert response is not None

    def test_query_updates_history(self, mock_rag_chain):
        """Test that query updates conversation history."""
        initial_length = len(mock_rag_chain.conversation_history)

        mock_rag_chain.query("Test question")

        assert len(mock_rag_chain.conversation_history) == initial_length + 2


class TestContextBuilding:
    """Tests for context building functionality."""

    def test_build_context_from_results(self, sample_documents):
        """Test building context from search results."""
        from src.generation.rag_chain import RAGChain

        chain = RAGChain(
            retriever=MagicMock(),
            reranker=MagicMock(),
        )

        context = chain._build_context(sample_documents)

        assert isinstance(context, str)
        assert len(context) > 0

        # Check that sources are numbered
        assert "[1]" in context
        assert "[2]" in context

    def test_context_includes_metadata(self, sample_documents):
        """Test that context includes source metadata."""
        from src.generation.rag_chain import RAGChain

        chain = RAGChain(
            retriever=MagicMock(),
            reranker=MagicMock(),
        )

        context = chain._build_context(sample_documents)

        # Should include source from metadata
        assert "Source:" in context or "source:" in context.lower()

    def test_default_system_prompt(self):
        """Test getting default system prompt."""
        from src.generation.rag_chain import RAGChain

        chain = RAGChain(
            retriever=MagicMock(),
            reranker=MagicMock(),
        )

        prompt = chain._get_default_system_prompt()

        assert len(prompt) > 0
        assert "answer based on the provided context" in prompt.lower()


class TestCitationExtraction:
    """Tests for citation extraction."""

    def test_extract_citations_basic(self):
        """Test basic citation extraction."""
        from src.generation.rag_chain import RAGChain

        chain = RAGChain(
            retriever=MagicMock(),
            reranker=MagicMock(),
        )

        answer = "According to the policy [1], refunds take 5-7 days. See also [2] for details."
        results = [
            MagicMock(chunk_id="chunk_1", doc_id="doc_1"),
            MagicMock(chunk_id="chunk_2", doc_id="doc_2"),
        ]

        for i, result in enumerate(results):
            result.doc_id = f"doc_{i+1}"
            result.content = f"Content {i+1}"
            result.metadata = {"source": f"source{i+1}.pdf"}
            result.fused_score = 0.8

        citations = chain._extract_citations(answer, results)

        assert len(citations) == 2
        assert citations[0].source == "source1.pdf"
        assert citations[1].source == "source2.pdf"

    def test_extract_citations_no_duplicates(self):
        """Test that duplicate citations are handled."""
        from src.generation.rag_chain import RAGChain

        chain = RAGChain(
            retriever=MagicMock(),
            reranker=MagicMock(),
        )

        answer = "See [1] and [1] for details."  # Duplicate
        results = [MagicMock(chunk_id="chunk_1", doc_id="doc_1")]

        results[0].doc_id = "doc_1"
        results[0].content = "Content 1"
        results[0].metadata = {"source": "test.pdf"}
        results[0].fused_score = 0.9

        citations = chain._extract_citations(answer, results)

        # Should deduplicate
        assert len(citations) <= 1

    def test_extract_citations_no_matches(self):
        """Test extraction with no citation markers."""
        from src.generation.rag_chain import RAGChain

        chain = RAGChain(
            retriever=MagicMock(),
            reranker=MagicMock(),
        )

        answer = "This answer has no citations."
        results = []

        citations = chain._extract_citations(answer, results)

        assert len(citations) == 0


class TestLLMProviders:
    """Tests for LLM provider implementations."""

    @patch("src.generation.rag_chain.OpenAI")
    def test_openai_generation(self, mock_openai):
        """Test OpenAI generation."""
        from src.generation.rag_chain import RAGChain

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test answer"
        mock_response.choices[0].message.content = "Test answer"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        chain = RAGChain(
            retriever=MagicMock(),
            reranker=MagicMock(),
            llm_provider=LLMProvider.OPENAI,
        )
        chain._llm_client = mock_client

        # Mock messages
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Question: test"},
        ]

        answer, usage = chain._generate_openai(messages)

        assert answer == "Test answer"
        assert usage["total_tokens"] == 150

    @patch("src.generation.rag_chain.Anthropic")
    def test_anthropic_generation(self, mock_anthropic):
        """Test Anthropic generation."""
        from src.generation.rag_chain import RAGChain

        # Mock Anthropic client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Anthropic answer")]

        mock_client.messages.create = MagicMock(return_value=mock_response)
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        chain = RAGChain(
            retriever=MagicMock(),
            reranker=MagicMock(),
            llm_provider=LLMProvider.ANTHROPIC,
        )
        chain._llm_client = mock_client

        messages = [
            {"role": "user", "content": "Question: test"},
        ]

        answer, usage = chain._generate_anthropic(messages)

        assert answer == "Anthropic answer"
        assert usage["total_tokens"] == 150


class TestConversationManagement:
    """Tests for conversation history management."""

    def test_clear_history(self, mock_rag_chain):
        """Test clearing conversation history."""
        # Add some history
        mock_rag_chain.conversation_history = [
            Message(role="user", content="Question 1"),
            Message(role="assistant", content="Answer 1"),
        ]

        mock_rag_chain.clear_history()

        assert len(mock_rag_chain.conversation_history) == 0

    def test_get_history(self, mock_rag_chain):
        """Test getting conversation history."""
        history = [
            Message(role="user", content="Question 1"),
            Message(role="assistant", content="Answer 1"),
        ]

        mock_rag_chain.conversation_history = history

        retrieved = mock_rag_chain.get_history()

        assert len(retrieved) == 2
        assert retrieved[0].content == "Question 1"

    def test_set_system_prompt(self, mock_rag_chain):
        """Test setting custom system prompt."""
        custom_prompt = "You are a helpful assistant."

        mock_rag_chain.set_system_prompt(custom_prompt)

        assert mock_rag_chain.system_prompt == custom_prompt


class TestErrorHandling:
    """Tests for error handling in RAG chain."""

    def test_query_empty_question_raises_error(self, mock_rag_chain):
        """Test that empty question raises error."""
        with pytest.raises(ValueError, match="empty"):
            mock_rag_chain.query("")

    def test_query_empty_question_whitespace(self, mock_rag_chain):
        """Test that whitespace-only question raises error."""
        with pytest.raises(ValueError, match="empty"):
            mock_rag_chain.query("   ")

    def test_query_no_results(self, test_hybrid_retriever):
        """Test query when no results are found."""
        from src.generation.rag_chain import RAGChain

        # Mock retriever to return no results
        test_hybrid_retriever.retrieve = Mock(return_value=[])

        chain = RAGChain(
            retriever=test_hybrid_retriever,
            reranker=MagicMock(),
        )

        response = chain.query("test question")

        # Should return a response indicating no information
        assert "enough information" in response.answer.lower() or "not found" in response.answer.lower()


class TestGeneration:
    """Tests for LLM generation."""

    def test_generate_includes_context(self):
        """Test that generation includes context."""
        from src.generation.rag_chain import RAGChain

        chain = RAGChain(
            retriever=MagicMock(),
            reranker=MagicMock(),
        )

        context = "Context: This is test context."
        question = "What is this about?"

        # Mock the LLM client
        chain._llm_client = MagicMock()
        chain._llm_client.chat.completions.create = MagicMock()

        def check_messages(messages):
            # Check that context is in messages
            message_texts = [m["content"] for m in messages]
            assert any("test context" in text.lower() for text in message_texts)
            return MagicMock()

        chain._llm_client.chat.completions.create = Mock(side_effect=check_messages)

        chain._generate(question, context)

    def test_generate_with_history(self, mock_rag_chain):
        """Test generation with conversation history."""
        # Add history
        mock_rag_chain.conversation_history = [
            Message(role="user", content="Previous question"),
            Message(role="assistant", content="Previous answer"),
        ]

        # Mock LLM to check history is included
        history_included = [False]

        def check_messages(messages):
            message_count = len(messages)
            # System + 2 history + current question = 4 messages
            history_included[0] = message_count >= 4
            return MagicMock()

        mock_rag_chain._llm_client.chat.completions.create = Mock(side_effect=check_messages)

        mock_rag_chain._generate(
            "New question",
            "Context",
            include_history=True,
        )

        assert history_included[0]


# ============================================================
# Utility Tests
# ============================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_rag_chain(self, test_hybrid_retriever, mock_reranker):
        """Test factory function."""
        from src.generation.rag_chain import create_rag_chain

        chain = create_rag_chain(
            retriever=test_hybrid_retriever,
            reranker=mock_reranker,
            llm_provider="openai",
        )

        assert chain is not None
        assert chain.retriever == test_hybrid_retriever
        assert chain.reranker == mock_reranker
