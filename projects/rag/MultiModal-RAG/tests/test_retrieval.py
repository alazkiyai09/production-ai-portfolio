# ============================================================
# Enterprise-RAG: Retrieval Tests
# ============================================================
"""
Tests for retrieval components:
- Embedding service
- Vector store
- BM25 sparse retriever
- Hybrid retriever
- Reranker
"""

from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

from src.retrieval.embedding_service import EmbeddingService
from src.retrieval.vector_store import (
    SearchResult,
    VectorStoreBase,
    ChromaVectorStore,
)
from src.retrieval.sparse_retriever import (
    BM25Retriever,
    SparseSearchResult,
)
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
    LLMProvider,
)
from src.retrieval.reranker import CrossEncoderReranker, RerankedSearchResult


# ============================================================
# Embedding Service Tests
# ============================================================

class TestEmbeddingService:
    """Tests for EmbeddingService."""

    def test_initialization(self):
        """Test service initialization."""
        service = EmbeddingService(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            batch_size=16,
        )

        assert service.model_name == "all-MiniLM-L6-v2"
        assert service.batch_size == 16
        assert service.embedding_dimension == 384

    def test_embedding_dimension_property(self, mock_embedding_service):
        """Test embedding dimension property."""
        assert mock_embedding_service.embedding_dimension == 384

    def test_embed_single(self, mock_embedding_service):
        """Test embedding a single text."""
        text = "This is a test"
        embedding = mock_embedding_service.embed_single(text)

        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_embed_texts(self, mock_embedding_service):
        """Test embedding multiple texts."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = mock_embedding_service.embed_texts(texts)

        assert embeddings.shape == (3, 384)

    def test_similarity_calculation(self, mock_embedding_service):
        """Test cosine similarity calculation."""
        # Create two similar vectors
        vec1 = np.random.rand(384).astype(np.float32)
        vec2 = vec1 + 0.01  # Very similar

        sim = mock_embedding_service.similarity(vec1, vec2)

        assert sim > 0.95  # Should be very high

    def test_similarity_identical(self, mock_embedding_service):
        """Test similarity of identical vectors."""
        vec = np.random.rand(384).astype(np.float32)
        sim = mock_embedding_service.similarity(vec, vec)

        assert sim == pytest.approx(1.0)

    def test_similarities_batch(self, mock_embedding_service):
        """Test batch similarity calculation."""
        query = np.random.rand(384).astype(np.float32)
        documents = np.random.rand(5, 384).astype(np.float32)

        sims = mock_embedding_service.similarities(query, documents)

        assert sims.shape == (5,)
        assert all(0 <= s <= 1 for s in sims)

    def test_cache_functionality(self):
        """Test embedding caching."""
        service = EmbeddingService(
            model_name="test-model",
            cache_size=10,
            enable_cache=True,
        )

        # Mock model
        service._model = MagicMock()
        service._model.encode = Mock(return_value=np.random.rand(1, 384))

        # Embed same text twice
        text = "Test text"
        emb1 = service.embed_single(text)
        emb2 = service.embed_single(text)

        np.testing.assert_array_equal(emb1, emb2)

    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        service = EmbeddingService(
            model_name="test-model",
            cache_size=100,
        )

        stats = service.get_cache_stats()

        assert "cache_size" in stats
        assert "cache_max_size" in stats
        assert stats["cache_max_size"] == 100


# ============================================================
# Vector Store Tests
# ============================================================

class TestVectorStore:
    """Tests for vector store implementations."""

    def test_add_documents(self, test_vector_store, sample_documents):
        """Test adding documents to vector store."""
        new_docs = [
            sample_documents[0],
        ]

        embeddings = np.random.rand(1, 384).astype(np.float32)

        count = test_vector_store.add_documents(new_docs, embeddings)

        assert count == 1

    def test_search(self, test_vector_store):
        """Test searching the vector store."""
        query = np.random.rand(384).astype(np.float32)

        results = test_vector_store.search(query, top_k=5)

        assert len(results) <= 5
        for result in results:
            assert isinstance(result, SearchResult)
            assert hasattr(result, "score")

    def test_search_returns_sorted_results(self, test_vector_store):
        """Test that search results are sorted by score."""
        query = np.random.rand(384).astype(np.float32)

        results = test_vector_store.search(query, top_k=10)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_delete_documents(self, test_vector_store):
        """Test deleting documents."""
        doc_ids = [sample_documents[0].doc_id]

        deleted = test_vector_store.delete(doc_ids)

        assert deleted >= 0

    def test_get_stats(self, test_vector_store):
        """Test getting vector store statistics."""
        stats = test_vector_store.get_stats()

        assert hasattr(stats, "total_documents")
        assert hasattr(stats, "total_chunks")

    def test_clear(self, test_vector_store):
        """Test clearing the vector store."""
        test_vector_store.clear()

        stats = test_vector_store.get_stats()
        assert stats.total_chunks == 0


# ============================================================
# BM25 Retriever Tests
# ============================================================

class TestBM25Retriever:
    """Tests for BM25 sparse retriever."""

    def test_initialization(self):
        """Test BM25 retriever initialization."""
        retriever = BM25Retriever(
            k1=1.5,
            b=0.75,
            normalize_scores=True,
        )

        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
        assert retriever.normalize_scores is True

    def test_build_index(self, sample_documents):
        """Test building BM25 index."""
        retriever = BM25Retriever()
        retriever.build_index(sample_documents)

        assert len(retriever.documents) == len(sample_documents)
        assert len(retriever.tokenized_corpus) == len(sample_documents)

    def test_search(self, test_bm25_retriever):
        """Test BM25 search."""
        results = test_bm25_retriever.search(
            query="refund policy",
            top_k=5,
        )

        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, SparseSearchResult)
            assert hasattr(result, "score")

    def test_search_returns_sorted(self, test_bm25_retriever):
        """Test that search results are sorted."""
        results = test_bm25_retriever.search(
            query="test query",
            top_k=10,
        )

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_filters(self, test_bm25_retriever):
        """Test search with metadata filters."""
        # Add metadata to documents
        for doc in test_bm25_retriever.documents:
            doc.metadata["category"] = "test"

        results = test_bm25_retriever.search(
            query="test",
            top_k=5,
            filters={"category": "test"},
        )

        # Should return results
        assert isinstance(results, list)

    def test_add_documents(self, test_bm25_retriever):
        """Test adding documents to existing index."""
        initial_count = len(test_bm25_retriever.documents)

        from src.ingestion.document_processor import Document
        new_doc = Document(
            content="New document about testing.",
            metadata={"source": "test.txt"},
            doc_id="doc_new",
            chunk_id="doc_new_chunk_0",
        )

        test_bm25_retriever.add_documents([new_doc])

        assert len(test_bm25_retriever.documents) == initial_count + 1

    def test_get_stats(self, test_bm25_retriever):
        """Test getting retriever statistics."""
        stats = test_bm25_retriever.get_stats()

        assert stats.total_chunks == len(test_bm25_retriever.documents)
        assert stats.total_documents <= stats.total_chunks

    def test_normalize_scores(self):
        """Test score normalization."""
        retriever = BM25Retriever()

        scores = np.array([-1.0, 0.0, 1.0, 5.0])
        normalized = retriever._normalize_scores(scores)

        assert all(0 <= s <= 1 for s in normalized)
        assert normalized[0] < 0.5  # Negative becomes < 0.5
        assert normalized[2] > 0.5  # Positive becomes > 0.5

    def test_tokenize(self):
        """Test text tokenization."""
        retriever = BM25Retriever()

        text = "Hello, World! This is a test."
        tokens = retriever._tokenize(text)

        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)
        # Stopwords should be removed
        assert "a" not in tokens  # Common stopword


# ============================================================
# Hybrid Retriever Tests
# ============================================================

class TestHybridRetriever:
    """Tests for hybrid retriever."""

    def test_initialization(self, test_hybrid_retriever):
        """Test hybrid retriever initialization."""
        assert test_hybrid_retriever.dense_weight == 0.7
        assert test_hybrid_retriever.sparse_weight == 0.3
        assert test_hybrid_retriever.rrf_k == 60

    def test_initialization_invalid_weights(self):
        """Test that invalid weights raise error."""
        with pytest.raises(ValueError, match="0-1"):
            HybridRetriever(
                vector_store=MagicMock(),
                embedding_service=MagicMock(),
                bm25_retriever=MagicMock(),
                dense_weight=1.5,  # Invalid
                sparse_weight=0.5,
            )

    def test_retrieve(self, test_hybrid_retriever):
        """Test hybrid retrieval."""
        results = test_hybrid_retriever.retrieve(
            query="test query",
            top_k=5,
            use_hybrid=True,
        )

        assert isinstance(results, list)
        assert all(isinstance(r, HybridSearchResult) for r in results)

    def test_retrieve_dense_only(self, test_hybrid_retriever):
        """Test dense-only retrieval."""
        results = test_hybrid_retriever.retrieve(
            query="test query",
            top_k=5,
            use_hybrid=False,  # Dense only
        )

        # Should still return results
        assert isinstance(results, list)

    def test_retrieve_sorted_by_fused_score(self, test_hybrid_retriever):
        """Test that results are sorted by fused score."""
        results = test_hybrid_retriever.retrieve(
            query="test query",
            top_k=10,
            use_hybrid=True,
        )

        scores = [r.fused_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_add_documents(self, test_hybrid_retriever):
        """Test adding documents to both indexes."""
        from src.ingestion.document_processor import Document

        initial_chunks = test_hybrid_retriever.retriever.retrieve("test", top_k=10)
        initial_count = len(initial_chunks)

        new_doc = Document(
            content="New document for testing hybrid search.",
            metadata={"source": "hybrid_test.txt"},
            doc_id="doc_hybrid",
            chunk_id="doc_hybrid_chunk_0",
        )

        added = test_hybrid_retriever.add_documents([new_doc])

        assert added == 1

    def test_reciprocal_rank_fusion(self, test_hybrid_retriever):
        """Test Reciprocal Rank Fusion algorithm."""
        # Create mock results
        dense_results = [("chunk_1", 0.9), ("chunk_2", 0.7)]
        sparse_results = [("chunk_2", 0.8), ("chunk_3", 0.6)]

        fused = test_hybrid_retriever._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k=60,
        )

        # chunk_2 appears in both, should have highest score
        assert fused[0][0] == "chunk_2"
        assert fused[0][1] > fused[1][1]

    def test_get_stats(self, test_hybrid_retriever):
        """Test getting retriever statistics."""
        stats = test_hybrid_retriever.get_stats()

        assert stats["provider"] == test_hybrid_retriever.llm_provider.value


# ============================================================
# Reranker Tests
# ============================================================

class TestCrossEncoderReranker:
    """Tests for cross-encoder reranker."""

    def test_initialization(self):
        """Test reranker initialization."""
        reranker = CrossEncoderReranker(
            model_name="test-model",
            device="cpu",
            batch_size=8,
        )

        assert reranker.model_name == "test-model"
        assert reranker.batch_size == 8
        assert reranker.normalize_scores is True

    def test_rerank(self, mock_reranker):
        """Test reranking functionality."""
        from src.retrieval.reranker import HybridSearchResult

        # Create mock results
        results = [
            HybridSearchResult(
                doc_id="doc_1",
                chunk_id="doc_1_chunk_0",
                content="Result 1",
                metadata={},
                dense_score=0.7,
                sparse_score=0.6,
                fused_score=0.65,
            ),
            HybridSearchResult(
                doc_id="doc_2",
                chunk_id="doc_2_chunk_0",
                content="Result 2",
                metadata={},
                dense_score=0.5,
                sparse_score=0.8,
                fused_score=0.65,
            ),
        ]

        # Mock rerank will return the same results
        mock_reranker.reranker = Mock(return_value=results)

        reranked = mock_reranker.rerank(
            query="test query",
            results=results,
            top_k=5,
        )

        assert len(reranked) <= 5
        assert all(isinstance(r, RerankedSearchResult) for r in reranked)

    def test_batch_score(self):
        """Test batch scoring of query-document pairs."""
        service = CrossEncoderReranker()

        # Mock the model
        service._model = MagicMock()
        service._model.predict = Mock(return_value=np.array([0.5, 0.7, 0.3]))

        scores = service._batch_score("query", ["doc1", "doc2", "doc3"])

        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_normalize_scores_sigmoid(self):
        """Test sigmoid normalization."""
        reranker = CrossEncoderReranker()

        raw_scores = np.array([-5, -1, 0, 1, 5])
        normalized = reranker._normalize_scores(raw_scores)

        assert all(0 <= s <= 1 for s in normalized)
        assert normalized[-1] > normalized[-2]  # Higher input → higher output

    def test_device_resolution_auto(self):
        """Test device resolution with 'auto'."""
        reranker = CrossEncoderReranker(device="auto")

        device = reranker._resolve_device("auto")

        assert device in ["cpu", "cuda", "mps"]

    def test_get_model_info(self):
        """Test getting model information."""
        reranker = CrossEncoderReranker()

        info = reranker.get_model_info()

        assert "name" in info
        assert "is_loaded" in info


# ============================================================
# SearchResult Tests
# ============================================================

class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a search result."""
        result = SearchResult(
            doc_id="doc_1",
            chunk_id="doc_1_chunk_0",
            content="Test content",
            metadata={"source": "test"},
            score=0.85,
        )

        assert result.doc_id == "doc_1"
        assert result.score == 0.85

    def test_search_result_to_dict(self):
        """Test converting to dictionary."""
        result = SearchResult(
            doc_id="doc_1",
            chunk_id="doc_1_chunk_0",
            content="Test",
            metadata={},
            score=0.9,
        )

        result_dict = result.to_dict()

        assert result_dict["doc_id"] == "doc_1"
        assert result_dict["score"] == 0.9
