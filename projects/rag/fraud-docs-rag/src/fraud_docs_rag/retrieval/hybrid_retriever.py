"""
Hybrid Retriever Module for FraudDocs-RAG.

This module implements hybrid retrieval combining:
- Vector similarity search using ChromaDB
- Metadata filtering for targeted retrieval
- Cross-encoder reranking for improved accuracy
- Source citation formatting for context
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError(
        "ChromaDB is required. Install with: pip install chromadb"
    )

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining vector search, metadata filtering, and reranking.

    This retriever provides:
    - Semantic similarity search via ChromaDB vector store
    - Metadata filtering by document category
    - Cross-encoder reranking for improved relevance
    - Formatted context with source citations

    Attributes:
        collection_name: Name of the ChromaDB collection
        chroma_path: Path to ChromaDB persistent storage
        top_k: Number of initial results to retrieve
        rerank_top_n: Number of results after reranking
        cross_encoder_model: CrossEncoder model for reranking
        vector_store: ChromaVectorStore instance
        index: VectorStoreIndex for queries

    Example:
        >>> retriever = HybridRetriever(
        ...     collection_name="financial_docs",
        ...     chroma_path="./data/chroma_db",
        ...     top_k=10,
        ...     rerank_top_n=5
        ... )
        >>> retriever.load_index()
        >>> nodes = retriever.retrieve("What are AML reporting requirements?")
        >>> context = retriever.format_context(nodes)
    """

    def __init__(
        self,
        collection_name: str = "financial_documents",
        chroma_path: str | Path = "./data/chroma_db",
        top_k: int = 10,
        rerank_top_n: int = 5,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
    ) -> None:
        """
        Initialize the hybrid retriever.

        Args:
            collection_name: Name of the ChromaDB collection
            chroma_path: Path to ChromaDB persistent storage directory
            top_k: Number of documents to retrieve initially
            rerank_top_n: Number of documents to return after reranking
            cross_encoder_model: HuggingFace model for cross-encoder reranking
            embedding_model: HuggingFace model for embeddings
            device: Device for cross-encoder ('cpu' or 'cuda')

        Raises:
            ValueError: If top_k <= 0 or rerank_top_n > top_k
            RuntimeError: If cross-encoder model fails to load
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if rerank_top_n > top_k:
            raise ValueError(
                f"rerank_top_n ({rerank_top_n}) cannot exceed top_k ({top_k})"
            )

        self.collection_name = collection_name
        self.chroma_path = Path(chroma_path)
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.embedding_model = embedding_model
        self.device = device

        # Initialize ChromaDB client
        self._chroma_client: chromadb.Client | None = None
        self._collection: chromadb.Collection | None = None
        self.vector_store: ChromaVectorStore | None = None
        self.index: VectorStoreIndex | None = None
        self._cross_encoder: CrossEncoder | None = None

        # Create storage directory if it doesn't exist
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized HybridRetriever: "
            f"collection={collection_name}, "
            f"top_k={top_k}, "
            f"rerank_top_n={rerank_top_n}"
        )

    def _init_chromadb(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            logger.info(f"Connected to ChromaDB at: {self.chroma_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}") from e

    def _get_or_create_collection(self) -> chromadb.Collection:
        """
        Get existing collection or create new one.

        Returns:
            ChromaDB collection object
        """
        if self._chroma_client is None:
            self._init_chromadb()

        if self._chroma_client is None:
            raise RuntimeError("Failed to initialize ChromaDB client")

        try:
            # Try to get existing collection
            self._collection = self._chroma_client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
            logger.info(f"Collection count: {self._collection.count()}")
        except Exception:
            # Collection doesn't exist, will be created when building index
            logger.info(
                f"Collection '{self.collection_name}' does not exist yet. "
                "It will be created when building the index."
            )
            self._collection = None

        return self._collection

    def _load_cross_encoder(self) -> None:
        """Load cross-encoder model for reranking."""
        if self._cross_encoder is not None:
            return

        try:
            logger.info(f"Loading cross-encoder model: {self._cross_encoder}")
            self._cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device=self.device,
            )
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            raise RuntimeError(f"Cross-encoder loading failed: {e}") from e

    def build_index(self, nodes: list[TextNode]) -> None:
        """
        Build vector index from nodes and store in ChromaDB.

        Args:
            nodes: List of TextNode objects with embeddings

        Raises:
            RuntimeError: If index building fails
            ValueError: If nodes list is empty

        Example:
            >>> from llama_index.core.schema import TextNode
            >>> nodes = [TextNode(text="Sample text", metadata={"category": "aml"})]
            >>> retriever.build_index(nodes)
        """
        if not nodes:
            raise ValueError("Cannot build index from empty nodes list")

        logger.info(f"Building index from {len(nodes)} nodes...")

        try:
            # Initialize ChromaDB
            self._init_chromadb()
            if self._chroma_client is None:
                raise RuntimeError("ChromaDB client initialization failed")

            # Create or get collection
            try:
                self._collection = self._chroma_client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Using existing collection: {self.collection_name}")
            except Exception:
                # Create new collection
                self._collection = self._chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Financial fraud detection documents"},
                )
                logger.info(f"Created new collection: {self.collection_name}")

            # Initialize vector store
            self.vector_store = ChromaVectorStore(
                chroma_collection=self._collection
            )

            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            # Import HuggingFaceEmbedding for use in index
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            # Use HuggingFace embeddings with "local" mode for pre-embedded nodes
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                device="cpu",
            )

            # Build index with explicit embed model
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=embed_model,
            )

            logger.info(
                f"✓ Index built successfully with {len(nodes)} nodes. "
                f"Collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"Failed to build index: {e}", exc_info=True)
            raise RuntimeError(f"Index building failed: {e}") from e

    def load_index(self) -> bool:
        """
        Load existing index from ChromaDB.

        Returns:
            True if index loaded successfully, False otherwise

        Example:
            >>> success = retriever.load_index()
            >>> if success:
            ...     print("Index loaded successfully")
        """
        logger.info(f"Attempting to load index from {self.chroma_path}...")

        try:
            # Initialize ChromaDB
            self._init_chromadb()
            if self._chroma_client is None:
                logger.error("ChromaDB client initialization failed")
                return False

            # Get existing collection
            try:
                self._collection = self._chroma_client.get_collection(
                    name=self.collection_name
                )
            except Exception as e:
                logger.error(
                    f"Collection '{self.collection_name}' not found: {e}"
                )
                return False

            collection_count = self._collection.count()
            logger.info(f"Found collection with {collection_count} documents")

            if collection_count == 0:
                logger.warning("Collection exists but is empty")
                return False

            # Initialize vector store
            self.vector_store = ChromaVectorStore(
                chroma_collection=self._collection
            )

            # Create storage context and load index
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            # Import HuggingFaceEmbedding for use in index
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            # Use HuggingFace embeddings
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                device="cpu",
            )

            self.index = VectorStoreIndex.from_documents(
                documents=[],
                storage_context=storage_context,
                embed_model=embed_model,
            )

            logger.info(
                f"✓ Index loaded successfully. "
                f"Collection: {self.collection_name}, "
                f"Documents: {collection_count}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}", exc_info=True)
            return False

    def retrieve(
        self,
        query: str,
        doc_type_filter: str | list[str] | None = None,
        use_rerank: bool = True,
        similarity_threshold: float = 0.0,
    ) -> list[NodeWithScore]:
        """
        Retrieve relevant documents using hybrid search and optional reranking.

        Args:
            query: Search query string
            doc_type_filter: Optional filter by document category (e.g., 'aml', 'kyc')
            use_rerank: Whether to apply cross-encoder reranking
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of retrieved nodes with scores, sorted by relevance

        Raises:
            RuntimeError: If index is not loaded
            ValueError: If query is empty

        Example:
            >>> nodes = retriever.retrieve(
            ...     query="What are suspicious transaction reporting requirements?",
            ...     doc_type_filter="aml",
            ...     use_rerank=True
            ... )
            >>> print(f"Retrieved {len(nodes)} relevant chunks")
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if self.index is None:
            raise RuntimeError(
                "Index not loaded. Call load_index() or build_index() first."
            )

        logger.info(
            f"Retrieving for query: '{query[:100]}...' "
            f"(filter={doc_type_filter}, rerank={use_rerank})"
        )

        try:
            # Build search kwargs for metadata filtering
            search_kwargs = {"similarity_top_k": self.top_k}

            # Add metadata filter if specified
            if doc_type_filter:
                if isinstance(doc_type_filter, str):
                    filters = {"category": doc_type_filter}
                else:  # list of strings
                    filters = {"category": {"$in": doc_type_filter}}

                search_kwargs["filters"] = filters
                logger.debug(f"Applied metadata filter: {filters}")

            # Create retriever
            retriever = self.index.as_retriever(
                retrieve_mode="default",
                **search_kwargs,
            )

            # Retrieve nodes
            nodes = retriever.retrieve(query)

            # Filter by similarity threshold
            if similarity_threshold > 0:
                nodes = [
                    node for node in nodes
                    if node.score and node.score >= similarity_threshold
                ]
                logger.debug(
                    f"Filtered to {len(nodes)} nodes with score >= {similarity_threshold}"
                )

            if not nodes:
                logger.warning(f"No nodes retrieved for query: {query[:100]}")
                return []

            logger.info(f"Retrieved {len(nodes)} nodes from vector search")

            # Apply cross-encoder reranking if requested
            if use_rerank and len(nodes) > 1:
                nodes = self._rerank(query, nodes)
                logger.info(f"Reranked to top {len(nodes)} nodes")

            return nodes

        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            return []

    def _rerank(
        self,
        query: str,
        nodes: list[NodeWithScore],
    ) -> list[NodeWithScore]:
        """
        Rerank nodes using cross-encoder.

        Args:
            query: Original search query
            nodes: Retrieved nodes to rerank

        Returns:
            Reranked and truncated list of nodes
        """
        # Load cross-encoder if not already loaded
        if self._cross_encoder is None:
            self._load_cross_encoder()

        if self._cross_encoder is None:
            logger.warning("Cross-encoder not available, skipping rerank")
            return nodes[: self.rerank_top_n]

        try:
            # Prepare query-document pairs
            node_texts = [node.node.text for node in nodes]
            pairs = [[query, text] for text in node_texts]

            # Compute cross-encoder scores
            logger.debug(f"Reranking {len(pairs)} query-document pairs...")
            scores = self._cross_encoder.predict(pairs)

            # Update node scores and resort
            for node, score in zip(nodes, scores):
                node.score = float(score)

            # Sort by new scores (descending) and take top N
            reranked = sorted(
                nodes,
                key=lambda x: x.score if x.score else 0,
                reverse=True,
            )[: self.rerank_top_n]

            logger.debug(
                f"Reranking complete. Top scores: "
                f"{[n.score for n in reranked[:3]]}"
            )

            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            # Return original nodes truncated to rerank_top_n
            return nodes[: self.rerank_top_n]

    def format_context(
        self,
        nodes: list[NodeWithScore],
        include_scores: bool = True,
        max_context_length: int | None = None,
    ) -> str:
        """
        Format retrieved nodes into context string with citations.

        Args:
            nodes: Retrieved nodes to format
            include_scores: Whether to include relevance scores
            max_context_length: Maximum total length in characters

        Returns:
            Formatted context string with source citations

        Example:
            >>> context = retriever.format_context(nodes)
            >>> print(context)
            [Source: aml_policy.pdf | Category: aml | Score: 0.92]
            Suspicious Activity Reports must be filed within...
        """
        if not nodes:
            return "No relevant context found."

        formatted_parts = []

        for i, node_with_score in enumerate(nodes, start=1):
            node = node_with_score.node
            metadata = node.metadata
            text = node.text or ""

            # Extract citation information
            source = metadata.get("file_name", "Unknown Source")
            category = metadata.get("category", "general")
            score = node_with_score.score

            # Build citation header
            citation = f"[{i}. {source} | Category: {category}"

            if include_scores and score is not None:
                citation += f" | Score: {score:.3f}"

            citation += "]"

            # Format the chunk
            formatted_chunk = f"{citation}\n{text}\n"
            formatted_parts.append(formatted_chunk)

        # Join all chunks
        context = "\n".join(formatted_parts)

        # Truncate if necessary
        if max_context_length and len(context) > max_context_length:
            context = context[:max_context_length]
            context += "\n\n[Context truncated due to length limit]"

        logger.debug(f"Formatted context: {len(context)} characters from {len(nodes)} nodes")

        return context

    def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the ChromaDB collection.

        Returns:
            Dictionary containing collection statistics

        Example:
            >>> stats = retriever.get_collection_stats()
            >>> print(f"Total documents: {stats['total_docs']}")
        """
        try:
            if self._collection is None:
                self._get_or_create_collection()

            if self._collection is None:
                return {
                    "collection_name": self.collection_name,
                    "status": "not_created",
                    "total_docs": 0,
                }

            total_docs = self._collection.count()

            # Get metadata statistics
            stats = {
                "collection_name": self.collection_name,
                "status": "loaded" if self.index else "created",
                "total_docs": total_docs,
                "chroma_path": str(self.chroma_path),
            }

            logger.debug(f"Collection stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "status": "error",
                "error": str(e),
            }

    def delete_index(self) -> bool:
        """
        Delete the collection from ChromaDB.

        Returns:
            True if deletion was successful

        Warning:
            This operation is irreversible. All data in the collection will be lost.

        Example:
            >>> if retriever.delete_index():
            ...     print("Index deleted successfully")
        """
        logger.warning(f"Attempting to delete collection: {self.collection_name}")

        try:
            if self._chroma_client is None:
                self._init_chromadb()

            if self._chroma_client is None:
                return False

            self._chroma_client.delete_collection(self.collection_name)

            # Reset instance variables
            self._collection = None
            self.vector_store = None
            self.index = None

            logger.info(f"✓ Collection '{self.collection_name}' deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}", exc_info=True)
            return False


def main() -> None:
    """
    Demonstration of HybridRetriever usage.

    This main block demonstrates:
    - Creating sample nodes with metadata
    - Building and loading an index
    - Retrieving with and without filters
    - Reranking results
    - Formatting context with citations
    """
    import sys

    from llama_index.core.schema import TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("FraudDocs-RAG Hybrid Retriever Demonstration")
    logger.info("=" * 70)

    try:
        # Initialize embedding model
        logger.info("Loading embedding model...")
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            device="cpu",
        )

        # Initialize retriever
        logger.info("Initializing HybridRetriever...")
        retriever = HybridRetriever(
            collection_name="demo_financial_docs",
            chroma_path="./data/chroma_db_demo",
            top_k=10,
            rerank_top_n=5,
            device="cpu",
        )
        logger.info("✓ Retriever initialized\n")

        # Example 1: Create sample nodes with embeddings
        logger.info("Example 1: Creating sample nodes with embeddings")
        logger.info("-" * 70)

        sample_nodes = [
            TextNode(
                text=(
                    "Suspicious Activity Reports (SAR) must be filed within 30 days "
                    "of detecting suspicious transaction activity. Financial institutions "
                    "are required to report any transactions that appear suspicious, "
                    "involve illegal funds, or have no business purpose."
                ),
                metadata={
                    "file_name": "aml_sar_requirements.pdf",
                    "category": "aml",
                    "title": "SAR Filing Requirements",
                }
            ),
            TextNode(
                text=(
                    "Customer Due Diligence (CDD) requires verification of customer "
                    "identity. Enhanced Due Diligence (EDD) must be applied for high-risk "
                    "customers, including politically exposed persons (PEPs) and customers "
                    "from high-risk jurisdictions."
                ),
                metadata={
                    "file_name": "kyc_procedures.pdf",
                    "category": "kyc",
                    "title": "KYC and Due Diligence Procedures",
                }
            ),
            TextNode(
                text=(
                    "Fraud detection systems must monitor for unusual transaction patterns, "
                    "including rapid movement of funds, structuring to avoid reporting "
                    "thresholds, and transactions with high-risk jurisdictions. "
                    "Alerts must be investigated within 24 hours."
                ),
                metadata={
                    "file_name": "fraud_detection_guide.pdf",
                    "category": "fraud",
                    "title": "Fraud Detection and Investigation",
                }
            ),
            TextNode(
                text=(
                    "Financial institutions must comply with regulatory requirements "
                    "including the Bank Secrecy Act (BSA), USA PATRIOT Act, and OFAC "
                    "regulations. Regular compliance training is mandatory for all staff."
                ),
                metadata={
                    "file_name": "compliance_regulations.pdf",
                    "category": "regulation",
                    "title": "Regulatory Compliance Framework",
                }
            ),
            TextNode(
                text=(
                    "Anti-money laundering programs must include: written policies, "
                    "a designated compliance officer, employee training, and independent "
                    "testing. The program must be approved by the board of directors."
                ),
                metadata={
                    "file_name": "aml_program_manual.pdf",
                    "category": "aml",
                    "title": "AML Program Requirements",
                }
            ),
        ]

        # Add embeddings to nodes
        for node in sample_nodes:
            embedding = embed_model.get_text_embedding(node.text)
            node.embedding = embedding

        logger.info(f"Created {len(sample_nodes)} sample nodes with embeddings\n")

        # Example 2: Build index
        logger.info("Example 2: Building vector index")
        logger.info("-" * 70)
        retriever.build_index(sample_nodes)

        stats = retriever.get_collection_stats()
        logger.info(f"Collection stats: {stats}\n")

        # Example 3: Basic retrieval without filters
        logger.info("Example 3: Basic retrieval (no filter)")
        logger.info("-" * 70)
        query1 = "What are the reporting requirements for suspicious transactions?"
        nodes1 = retriever.retrieve(query1, use_rerank=True)

        logger.info(f"\nQuery: {query1}")
        logger.info(f"Retrieved {len(nodes1)} nodes")
        for i, node in enumerate(nodes1[:3], start=1):
            logger.info(f"  {i}. [{node.node.metadata.get('category')}] {node.score:.3f}")
        print()

        # Example 4: Filtered retrieval by category
        logger.info("Example 4: Filtered retrieval (AML only)")
        logger.info("-" * 70)
        query2 = "What are the compliance requirements?"
        nodes2 = retriever.retrieve(query2, doc_type_filter="aml", use_rerank=True)

        logger.info(f"\nQuery: {query2}")
        logger.info(f"Filter: category='aml'")
        logger.info(f"Retrieved {len(nodes2)} nodes")
        for i, node in enumerate(nodes2, start=1):
            logger.info(
                f"  {i}. [{node.node.metadata.get('file_name')}] "
                f"{node.score:.3f}"
            )
        print()

        # Example 5: Multi-category filter
        logger.info("Example 5: Multi-category filter (AML or KYC)")
        logger.info("-" * 70)
        query3 = "What customer verification procedures are required?"
        nodes3 = retriever.retrieve(
            query3,
            doc_type_filter=["aml", "kyc"],
            use_rerank=True
        )

        logger.info(f"\nQuery: {query3}")
        logger.info(f"Filter: category in ['aml', 'kyc']")
        logger.info(f"Retrieved {len(nodes3)} nodes")
        for i, node in enumerate(nodes3, start=1):
            logger.info(
                f"  {i}. [{node.node.metadata.get('category')}] "
                f"{node.score:.3f}"
            )
        print()

        # Example 6: Format context with citations
        logger.info("Example 6: Format context with citations")
        logger.info("-" * 70)
        context = retriever.format_context(nodes2, include_scores=True)
        logger.info("\nFormatted Context:\n")
        logger.info(context)
        print()

        # Example 7: Comparison with and without reranking
        logger.info("Example 7: Impact of reranking")
        logger.info("-" * 70)
        query4 = "suspicious activity and fraud detection procedures"

        nodes_no_rerank = retriever.retrieve(query4, use_rerank=False)
        nodes_with_rerank = retriever.retrieve(query4, use_rerank=True)

        logger.info(f"\nQuery: {query4}\n")
        logger.info("Without reranking:")
        for i, node in enumerate(nodes_no_rerank[:3], start=1):
            logger.info(f"  {i}. [{node.node.metadata.get('category')}] {node.score:.3f}")

        logger.info("\nWith reranking:")
        for i, node in enumerate(nodes_with_rerank[:3], start=1):
            logger.info(f"  {i}. [{node.node.metadata.get('category')}] {node.score:.3f}")
        print()

        logger.info("=" * 70)
        logger.info("Demonstration complete!")

        # Cleanup: Delete demo collection
        import shutil
        demo_chroma_path = Path("./data/chroma_db_demo")
        if demo_chroma_path.exists():
            retriever.delete_index()
            shutil.rmtree(demo_chroma_path)
            logger.info(f"Cleaned up demo data")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
