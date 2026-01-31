"""
Document Processor Module for FraudDocs-RAG.

This module handles document ingestion, including:
- Loading documents from multiple file formats (PDF, DOCX, TXT, HTML)
- Semantic chunking using LlamaIndex's SemanticSplitterNodeParser
- Document classification into fraud-related categories
- Metadata extraction and enrichment
- Document deduplication using content hashing
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import Document, BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)


class DocumentCategory(str, Enum):
    """Document categories for financial fraud detection."""

    AML = "aml"
    KYC = "kyc"
    FRAUD = "fraud"
    REGULATION = "regulation"
    GENERAL = "general"


class DocumentType(str, Enum):
    """Supported document file types."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"


# Classification keywords for each category
CLASSIFICATION_KEYWORDS = {
    DocumentCategory.AML: [
        "anti-money laundering",
        "aml",
        "suspicious activity",
        "sar",
        "suspicious activity report",
        "currency transaction report",
        "ctr",
        "fintrac",
        "financial crimes",
        "money laundering",
        "terrorist financing",
        "bsa",
        "bank secrecy act",
    ],
    DocumentCategory.KYC: [
        "know your customer",
        "kyc",
        "identity verification",
        "customer due diligence",
        "cdd",
        "enhanced due diligence",
        "edd",
        "beneficial owner",
        "ultimate beneficial owner",
        "ubo",
        "customer identification",
        "client onboarding",
    ],
    DocumentCategory.FRAUD: [
        "fraud",
        "fraudulent",
        "suspicious transaction",
        "fraud detection",
        "fraud prevention",
        "fraud alert",
        "fraud investigation",
        "unauthorized transaction",
        "account takeover",
        "phishing",
        "social engineering",
        "embezzlement",
        "internal fraud",
        "external fraud",
    ],
    DocumentCategory.REGULATION: [
        "regulation",
        "regulatory",
        "compliance",
        "regulatory requirement",
        "compliance program",
        "guideline",
        "standard",
        "policy",
        "procedure",
        "office of foreign assets",
        "ofac",
        "financial conduct authority",
        "fca",
        "sec",
        "securities and exchange",
    ],
}


class DocumentProcessor:
    """
    Process documents for ingestion into the RAG system.

    Handles document loading, classification, semantic chunking,
    metadata extraction, and deduplication.

    Attributes:
        embedding_model: HuggingFace embedding model for semantic chunking
        chunker: SemanticSplitterNodeParser for intelligent chunking
        processed_hashes: Set of content hashes for deduplication

    Example:
        >>> processor = DocumentProcessor(embed_model_name="BAAI/bge-small-en-v1.5")
        >>> nodes = processor.process_directory("data/documents/")
        >>> print(f"Processed {len(nodes)} chunks from {len(processor.processed_hashes)} documents")
    """

    def __init__(
        self,
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        buffer_size: int = 1,
        breakpoint_threshold: int = 60,
    ) -> None:
        """
        Initialize the document processor.

        Args:
            embed_model_name: Name of HuggingFace embedding model
            buffer_size: Number of sentences to group for embeddings
            breakpoint_threshold: Percentile similarity threshold for splits

        Raises:
            ValueError: If embed_model_name is empty
            RuntimeError: If embedding model fails to load
        """
        if not embed_model_name:
            raise ValueError("embed_model_name cannot be empty")

        try:
            self.embedding_model = HuggingFaceEmbedding(
                model_name=embed_model_name,
                device="cpu",
            )
            logger.info(f"Loaded embedding model: {embed_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to initialize embedding model: {e}") from e

        try:
            self.chunker = SemanticSplitterNodeParser(
                buffer_size=buffer_size,
                breakpoint_percentile_threshold=breakpoint_threshold,
                embed_model=self.embedding_model,
            )
            logger.info(
                f"Initialized SemanticSplitterNodeParser with buffer_size={buffer_size}, "
                f"breakpoint_threshold={breakpoint_threshold}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize chunker: {e}")
            raise RuntimeError(f"Failed to initialize semantic chunker: {e}") from e

        self.processed_hashes: set[str] = set()
        self._classification_cache: dict[str, DocumentCategory] = {}

    def calculate_content_hash(self, content: str) -> str:
        """
        Calculate SHA-256 hash of document content for deduplication.

        Args:
            content: Document text content

        Returns:
            Hexadecimal SHA-256 hash string

        Example:
            >>> hash_val = processor.calculate_content_hash("Sample document text")
            >>> print(hash_val)
            'a4ae1f3a4b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7'
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def classify_document(self, content: str, metadata: dict[str, Any] | None = None) -> DocumentCategory:
        """
        Classify document into fraud-related categories using keyword matching.

        Classification priority: FRAUD > AML > KYC > REGULATION > GENERAL
        (More specific categories take precedence)

        Args:
            content: Document text content
            metadata: Optional metadata to aid classification

        Returns:
            DocumentCategory enum value

        Example:
            >>> category = processor.classify_document("This SAR form is for suspicious...")
            >>> print(category)
            <DocumentCategory.AML: 'aml'>
        """
        # Check cache first
        content_hash = self.calculate_content_hash(content[:1000])  # Hash first 1000 chars
        if content_hash in self._classification_cache:
            return self._classification_cache[content_hash]

        content_lower = content.lower()

        # Count keyword matches for each category
        category_scores: dict[DocumentCategory, int] = {
            DocumentCategory.AML: 0,
            DocumentCategory.KYC: 0,
            DocumentCategory.FRAUD: 0,
            DocumentCategory.REGULATION: 0,
            DocumentCategory.GENERAL: 0,
        }

        for category, keywords in CLASSIFICATION_KEYWORDS.items():
            for keyword in keywords:
                # Count occurrences of each keyword
                count = content_lower.count(keyword.lower())
                category_scores[category] += count

        # Determine category with highest score (with priority tie-breaking)
        # Priority: FRAUD > AML > KYC > REGULATION > GENERAL
        priority_order = [
            DocumentCategory.FRAUD,
            DocumentCategory.AML,
            DocumentCategory.KYC,
            DocumentCategory.REGULATION,
            DocumentCategory.GENERAL,
        ]

        best_category = DocumentCategory.GENERAL
        best_score = -1

        for category in priority_order:
            if category_scores[category] > best_score:
                best_score = category_scores[category]
                best_category = category

        # Cache the result
        self._classification_cache[content_hash] = best_category

        logger.debug(
            f"Document classified as {best_category.value} "
            f"(scores: {category_scores})"
        )

        return best_category

    def extract_metadata(
        self,
        file_path: Path,
        content: str,
        doc_type: DocumentType,
    ) -> dict[str, Any]:
        """
        Extract and enrich metadata for a document.

        Args:
            file_path: Path to the source file
            content: Document text content
            doc_type: Type of document (PDF, DOCX, etc.)

        Returns:
            Dictionary containing extracted metadata

        Example:
            >>> metadata = processor.extract_metadata(
            ...     Path("doc.pdf"),
            ...     "Document content...",
            ...     DocumentType.PDF
            ... )
            >>> print(metadata['category'])
            'fraud'
        """
        category = self.classify_document(content)

        metadata = {
            "file_name": file_path.name,
            "file_path": str(file_path.absolute()),
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "document_type": doc_type.value,
            "category": category.value,
            "ingestion_date": datetime.now().isoformat(),
            "content_hash": self.calculate_content_hash(content),
            "char_count": len(content),
            "word_count": len(content.split()),
            "title": file_path.stem,
        }

        return metadata

    def add_contextual_header(self, node: BaseNode, metadata: dict[str, Any]) -> str:
        """
        Add contextual header information to a chunk.

        The header includes document title, category, and source information
        to provide context for LLM retrieval.

        Args:
            node: LlamaIndex node containing the chunk
            metadata: Document metadata

        Returns:
            Text with contextual header prepended

        Example:
            >>> contextual_text = processor.add_contextual_header(node, metadata)
            >>> print(contextual_text[:100])
            '[Document: AML Policy | Category: aml | Source: /path/to/aml_policy.pdf]\\n\\nOriginal chunk text...'
        """
        header = (
            f"[Document: {metadata.get('title', 'Unknown')} | "
            f"Category: {metadata.get('category', 'general')} | "
            f"Source: {metadata.get('file_name', 'Unknown')}]\n\n"
        )

        # Store original text for reference
        original_text = node.text or ""

        return header + original_text

    def load_document(
        self,
        file_path: Path | str,
    ) -> list[Document] | None:
        """
        Load a single document from file.

        Supports PDF, DOCX, TXT, and HTML formats.

        Args:
            file_path: Path to the document file

        Returns:
            List of LlamaIndex Document objects, or None if loading fails

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported

        Example:
            >>> docs = processor.load_document("data/documents/policy.pdf")
            >>> print(f"Loaded {len(docs)} document(s)")
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Document file not found: {file_path}")

        # Get file extension
        suffix = file_path.suffix.lower().lstrip(".")

        # Map extension to DocumentType
        extension_to_type = {
            "pdf": DocumentType.PDF,
            "docx": DocumentType.DOCX,
            "doc": DocumentType.DOCX,
            "txt": DocumentType.TXT,
            "html": DocumentType.HTML,
            "htm": DocumentType.HTML,
        }

        if suffix not in extension_to_type:
            logger.error(f"Unsupported file format: {suffix}")
            raise ValueError(
                f"Unsupported file format: .{suffix}. "
                f"Supported formats: {', '.join(extension_to_type.keys())}"
            )

        doc_type = extension_to_type[suffix]

        try:
            # Use LlamaIndex's SimpleDirectoryReader for single file
            reader = SimpleDirectoryReader(
                input_files=[str(file_path)],
                recursive=False,
            )

            docs = reader.load_data()

            if not docs:
                logger.warning(f"No content extracted from file: {file_path}")
                return None

            # Enrich documents with metadata
            for doc in docs:
                metadata = self.extract_metadata(file_path, doc.text or "", doc_type)
                doc.metadata.update(metadata)

                # Add category as a separate metadata field for easy filtering
                doc.metadata["doc_category"] = metadata["category"]

            logger.info(f"Successfully loaded document: {file_path}")
            return docs

        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}", exc_info=True)
            return None

    def process_document(
        self,
        file_path: Path | str,
        add_context: bool = True,
    ) -> list[BaseNode] | None:
        """
        Process a single document: load, deduplicate, classify, and chunk.

        Args:
            file_path: Path to the document file
            add_context: Whether to add contextual headers to chunks

        Returns:
            List of chunked nodes, or None if processing fails

        Example:
            >>> nodes = processor.process_document("data/documents/fraud_policy.pdf")
            >>> print(f"Created {len(nodes)} chunks")
        """
        file_path = Path(file_path)

        # Load document
        documents = self.load_document(file_path)

        if not documents:
            return None

        processed_nodes = []

        for doc in documents:
            # Check for duplicates
            content_hash = doc.metadata.get("content_hash", "")

            if content_hash in self.processed_hashes:
                logger.info(f"Duplicate document detected (hash: {content_hash}): {file_path}")
                continue

            # Mark as processed
            self.processed_hashes.add(content_hash)

            # Perform semantic chunking
            try:
                nodes = self.chunker.get_nodes_from_documents([doc])

                # Add contextual headers if requested
                if add_context:
                    for node in nodes:
                        contextual_text = self.add_contextual_header(node, doc.metadata)
                        node.text = contextual_text
                        # Add metadata to node
                        node.metadata.update(doc.metadata)

                processed_nodes.extend(nodes)
                logger.info(
                    f"Processed {file_path.name}: "
                    f"created {len(nodes)} semantic chunks, "
                    f"category={doc.metadata.get('category', 'general')}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to chunk document {file_path}: {e}",
                    exc_info=True
                )
                continue

        return processed_nodes if processed_nodes else None

    def process_directory(
        self,
        directory_path: Path | str,
        recursive: bool = True,
        add_context: bool = True,
    ) -> list[BaseNode]:
        """
        Process all supported documents in a directory.

        Args:
            directory_path: Path to the directory containing documents
            recursive: Whether to recursively process subdirectories
            add_context: Whether to add contextual headers to chunks

        Returns:
            List of all chunked nodes from all documents

        Example:
            >>> nodes = processor.process_directory("data/documents/")
            >>> print(f"Processed {len(nodes)} total chunks")
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not directory_path.is_dir():
            logger.error(f"Path is not a directory: {directory_path}")
            raise ValueError(f"Path is not a directory: {directory_path}")

        logger.info(f"Processing directory: {directory_path} (recursive={recursive})")

        # Find all supported files
        supported_extensions = {".pdf", ".docx", ".doc", ".txt", ".html", ".htm"}

        if recursive:
            files = [
                f for f in directory_path.rglob("*")
                if f.is_file() and f.suffix.lower() in supported_extensions
            ]
        else:
            files = [
                f for f in directory_path.glob("*")
                if f.is_file() and f.suffix.lower() in supported_extensions
            ]

        if not files:
            logger.warning(f"No supported files found in directory: {directory_path}")
            return []

        logger.info(f"Found {len(files)} files to process")

        all_nodes = []
        processing_stats = {
            "total": len(files),
            "successful": 0,
            "failed": 0,
            "duplicates": 0,
            "categories": {
                "aml": 0,
                "kyc": 0,
                "fraud": 0,
                "regulation": 0,
                "general": 0,
            }
        }

        for file_path in files:
            try:
                nodes = self.process_document(file_path, add_context=add_context)

                if nodes:
                    all_nodes.extend(nodes)
                    processing_stats["successful"] += 1

                    # Update category counts from metadata
                    if nodes and nodes[0].metadata.get("category"):
                        category = nodes[0].metadata["category"]
                        if category in processing_stats["categories"]:
                            processing_stats["categories"][category] += 1

                else:
                    processing_stats["failed"] += 1

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
                processing_stats["failed"] += 1
                continue

        # Log statistics
        logger.info(
            f"Directory processing complete:\n"
            f"  Total files: {processing_stats['total']}\n"
            f"  Successful: {processing_stats['successful']}\n"
            f"  Failed: {processing_stats['failed']}\n"
            f"  Total chunks: {len(all_nodes)}\n"
            f"  Categories: {processing_stats['categories']}\n"
            f"  Duplicates skipped: {len(self.processed_hashes) - processing_stats['successful']}"
        )

        return all_nodes

    def get_statistics(self) -> dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary containing processing statistics

        Example:
            >>> stats = processor.get_statistics()
            >>> print(f"Processed {stats['documents_processed']} documents")
        """
        return {
            "documents_processed": len(self.processed_hashes),
            "cached_classifications": len(self._classification_cache),
            "embedding_model": self.embedding_model.model_name,
        }


def main() -> None:
    """
    Demonstration of DocumentProcessor usage.

    This main block shows how to use the DocumentProcessor for:
    - Processing individual documents
    - Processing entire directories
    - Viewing processing statistics
    """
    import sys

    # Configure logging for demonstration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("FraudDocs-RAG Document Processor Demonstration")
    logger.info("=" * 60)

    try:
        # Initialize processor
        logger.info("Initializing DocumentProcessor...")
        processor = DocumentProcessor(
            embed_model_name="BAAI/bge-small-en-v1.5",
            buffer_size=1,
            breakpoint_threshold=60,
        )
        logger.info("✓ Processor initialized successfully\n")

        # Example 1: Create sample documents for demonstration
        logger.info("Creating sample documents for demonstration...")
        demo_dir = Path("data/demo_documents")
        demo_dir.mkdir(parents=True, exist_ok=True)

        # Sample AML document
        aml_doc = demo_dir / "aml_policy.txt"
        aml_doc.write_text(
            "Anti-Money Laundering Policy\n\n"
            "This document outlines the procedures for detecting and reporting "
            "suspicious activity. All staff must complete Suspicious Activity Reports (SAR) "
            "when they detect potential money laundering or terrorist financing. "
            "Currency Transaction Reports (CTR) must be filed for transactions over $10,000. "
            "The Bank Secrecy Act (BSA) requires all financial institutions to maintain "
            "proper records and report suspicious activities to FinCEN."
        )

        # Sample Fraud document
        fraud_doc = demo_dir / "fraud_detection_guide.txt"
        fraud_doc.write_text(
            "Fraud Detection and Investigation Guide\n\n"
            "This guide provides procedures for fraud detection and prevention. "
            "When a fraud alert is triggered, investigators must conduct a thorough "
            "fraud investigation. Common indicators include unauthorized transactions, "
            "account takeover attempts, and phishing incidents. "
            "Internal fraud and external fraud require different investigative approaches. "
            "All fraudulent activity must be documented and reported to the fraud prevention team."
        )

        # Sample KYC document
        kyc_doc = demo_dir / "kyc_procedures.txt"
        kyc_doc.write_text(
            "Know Your Customer (KYC) Procedures\n\n"
            "This document describes the customer identification and due diligence procedures. "
            "All new customers must complete the Know Your Customer (KYC) process. "
            "Customer Due Diligence (CDD) includes identity verification and risk assessment. "
            "Enhanced Due Diligence (EDD) is required for high-risk customers. "
            "Beneficial ownership information must be collected for all entity accounts. "
            "The Ultimate Beneficial Owner (UBO) must be identified and verified."
        )

        logger.info(f"✓ Created 3 sample documents in {demo_dir}\n")

        # Example 2: Process individual document
        logger.info("Example 1: Processing a single document")
        logger.info("-" * 60)
        nodes = processor.process_document(aml_doc, add_context=True)

        if nodes:
            logger.info(f"✓ Processed document into {len(nodes)} chunks")
            logger.info(f"  Document Category: {nodes[0].metadata.get('category')}")
            logger.info(f"  Content Hash: {nodes[0].metadata.get('content_hash')[:16]}...")
            if len(nodes) > 0:
                logger.info(f"  First chunk preview (first 150 chars):")
                logger.info(f"    {nodes[0].text[:150]}...")
        print()

        # Example 3: Process entire directory
        logger.info("Example 2: Processing entire directory")
        logger.info("-" * 60)
        all_nodes = processor.process_directory(demo_dir, recursive=False)

        if all_nodes:
            logger.info(f"✓ Processed directory into {len(all_nodes)} total chunks")

            # Show breakdown by category
            category_counts = {}
            for node in all_nodes:
                cat = node.metadata.get("category", "general")
                category_counts[cat] = category_counts.get(cat, 0) + 1

            logger.info("  Chunk breakdown by category:")
            for category, count in sorted(category_counts.items()):
                logger.info(f"    {category}: {count} chunks")
        print()

        # Example 4: Show statistics
        logger.info("Example 3: Processing Statistics")
        logger.info("-" * 60)
        stats = processor.get_statistics()
        logger.info(f"Documents processed: {stats['documents_processed']}")
        logger.info(f"Classifications cached: {stats['cached_classifications']}")
        logger.info(f"Embedding model: {stats['embedding_model']}")
        print()

        # Example 5: Test deduplication
        logger.info("Example 4: Testing Deduplication")
        logger.info("-" * 60)
        logger.info("Processing the same document again (should be detected as duplicate)...")
        duplicate_nodes = processor.process_document(aml_doc, add_context=True)
        if duplicate_nodes is None:
            logger.info("✓ Duplicate correctly detected and skipped!")
        print()

        logger.info("=" * 60)
        logger.info("Demonstration complete!")

        # Cleanup demo files
        import shutil
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
            logger.info(f"Cleaned up demo directory: {demo_dir}")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
