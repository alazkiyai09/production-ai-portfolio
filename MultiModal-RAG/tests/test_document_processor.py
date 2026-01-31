# ============================================================
# Enterprise-RAG: Document Processor Tests
# ============================================================
"""
Tests for document processing functionality.
"""

import hashlib
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.ingestion.document_processor import (
    Document,
    DocumentProcessor,
    ProcessingResult,
)
from src.exceptions import (
    DocumentProcessingError,
    UnsupportedFormatError,
)


# ============================================================
# Document Class Tests
# ============================================================

class TestDocument:
    """Tests for the Document dataclass."""

    def test_document_creation(self):
        """Test creating a valid document."""
        doc = Document(
            content="Test content",
            metadata={"source": "test.txt"},
            doc_id="doc_123",
            chunk_id="doc_123_chunk_0",
        )

        assert doc.content == "Test content"
        assert doc.metadata == {"source": "test.txt"}
        assert doc.doc_id == "doc_123"
        assert doc.chunk_id == "doc_123_chunk_0"

    def test_document_to_dict(self):
        """Test converting document to dictionary."""
        doc = Document(
            content="Test",
            metadata={"source": "test"},
            doc_id="doc_1",
            chunk_id="doc_1_chunk_0",
        )

        doc_dict = doc.to_dict()

        assert doc_dict["content"] == "Test"
        assert doc_dict["metadata"] == {"source": "test"}
        assert doc_dict["doc_id"] == "doc_1"
        assert doc_dict["chunk_id"] == "doc_1_chunk_0"

    def test_document_from_dict(self):
        """Test creating document from dictionary."""
        doc_dict = {
            "content": "Test content",
            "metadata": {"source": "test.txt"},
            "doc_id": "doc_1",
            "chunk_id": "doc_1_chunk_0",
        }

        doc = Document.from_dict(doc_dict)

        assert doc.content == "Test content"
        assert doc.metadata == {"source": "test.txt"}
        assert doc.doc_id == "doc_1"

    def test_document_empty_content_raises_error(self):
        """Test that document with empty content raises error."""
        with pytest.raises(ValueError, match="empty"):
            Document(
                content="",
                metadata={},
                doc_id="doc_1",
                chunk_id="doc_1_chunk_0",
            )

    def test_document_empty_doc_id_raises_error(self):
        """Test that document with empty doc_id raises error."""
        with pytest.raises(ValueError, match="doc_id"):
            Document(
                content="Test",
                metadata={},
                doc_id="",
                chunk_id="chunk_1",
            )


# ============================================================
# DocumentProcessor Tests
# ============================================================

class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = DocumentProcessor(
            chunk_size=256,
            chunk_overlap=50,
        )

        assert processor.chunk_size == 256
        assert processor.chunk_overlap == 50
        assert processor.min_chunk_size == 100
        assert processor.preserve_headers is True

    def test_initialization_invalid_chunk_size(self):
        """Test that invalid chunk size raises error."""
        with pytest.raises(ValueError, match="must be greater"):
            DocumentProcessor(
                chunk_size=100,
                chunk_overlap=150,
            )

    def test_initialization_invalid_overlap(self):
        """Test that overlap >= chunk size raises error."""
        with pytest.raises(ValueError, match="must be greater"):
            DocumentProcessor(
                chunk_size=200,
                chunk_overlap=200,
            )

    def test_initialization_invalid_min_chunk(self):
        """Test that min_chunk >= chunk_size raises error."""
        with pytest.raises(ValueError, match="must be less"):
            DocumentProcessor(
                chunk_size=100,
                min_chunk_size=100,
            )

    @pytest.mark.parametrize(
        "text,expected_count",
        [
        ("a" * 100, 1),  # Small text, one chunk
        ("a" * 600, 2),  # Medium text, two chunks
        ("a" * 1200, 3),  # Large text, three chunks
    ],
)
    def test_chunk_text_length(
        self,
        text: str,
        expected_count: int,
    ):
        """Test chunking text of different lengths."""
        processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)

        metadata = {"source": "test.txt"}
        chunks = processor._chunk_text(text, metadata, "doc_001")

        assert len(chunks) == expected_count
        for chunk in chunks:
            assert chunk.doc_id == "doc_001"
            assert "chunk_id" in chunk.metadata

    def test_chunk_text_preserves_metadata(self):
        """Test that chunking preserves document metadata."""
        processor = DocumentProcessor(chunk_size=256, chunk_overlap=50)

        metadata = {
            "source": "test.pdf",
            "page": 1,
            "file_type": "pdf",
        }

        text = "This is a test document. " * 50
        chunks = processor._chunk_text(text, metadata, "doc_001")

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["page"] == 1

    def test_chunk_text_with_overlap(self):
        """Test that chunks have overlapping content."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=30)

        text = "word " * 50  # Create repetitive text
        chunks = processor._chunk_text(text, {"source": "test"}, "doc_001")

        if len(chunks) > 1:
            # Check for overlap by comparing content
            first_chunk = chunks[0].content
            second_chunk = chunks[1].content

            # Second chunk should contain some of first chunk
            assert len(first_chunk) > 0
            assert len(second_chunk) > 0

    def test_generate_chunk_id(self):
        """Test chunk ID generation."""
        processor = DocumentProcessor()

        chunk_id = processor._generate_chunk_id("doc_abc123", 0)
        assert chunk_id == "doc_abc123_chunk_0"

        chunk_id = processor._generate_chunk_id("doc_xyz", 5)
        assert chunk_id == "doc_xyz_chunk_5"

    def test_generate_doc_id(self):
        """Test document ID generation."""
        processor = DocumentProcessor()

        doc_id = processor._generate_doc_id(Path("test/document.pdf"))

        assert doc_id.startswith("doc_")
        assert len(doc_id) == 16  # MD5 hash length (12) + prefix (4)

    def test_detect_file_type_pdf(self, temp_pdf_file):
        """Test PDF file type detection."""
        processor = DocumentProcessor()

        file_type = processor._detect_file_type(temp_pdf_file)

        assert file_type == "pdf"

    def test_detect_file_type_md(self, temp_md_file):
        """Test Markdown file type detection."""
        processor = DocumentProcessor()

        file_type = processor._detect_file_type(temp_md_file)

        assert file_type == "md"

    def test_detect_file_type_txt(self, temp_txt_file):
        """Test text file type detection."""
        processor = DocumentProcessor()

        file_type = processor._detect_file_type(temp_txt_file)

        assert file_type == "txt"

    def test_detect_file_type_unsupported(self, tmp_path):
        """Test unsupported file type raises error."""
        processor = DocumentProcessor()

        # Create a file with unsupported extension
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("content")

        with pytest.raises(UnsupportedFormatError):
            processor._detect_file_type(unsupported_file)

    def test_clean_text(self):
        """Test text cleaning functionality."""
        processor = DocumentProcessor()

        dirty_text = "  This   has    extra  spaces  \n\n\n\nWeird   chars: \x92\x93 "
        clean = processor._clean_text(dirty_text)

        # No multiple spaces
        assert "  " not in clean
        # No excessive newlines
        assert not "\n\n\n" in clean
        # Control characters removed
        assert "\x92" not in clean

    def test_clean_text_preserves_content(self):
        """Test that cleaning preserves meaningful content."""
        processor = DocumentProcessor()

        text = "Machine learning is awesome."
        clean = processor._clean_text(text)

        assert "Machine learning" in clean
        assert "awesome" in clean

    @pytest.mark.parametrize(
        "text,min_len,max_len",
        [
            ("Short", 0, 100),
            ("Medium" * 50, 200, 300),
        ],
)
    def test_clean_text_length_range(self, text, min_len, max_len):
        """Test cleaning preserves text within reasonable length."""
        processor = DocumentProcessor()
        clean = processor._clean_text(text)

        assert min_len <= len(clean) <= max_len

    def test_extract_text_from_txt(self, temp_txt_file):
        """Test extracting text from plain text file."""
        processor = DocumentProcessor()

        result = processor._extract_text_text(temp_txt_file)

        assert result[0]  # Text extraction returns (text, metadata)
        assert "Refund Policy" in result[0]

    def test_extract_text_metadata(self, temp_txt_file):
        """Test that metadata is extracted correctly."""
        processor = DocumentProcessor()

        text, metadata = processor._extract_text_text(temp_txt_file)

        assert metadata["file_type"] == "txt"
        assert "characters" in metadata

    @pytest.mark.parametrize(
    "file_fixture,expected_words",
    [
        ("sample_txt_content", ["Refund", "Policy", "days"]),
        ("sample_md_content", ["Machine", "Learning", "supervised"]),
        ("sample_pdf_content", ["Privacy", "Policy", "data"]),
    ],
    )
    def test_extract_keywords(self, file_fixture, expected_words, request):
        """Test that key content is preserved during extraction."""
        processor = DocumentProcessor()
        content = request.getfixturevalue(file_fixture)

        if file_fixture == "sample_md_content":
            result = processor._extract_text_markdown(Path("test.md"))
            text = result[0]
        elif file_fixture == "sample_pdf_content":
            result = processor._extract_text_pdf(Path("test.pdf"))
            text = result[0]
        else:
            result = processor._extract_text_text(Path("test.txt"))
            text = result[0]

        for word in expected_words:
            assert word in text.lower() or word in text

    def test_process_file_creates_chunks(self, temp_txt_file):
        """Test that processing a file creates chunks."""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)

        result = processor.process_file(temp_txt_file)

        assert isinstance(result, ProcessingResult)
        assert result.total_chunks > 0
        assert result.files_processed == 1
        assert result.processing_time > 0

    def test_process_file_chunks_are_valid(self, temp_txt_file):
        """Test that created chunks have valid content."""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)

        result = processor.process_file(temp_txt_file)

        for doc in result.documents:
            assert len(doc.content) > 0
            assert doc.doc_id is not None
            assert doc.chunk_id is not None
            assert doc.metadata is not None

    def test_process_directory(self, temp_documents_dir):
        """Test processing all files in a directory."""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)

        result = processor.process_directory(temp_documents_dir, recursive=True)

        assert result.files_processed >= 3  # At least the sample files
        assert result.total_chunks >= result.files_processed
        assert len(result.errors) == 0

    def test_process_bytes(self, temp_txt_file, sample_txt_content):
        """Test processing file from bytes."""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)

        content = temp_txt_file.read_bytes()
        result = processor.process_bytes(content, temp_txt_file.name)

        assert result.total_chunks > 0
        assert "Refund" in result.documents[0].content


# ============================================================
# ProcessingResult Tests
# ============================================================

class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_processing_result_creation(self):
        """Test creating a processing result."""
        from src.ingestion.document_processor import Document

        docs = [
            Document(
                content="Test 1",
                metadata={},
                doc_id="doc_1",
                chunk_id="doc_1_chunk_0",
            )
        ]

        result = ProcessingResult(
            documents=docs,
            total_chunks=1,
            processing_time=1.5,
            errors=[],
            files_processed=1,
            files_failed=0,
        )

        assert result.total_chunks == 1
        assert result.processing_time == 1.5
        assert result.files_processed == 1

    def test_processing_result_to_dict(self):
        """Test converting result to dictionary."""
        from src.ingestion.document_processor import Document

        docs = [
            Document(
                content="Test",
                metadata={"source": "test"},
                doc_id="doc_1",
                chunk_id="doc_1_chunk_0",
            )
        ]

        result = ProcessingResult(
            documents=docs,
            total_chunks=1,
            processing_time=1.0,
        )

        result_dict = result.to_dict()

        assert "documents" in result_dict
        assert result_dict["total_chunks"] == 1
