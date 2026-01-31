"""
Pytest configuration and fixtures for FraudDocs-RAG tests.
"""

import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
from httpx import AsyncClient

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ["ENVIRONMENT"] = "testing"
os.environ["LOG_LEVEL"] = "WARNING"


@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent.parent / "tests" / "data"


@pytest.fixture
def sample_document_path(test_data_dir: Path) -> Path:
    """Path to a sample test document."""
    doc_path = test_data_dir / "sample_document.txt"
    if not doc_path.exists():
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(
            "This is a sample document for testing.\n\n"
            "It contains multiple paragraphs.\n\n"
            "Each paragraph represents different content."
        )
    return doc_path


@pytest.fixture
def temp_vector_store(tmp_path: Path) -> str:
    """Temporary directory for vector store testing."""
    vector_dir = tmp_path / "chroma_test"
    vector_dir.mkdir(exist_ok=True)
    return str(vector_dir)


@pytest.fixture
def mock_llm_response() -> str:
    """Mock LLM response for testing."""
    return "This is a simulated response based on the retrieved context."


@pytest.fixture
def mock_retrieved_context() -> list[str]:
    """Mock retrieved document chunks for testing."""
    return [
        "Financial regulations require strict compliance.",
        "Fraud detection systems must monitor transactions.",
        "AML procedures include KYC verification.",
    ]
