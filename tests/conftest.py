"""
Pytest configuration and shared fixtures for AgenticFlow tests.

This module provides:
- Shared fixtures for all tests
- Pytest configuration
- Test database setup
- Mock configurations
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Environment Setup
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Set up test environment for all tests.

    This runs once before any tests and sets up:
    - Test environment variables
    - Temporary directories
    - Mock API keys
    """
    # Set test environment variables
    os.environ["OPENAI_API_KEY"] = "test-key-openai"
    os.environ["ANTHROPIC_API_KEY"] = "test-key-anthropic"
    os.environ["TAVILY_API_KEY"] = "tvly-test-key-12345"
    os.environ["WORKSPACE_ROOT"] = tempfile.gettempdir()

    # Set test-specific config
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce noise in tests
    os.environ["CHECKPOINT_STORAGE"] = "memory"

    yield

    # Cleanup
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"]:
        if key in os.environ:
            del os.environ[key]


@pytest.fixture(autouse=True)
def reset_storage():
    """
    Reset workflow storage before each test.

    Ensures tests don't interfere with each other.
    """
    from src.api.main import storage

    # Clear storage
    storage.workflows.clear()
    storage.results.clear()

    yield

    # Cleanup after test
    storage.workflows.clear()
    storage.results.clear()


# =============================================================================
# Common Fixtures
# =============================================================================


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response from LLM",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def sample_research_data():
    """Provide sample research data for testing."""
    return [
        {
            "query": "quantum computing",
            "source": "test",
            "title": "Quantum Computing Advances",
            "url": "https://example.com/quantum",
            "content": "Recent breakthroughs in quantum computing have shown...",
            "relevance_score": 0.9,
            "timestamp": "2024-01-15T10:30:00",
        },
        {
            "query": "quantum algorithms",
            "source": "test",
            "title": "New Quantum Algorithms",
            "url": "https://example.com/algorithms",
            "content": "Researchers have developed new quantum algorithms...",
            "relevance_score": 0.85,
            "timestamp": "2024-01-15T10:31:00",
        },
    ]


@pytest.fixture
def sample_agent_output():
    """Provide sample agent output data."""
    return {
        "agent_name": "TestAgent",
        "output": "Test agent output",
        "tools_used": ["test_tool"],
        "timestamp": "2024-01-15T10:30:00",
        "duration_seconds": 1.5,
        "metadata": {"test": "data"},
        "success": True,
    }


# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_collection_modifyitems(config, items):
    """
    Modify test items before collection.

    Adds markers to tests automatically based on their names/locations.
    """
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

        # Mark unit tests
        if "unit" in item.nodeid.lower():
            item.add_marker(pytest.mark.unit)


# =============================================================================
# Test Utilities
# =============================================================================


class MockLLMResponse:
    """Helper class for creating mock LLM responses."""

    @staticmethod
    def text(content: str):
        """Create a mock text response."""
        from langchain_core.messages import AIMessage
        return AIMessage(content=content)

    @staticmethod
    def tool_call(tool_name: str, tool_input: dict):
        """Create a mock tool call response."""
        from langchain_core.messages import ToolMessage
        return ToolMessage(
            content=f"Called {tool_name}",
            tool_call_id="test-id",
        )


@pytest.fixture
def mock_llm_response():
    """Provide a helper for creating mock responses."""
    return MockLLMResponse


# =============================================================================
# Skip Conditions
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API test"
    )
    config.addinivalue_line(
        "markers", "tools: mark test as tools test"
    )
    config.addinivalue_line(
        "markers", "agents: mark test as agents test"
    )
    config.addinivalue_line(
        "markers", "workflow: mark test as workflow test"
    )
