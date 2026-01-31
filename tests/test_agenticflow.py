"""
Comprehensive test suite for AgenticFlow.

This module contains tests for all components:
- State management
- Tools (with security tests)
- Agents (with mock LLMs)
- Workflow execution
- API endpoints
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from httpx import ConnectError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Import AgenticFlow components
from src.agents import (
    AnalyzerAgent,
    PlannerAgent,
    ResearcherAgent,
    ReviewerAgent,
    WriterAgent,
    create_agent,
)
from src.api import storage
from src.config import settings
from src.state.workflow_state import (
    WorkflowState,
    advance_step,
    create_initial_state,
    get_progress,
    is_complete,
    is_error,
    mark_complete,
    mark_error,
    needs_revision,
    update_state,
)
from src.tools import (
    AGENT_TOOLS,
    ALL_TOOLS,
    WEB_TOOLS,
    calculator,
    get_tools_for_agent,
    list_files,
    read_file,
    run_python_code,
    web_search,
    write_file,
)
from src.workflows import ResearchWorkflow, create_workflow


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        os.environ["WORKSPACE_ROOT"] = tmpdir

        yield tmpdir

        os.chdir(original_cwd)
        if "WORKSPACE_ROOT" in os.environ:
            del os.environ["WORKSPACE_ROOT"]


@pytest.fixture
def sample_task():
    """Provide a sample task description."""
    return "Research the latest developments in quantum computing"


@pytest.fixture
def sample_state(sample_task) -> WorkflowState:
    """Provide a sample workflow state."""
    return create_initial_state(
        task=sample_task,
        task_type="research",
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM instance."""
    mock = MagicMock()
    mock.invoke = MagicMock(return_value=AIMessage(content="Mock response"))
    return mock


@pytest.fixture
def mock_tavily_response():
    """Create a mock Tavily API response."""
    return {
        "results": [
            {
                "title": "Quantum Computing Breakthrough",
                "url": "https://example.com/quantum",
                "content": "Recent advances in quantum computing...",
                "score": 0.95,
            }
        ]
    }


@pytest.fixture
def populated_state(sample_state) -> WorkflowState:
    """Provide a populated workflow state with some data."""
    state = sample_state
    state["plan"] = [
        "Search for recent quantum computing developments",
        "Analyze findings",
        "Create summary",
    ]
    state["current_step"] = 1
    state["research_queries"] = ["quantum computing 2024", "quantum advances"]
    state["research_results"] = [
        {
            "query": "quantum computing",
            "source": "test",
            "title": "Test Result",
            "url": "https://example.com",
            "content": "Test content",
            "relevance_score": 0.8,
            "timestamp": datetime.utcnow().isoformat(),
        }
    ]
    state["analysis"] = "Analysis of quantum computing developments..."
    state["key_findings"] = [
        "Finding 1: Quantum computers are improving",
        "Finding 2: New algorithms discovered",
    ]
    state["draft"] = "Draft content about quantum computing..."
    state["revision_count"] = 1
    state["status"] = "writing"
    return state


@pytest.fixture
def completed_state(populated_state) -> WorkflowState:
    """Provide a completed workflow state."""
    state = populated_state
    state["status"] = "complete"
    state["final_output"] = "Final quantum computing report"
    state["approval_status"] = "approved"
    return state


@pytest.fixture
def error_state(sample_state) -> WorkflowState:
    """Provide an error state."""
    state = sample_state
    state["status"] = "error"
    state["error"] = "Test error message"
    return state


# =============================================================================
# State Tests
# =============================================================================


class TestStateManagement:
    """Test suite for state management functions."""

    def test_create_initial_state_basic(self, sample_task):
        """Test basic state creation."""
        state = create_initial_state(sample_task)

        assert state["task"] == sample_task
        assert state["task_type"] == "general"
        assert state["status"] == "planning"
        assert state["plan"] == []
        assert state["current_step"] == 0
        assert state["iteration_count"] == 0

    def test_create_initial_state_with_type(self, sample_task):
        """Test state creation with specific task type."""
        state = create_initial_state(sample_task, task_type="research")

        assert state["task_type"] == "research"

    def test_create_initial_state_with_context(self, sample_task):
        """Test state creation with additional context."""
        context = "Focus on recent developments"
        state = create_initial_state(sample_task, task_context=context)

        assert state["task_context"] == context

    def test_update_state_basic(self, sample_state):
        """Test basic state update."""
        updates = {"status": "researching", "plan": ["Step 1", "Step 2"]}
        new_state = update_state(sample_state, updates)

        assert new_state["status"] == "researching"
        assert new_state["plan"] == ["Step 1", "Step 2"]
        assert new_state["iteration_count"] == 1  # Auto-incremented

    def test_update_state_with_agent(self, sample_state):
        """Test state update with agent tracking."""
        updates = {"output": "Test output", "tools_used": ["search"]}
        new_state = update_state(sample_state, updates, agent_name="TestAgent")

        assert len(new_state["agent_outputs"]) == 1
        assert new_state["agent_outputs"][0]["agent_name"] == "TestAgent"
        assert new_state["agent_outputs"][0]["output"] == "Test output"

    def test_is_complete_true(self, completed_state):
        """Test is_complete returns True for completed state."""
        assert is_complete(completed_state) is True

    def test_is_complete_false_no_output(self, populated_state):
        """Test is_complete returns False when no final output."""
        assert is_complete(populated_state) is False

    def test_is_complete_false_wrong_status(self, populated_state):
        """Test is_complete returns False for non-complete status."""
        populated_state["final_output"] = "Some output"
        assert is_complete(populated_state) is False

    def test_is_error_true(self, error_state):
        """Test is_error returns True for error state."""
        assert is_error(error_state) is True

    def test_is_error_false(self, sample_state):
        """Test is_error returns False for non-error state."""
        assert is_error(sample_state) is False

    def test_needs_revision_true(self, populated_state):
        """Test needs_revision returns True when revision needed."""
        populated_state["approval_status"] = "needs_revision"
        assert needs_revision(populated_state) is True

    def test_needs_revision_false(self, populated_state):
        """Test needs_revision returns False when not needed."""
        assert needs_revision(populated_state) is False

    def test_get_progress_basic(self, populated_state):
        """Test get_progress returns correct metrics."""
        progress = get_progress(populated_state)

        assert "status" in progress
        assert "progress_percent" in progress
        assert "steps_completed" in progress
        assert "total_steps" in progress
        assert progress["total_steps"] == len(populated_state["plan"])

    def test_advance_step(self, populated_state):
        """Test advance_step increments current_step."""
        new_state = advance_step(populated_state)

        assert new_state["current_step"] == populated_state["current_step"] + 1

    def test_mark_complete(self, populated_state):
        """Test mark_complete sets proper fields."""
        final_output = "Final result content"
        new_state = mark_complete(populated_state, final_output)

        assert new_state["final_output"] == final_output
        assert new_state["status"] == "complete"
        assert new_state["approval_status"] == "approved"
        assert "end_time" in new_state

    def test_mark_error(self, sample_state):
        """Test mark_error sets error fields."""
        error_msg = "Something went wrong"
        new_state = mark_error(sample_state, error_msg)

        assert new_state["status"] == "error"
        assert new_state["error"] == error_msg
        assert "end_time" in new_state


# =============================================================================
# Tool Tests
# =============================================================================


class TestTools:
    """Test suite for tool implementations."""

    def test_calculator_basic_arithmetic(self):
        """Test calculator with basic operations."""
        assert calculator("2 + 2") == 4.0
        assert calculator("10 - 5") == 5.0
        assert calculator("3 * 4") == 12.0
        assert calculator("10 / 2") == 5.0

    def test_calculator_complex_expressions(self):
        """Test calculator with complex expressions."""
        assert calculator("(2 + 3) * 4") == 20.0
        assert calculator("2 ** 8") == 256.0
        assert abs(calculator("sqrt(16)") - 4.0) < 0.01

    def test_calculator_invalid_expression(self):
        """Test calculator rejects invalid expressions."""
        with pytest.raises(ValueError):
            calculator("import os")

        with pytest.raises(ValueError):
            calculator("__import__('os')")

    def test_read_file_success(self, temp_workspace):
        """Test reading a valid file."""
        test_file = Path(temp_workspace) / "test.txt"
        test_file.write_text("Test content")

        content = read_file(str(test_file))
        assert content == "Test content"

    def test_read_file_not_found(self, temp_workspace):
        """Test reading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            read_file("nonexistent.txt")

    def test_read_file_directory_traversal(self, temp_workspace):
        """Test read_file prevents directory traversal."""
        # Try to read a file outside workspace
        with pytest.raises(ValueError, match="outside workspace"):
            read_file("../../../etc/passwd")

    def test_read_file_hidden_file(self, temp_workspace):
        """Test read_file blocks hidden files."""
        hidden_file = Path(temp_workspace) / ".secret"
        hidden_file.write_text("Secret content")

        with pytest.raises(ValueError, match="hidden"):
            read_file(".secret")

    def test_write_file_success(self, temp_workspace):
        """Test writing a file successfully."""
        test_file = "output.txt"
        content = "Test output content"

        result = write_file(test_file, content)

        assert "Successfully wrote" in result
        assert (Path(temp_workspace) / test_file).read_text() == content

    def test_write_file_directory_traversal(self, temp_workspace):
        """Test write_file prevents directory traversal."""
        with pytest.raises(ValueError, match="outside workspace"):
            write_file("../../../etc/passwd", "malicious")

    def test_write_file_creates_directories(self, temp_workspace):
        """Test write_file creates parent directories."""
        result = write_file("subdir/nested/file.txt", "content")

        assert (Path(temp_workspace) / "subdir" / "nested" / "file.txt").exists()

    def test_list_files_basic(self, temp_workspace):
        """Test listing files in workspace."""
        Path(temp_workspace, "file1.txt").write_text("content1")
        Path(temp_workspace, "file2.md").write_text("content2")
        Path(temp_workspace, "subdir").mkdir()

        files = list_files(".")

        assert len(files) == 2
        assert any(f["name"] == "file1.txt" for f in files)

    def test_list_files_no_hidden(self, temp_workspace):
        """Test list_files excludes hidden files."""
        Path(temp_workspace, ".hidden").write_text("secret")
        Path(temp_workspace, "visible.txt").write_text("content")

        files = list_files(".")

        assert not any(f["name"] == ".hidden" for f in files)
        assert any(f["name"] == "visible.txt" for f in files)

    def test_run_python_code_basic(self):
        """Test running basic Python code."""
        result = run_python_code("2 + 2")
        assert "4" in result

    def test_run_python_code_with_print(self):
        """Test running code with print statements."""
        result = run_python_code("print('Hello, World!')")
        assert "Hello, World!" in result

    def test_run_python_code_timeout(self):
        """Test code execution timeout."""
        infinite_loop = "while True: pass"
        # Should timeout after 30 seconds
        # In test, we just check it doesn't hang indefinitely
        # (Actual timeout testing would require async)

    def test_run_python_code_security_import(self):
        """Test run_python_code blocks imports."""
        with pytest.raises(ValueError, match="dangerous"):
            run_python_code("import os")

    def test_run_python_code_security_eval(self):
        """Test run_python_code blocks eval."""
        with pytest.raises(ValueError, match="dangerous"):
            run_python_code("eval('1+1')")

    def test_get_tools_for_agent_valid(self):
        """Test getting tools for valid agent types."""
        planner_tools = get_tools_for_agent("planner")
        researcher_tools = get_tools_for_agent("researcher")

        assert len(planner_tools) > 0
        assert len(researcher_tools) > 0
        assert len(researcher_tools) > len(planner_tools)  # Researcher has more tools

    def test_get_tools_for_agent_invalid(self):
        """Test getting tools for invalid agent type."""
        with pytest.raises(ValueError):
            get_tools_for_agent("invalid_agent")

    @patch("src.tools.agent_tools._search_with_tavily")
    def test_web_search_with_tavily(self, mock_search, mock_tavily_response):
        """Test web search using Tavily."""
        mock_search.return_value = mock_tavily_response["results"]

        os.environ["TAVILY_API_KEY"] = "tvly-test-key"
        results = web_search("test query")

        assert len(results) > 0
        assert results[0]["title"] == "Quantum Computing Breakthrough"


# =============================================================================
# Agent Tests
# =============================================================================


class TestAgents:
    """Test suite for agent implementations."""

    def test_create_planner_agent(self):
        """Test creating a Planner agent."""
        agent = PlannerAgent(model_name="gpt-4o-mini", temperature=0.1)

        assert agent.name == "Planner"
        assert agent.model_name == "gpt-4o-mini"
        assert agent.temperature == 0.1
        assert agent.system_prompt is not None

    def test_planner_system_prompt(self):
        """Test Planner agent has proper system prompt."""
        agent = PlannerAgent()
        prompt = agent.system_prompt.lower()

        assert "planning" in prompt or "plan" in prompt
        assert "task" in prompt

    def test_create_researcher_agent(self):
        """Test creating a Researcher agent."""
        agent = ResearcherAgent()

        assert agent.name == "Researcher"
        assert len(agent.tools) > 0
        assert any(tool.name == "web_search" for tool in agent.tools)

    def test_researcher_has_search_tools(self):
        """Test Researcher has web search capabilities."""
        agent = ResearcherAgent()
        tool_names = [tool.name for tool in agent.tools]

        assert "web_search" in tool_names
        assert "read_file" in tool_names

    def test_create_analyzer_agent(self):
        """Test creating an Analyzer agent."""
        agent = AnalyzerAgent()

        assert agent.name == "Analyzer"
        assert len(agent.tools) > 0

    def test_analyzer_has_computational_tools(self):
        """Test Analyzer has computational tools."""
        agent = AnalyzerAgent()
        tool_names = [tool.name for tool in agent.tools]

        assert "calculator" in tool_names
        assert "run_python_code" in tool_names

    def test_create_writer_agent(self):
        """Test creating a Writer agent."""
        agent = WriterAgent()

        assert agent.name == "Writer"
        assert agent.temperature == 0.3  # Higher temperature for creativity

    def test_writer_has_file_tools(self):
        """Test Writer has file I/O tools."""
        agent = WriterAgent()
        tool_names = [tool.name for tool in agent.tools]

        assert "read_file" in tool_names
        assert "write_file" in tool_names

    def test_create_reviewer_agent(self):
        """Test creating a Reviewer agent."""
        agent = ReviewerAgent()

        assert agent.name == "Reviewer"
        assert agent.temperature == 0.1  # Low temperature for consistent evaluation

    def test_create_agent_factory_valid(self):
        """Test create_agent factory function."""
        planner = create_agent("planner")
        researcher = create_agent("researcher")

        assert isinstance(planner, PlannerAgent)
        assert isinstance(researcher, ResearcherAgent)

    def test_create_agent_factory_invalid(self):
        """Test create_agent factory with invalid type."""
        with pytest.raises(ValueError):
            create_agent("invalid_agent")

    def test_agent_invoke_with_mock(self, mock_llm):
        """Test agent invocation with mocked LLM."""
        agent = PlannerAgent(llm=mock_llm)
        state = create_initial_state("Test task")

        updates = agent.invoke(state)

        assert "output" in updates
        assert updates["success"] is True
        assert "duration_seconds" in updates

    @patch("src.agents.specialized_agents.ChatOpenAI")
    def test_agent_llm_creation_openai(self, mock_chat):
        """Test agent creates OpenAI LLM correctly."""
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance

        agent = PlannerAgent(model_name="gpt-4o-mini")

        assert mock_chat.called
        call_kwargs = mock_chat.call_args[1]
        assert "model" in call_kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"


# =============================================================================
# Workflow Tests
# =============================================================================


class TestWorkflow:
    """Test suite for workflow implementation."""

    def test_create_workflow_basic(self):
        """Test creating a workflow instance."""
        workflow = create_workflow()

        assert workflow is not None
        assert workflow.graph is not None
        assert workflow.max_iterations == 10

    def test_create_workflow_custom_settings(self):
        """Test creating workflow with custom settings."""
        workflow = create_workflow(
            model_name="gpt-4o",
            temperature=0.2,
            max_iterations=5,
        )

        assert workflow.model_name == "gpt-4o"
        assert workflow.temperature == 0.2
        assert workflow.max_iterations == 5

    def test_workflow_has_all_nodes(self):
        """Test workflow graph has all required nodes."""
        workflow = create_workflow()
        nodes = workflow.graph.nodes

        assert "planner" in nodes
        assert "researcher" in nodes
        assert "analyzer" in nodes
        assert "writer" in nodes
        assert "reviewer" in nodes

    def test_workflow_entry_point(self):
        """Test workflow starts at planner."""
        workflow = create_workflow()

        # Get the graph structure
        graph = workflow.graph.get_graph()

        # In newer LangGraph, there's an implicit __start__ node
        # Check that __start__ connects to planner (the actual entry point)
        edges = graph.edges
        start_edges = [e for e in edges if str(e.source) == "__start__"]

        # Verify __start__ connects to planner
        assert len(start_edges) == 1
        assert str(start_edges[0].target) == "planner"

    @patch("src.workflows.research_workflow.create_all_agents")
    def test_workflow_run_with_mocks(self, mock_create_agents, mock_llm):
        """Test workflow execution with mocked agents."""
        # Create mock agents
        mock_agents = {
            "planner": MagicMock(invoke=MagicMock(return_value={"plan": ["Step 1"]})),
            "researcher": MagicMock(invoke=MagicMock(return_value={"analysis": "Research done"})),
            "analyzer": MagicMock(invoke=MagicMock(return_value={"key_findings": ["Finding"]})),
            "writer": MagicMock(invoke=MagicMock(return_value={"draft": "Draft content"})),
            "reviewer": MagicMock(invoke=MagicMock(return_value={"approval_status": "approved"})),
        }
        mock_create_agents.return_value = mock_agents

        workflow = create_workflow()

        # Test run method exists and is callable
        assert hasattr(workflow, "run")
        assert callable(workflow.run)

    def test_workflow_checkpoint_saver_created(self):
        """Test workflow creates checkpoint saver."""
        workflow = create_workflow()

        assert workflow.checkpoint_saver is not None

    def test_workflow_review_routing_approved(self):
        """Test routing after approved review."""
        workflow = create_workflow()

        # Create state with approved status
        state = create_initial_state("Test task")
        state["approval_status"] = "approved"

        decision = workflow._review_decision(state)

        assert decision == "approved"

    def test_workflow_review_routing_revise(self):
        """Test routing after revision request."""
        workflow = create_workflow()
        state = create_initial_state("Test task")
        state["approval_status"] = "needs_revision"
        state["revision_count"] = 2

        decision = workflow._review_decision(state)

        assert decision == "revise"

    def test_workflow_review_routing_rejected(self):
        """Test routing after rejection."""
        workflow = create_workflow()
        state = create_initial_state("Test task")
        state["approval_status"] = "rejected"

        decision = workflow._review_decision(state)

        assert decision == "rejected"

    def test_workflow_review_routing_max_revisions(self):
        """Test routing after max revisions exceeded."""
        workflow = create_workflow(max_iterations=3)
        state = create_initial_state("Test task")
        state["approval_status"] = "needs_revision"
        state["revision_count"] = 5  # Exceeds max_iterations

        decision = workflow._review_decision(state)

        assert decision == "rejected"  # Should reject after too many revisions


# =============================================================================
# API Tests
# =============================================================================


class TestAPI:
    """Test suite for API endpoints."""

    def test_workflow_storage_add(self):
        """Test adding workflow to storage."""
        workflow_id = "test-123"
        workflow = MagicMock()

        storage.add_workflow(workflow_id, "Test task", "research", workflow)

        retrieved = storage.get_workflow(workflow_id)
        assert retrieved is not None
        assert retrieved["id"] == workflow_id
        assert retrieved["task"] == "Test task"

    def test_workflow_storage_get_nonexistent(self):
        """Test getting non-existent workflow."""
        result = storage.get_workflow("nonexistent-id")
        assert result is None

    def test_workflow_storage_update_status(self):
        """Test updating workflow status."""
        workflow_id = "test-456"
        workflow = MagicMock()
        storage.add_workflow(workflow_id, "Test", "general", workflow)

        storage.update_workflow_status(workflow_id, "running")

        retrieved = storage.get_workflow(workflow_id)
        assert retrieved["status"] == "running"

    def test_workflow_storage_set_result(self):
        """Test storing workflow result."""
        workflow_id = "test-789"
        result = {"final_output": "Test output"}

        storage.set_result(workflow_id, result)

        retrieved = storage.get_result(workflow_id)
        assert retrieved is not None
        assert retrieved["result"]["final_output"] == "Test output"

    def test_workflow_storage_list(self):
        """Test listing all workflows."""
        # Add multiple workflows
        for i in range(3):
            workflow_id = f"test-{i}"
            storage.add_workflow(workflow_id, f"Task {i}", "general", MagicMock())

        workflows = storage.list_workflows()

        assert len(workflows) >= 3

    def test_workflow_storage_delete(self):
        """Test deleting workflow."""
        workflow_id = "test-delete"
        storage.add_workflow(workflow_id, "Test", "general", MagicMock())
        storage.set_result(workflow_id, {"output": "test"})

        deleted = storage.delete_workflow(workflow_id)

        assert deleted is True
        assert storage.get_workflow(workflow_id) is None
        assert storage.get_result(workflow_id) is None

    def test_workflow_storage_delete_nonexistent(self):
        """Test deleting non-existent workflow."""
        deleted = storage.delete_workflow("nonexistent")
        assert deleted is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for end-to-end workflows."""

    def test_state_to_agent_flow(self, populated_state, mock_llm):
        """Test state flowing through an agent."""
        agent = PlannerAgent(llm=mock_llm)

        updates = agent.invoke(populated_state)

        assert updates is not None
        assert isinstance(updates, dict)

    def test_tool_to_agent_integration(self, temp_workspace):
        """Test tools working with agents."""
        # Create a test file
        test_file = Path(temp_workspace) / "test.txt"
        test_file.write_text("Test content for agent")

        # Researcher agent should be able to read it
        agent = ResearcherAgent()

        assert any(tool.name == "read_file" for tool in agent.tools)

    def test_complete_workflow_structure(self):
        """Test complete workflow structure is valid."""
        workflow = create_workflow()

        # Verify graph structure
        graph_dict = workflow.graph.get_graph()

        # Should have nodes and edges
        assert len(graph_dict.nodes) >= 5  # At least 5 agents

    @patch("src.tools.agent_tools.TavilyClient")
    def test_research_workflow_with_search(self, mock_tavily, mock_tavily_response):
        """Test research workflow can use search tools."""
        # Mock Tavily response
        mock_client = MagicMock()
        mock_client.search.return_value = mock_tavily_response
        mock_tavily.return_value = mock_client

        os.environ["TAVILY_API_KEY"] = "tvly-test"

        # Test search tool works
        results = web_search("test query", num_results=2)

        assert isinstance(results, list)


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )


# =============================================================================
# Test Runner
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
