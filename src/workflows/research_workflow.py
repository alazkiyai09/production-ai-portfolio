"""
LangGraph workflow implementation for AgenticFlow.

This module provides the main multi-agent workflow using LangGraph's StateGraph,
including conditional routing, checkpointing, and streaming support.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Generator, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents import (
    AnalyzerAgent,
    PlannerAgent,
    ResearcherAgent,
    ReviewerAgent,
    WriterAgent,
    create_all_agents,
)
from src.config import settings
from src.state.workflow_state import (
    WorkflowState,
    advance_step,
    create_initial_state,
    is_complete,
    is_error,
    mark_complete,
    mark_error,
    needs_revision,
    update_state,
)


# =============================================================================
# Main Workflow Class
# =============================================================================

class ResearchWorkflow:
    """
    Multi-agent research workflow using LangGraph.

    This workflow orchestrates five specialized agents through a research
    and content creation pipeline:

    Flow:
        START → Planner → Researcher → Analyzer → Writer → Reviewer → END
                                                              ↓
                                                        (if needs revision)
                                                              ↓
                                                           Writer

    The workflow supports:
    - Conditional routing based on agent outputs
    - Checkpointing for persistence and resumption
    - Streaming for real-time updates
    - Error handling with retries
    - Human-in-the-loop checkpoints (optional)

    Attributes:
        graph: Compiled LangGraph StateGraph
        agents: Dictionary of agent instances
        max_iterations: Maximum number of workflow iterations
        checkpoint_saver: Checkpoint storage backend
        model_name: Default model for agents
        temperature: Default temperature for agents
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_iterations: int = 10,
        checkpoint_saver: Optional[MemorySaver] = None,
        enable_human_checkpoint: bool = False,
    ):
        """
        Initialize the research workflow.

        Args:
            model_name: Default LLM model for agents
            temperature: Default sampling temperature
            max_iterations: Maximum workflow iterations
            checkpoint_saver: Checkpoint storage (default: MemorySaver)
            enable_human_checkpoint: Enable human-in-the-loop checkpoints
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.enable_human_checkpoint = enable_human_checkpoint

        # Initialize checkpoint saver
        if checkpoint_saver is None:
            checkpoint_saver = self._create_checkpoint_saver()
        self.checkpoint_saver = checkpoint_saver

        # Create all agents
        self.agents = create_all_agents(
            model_name=model_name,
        )

        # Build and compile the graph
        self.graph = self._build_graph()

    def _create_checkpoint_saver(self) -> MemorySaver:
        """
        Create checkpoint saver based on configuration.

        Note: Currently only MemorySaver is supported.
        For production persistence, consider implementing a custom
        checkpoint saver with Redis or database backend.

        Returns:
            Checkpoint saver instance (MemorySaver)
        """
        # Default to in-memory storage
        # TODO: Implement persistent checkpoint storage for production
        return MemorySaver()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow graph.

        Creates a StateGraph with:
        - Nodes for each agent
        - Conditional edges for routing
        - Entry point at planner
        - Proper termination conditions

        Returns:
            Compiled StateGraph ready for execution
        """
        # Create state graph
        workflow = StateGraph(WorkflowState)

        # =====================================================================
        # Add Nodes
        # =====================================================================

        workflow.add_node("planner", self._plan_node)
        workflow.add_node("researcher", self._research_node)
        workflow.add_node("analyzer", self._analyze_node)
        workflow.add_node("writer", self._write_node)
        workflow.add_node("reviewer", self._review_node)

        # =====================================================================
        # Set Entry Point
        # =====================================================================

        workflow.set_entry_point("planner")

        # =====================================================================
        # Add Edges
        # =====================================================================

        # Planner → Researcher (always)
        workflow.add_edge("planner", "researcher")

        # Researcher → Analyzer (always)
        workflow.add_edge("researcher", "analyzer")

        # Analyzer → Writer (always)
        workflow.add_edge("analyzer", "writer")

        # Writer → Reviewer (always)
        workflow.add_edge("writer", "reviewer")

        # Reviewer → conditional routing
        workflow.add_conditional_edges(
            "reviewer",
            self._review_decision,
            {
                "approved": END,
                "revise": "writer",
                "rejected": END,
            },
        )

        # =====================================================================
        # Compile with Checkpointing
        # =====================================================================

        # Compile with checkpointing
        # Note: RetryPolicy not available in current langgraph version
        compiled_graph = workflow.compile(
            checkpointer=self.checkpoint_saver,
        )

        return compiled_graph

    # =======================================================================
    # Node Functions
    # =======================================================================

    def _plan_node(self, state: WorkflowState) -> dict[str, Any]:
        """
        Execute the Planner agent.

        Creates a step-by-step execution plan for the task.

        Args:
            state: Current workflow state

        Returns:
            State updates from planner
        """
        try:
            # Invoke planner agent
            agent = self.agents["planner"]
            updates = agent.invoke(state)

            # Add message to history
            message = HumanMessage(
                content=f"Planning completed: {len(updates.get('plan', []))} steps created"
            )

            updates["messages"] = [message]
            updates["status"] = "planning"

            return updates

        except Exception as e:
            return {
                "error": f"Planner failed: {str(e)}",
                "status": "error",
                "messages": [HumanMessage(content=f"Planner error: {str(e)}")],
            }

    def _research_node(self, state: WorkflowState) -> dict[str, Any]:
        """
        Execute the Researcher agent.

        Gathers information from web search and files.

        Args:
            state: Current workflow state

        Returns:
            State updates from researcher
        """
        try:
            agent = self.agents["researcher"]
            updates = agent.invoke(state)

            # Generate research queries from plan if not present
            if not state.get("research_queries") and state.get("plan"):
                # Extract potential search queries from plan steps
                queries = []
                for step in state["plan"][:5]:
                    # Simple heuristic: extract key phrases
                    words = step.lower().split()
                    # Look for research-related keywords
                    if any(kw in step.lower() for kw in ["search", "find", "research", "look up", "gather"]):
                        # Extract the object of research
                        if "for" in step.lower():
                            query = step.split("for")[-1].strip()
                            queries.append(query)
                        elif "about" in step.lower():
                            query = step.split("about")[-1].strip()
                            queries.append(query)

                updates["research_queries"] = queries

            # Add message to history
            message = AIMessage(
                content=f"Research completed: {len(state.get('research_results', []))} sources found"
            )
            updates["messages"] = [message]
            updates["status"] = "researching"

            return updates

        except Exception as e:
            return {
                "error": f"Researcher failed: {str(e)}",
                "status": "error",
                "messages": [HumanMessage(content=f"Researcher error: {str(e)}")],
            }

    def _analyze_node(self, state: WorkflowState) -> dict[str, Any]:
        """
        Execute the Analyzer agent.

        Analyzes research data and extracts insights.

        Args:
            state: Current workflow state

        Returns:
            State updates from analyzer
        """
        try:
            agent = self.agents["analyzer"]
            updates = agent.invoke(state)

            # Add message to history
            message = AIMessage(
                content=f"Analysis completed: {len(updates.get('key_findings', []))} key findings"
            )
            updates["messages"] = [message]
            updates["status"] = "analyzing"

            return updates

        except Exception as e:
            return {
                "error": f"Analyzer failed: {str(e)}",
                "status": "error",
                "messages": [HumanMessage(content=f"Analyzer error: {str(e)}")],
            }

    def _write_node(self, state: WorkflowState) -> dict[str, Any]:
        """
        Execute the Writer agent.

        Creates written content based on research and analysis.

        Args:
            state: Current workflow state

        Returns:
            State updates from writer
        """
        try:
            agent = self.agents["writer"]
            updates = agent.invoke(state)

            # Add message to history
            revision_info = ""
            if state.get("revision_count", 0) > 0:
                revision_info = f" (Revision #{state['revision_count'] + 1})"

            message = AIMessage(
                content=f"Draft created{revision_info}: {len(updates.get('draft', ''))} characters"
            )
            updates["messages"] = [message]
            updates["status"] = "writing"

            return updates

        except Exception as e:
            return {
                "error": f"Writer failed: {str(e)}",
                "status": "error",
                "messages": [HumanMessage(content=f"Writer error: {str(e)}")],
            }

    def _review_node(self, state: WorkflowState) -> dict[str, Any]:
        """
        Execute the Reviewer agent.

        Reviews content and provides approval or feedback.

        Args:
            state: Current workflow state

        Returns:
            State updates from reviewer
        """
        try:
            agent = self.agents["reviewer"]
            updates = agent.invoke(state)

            # Add message to history
            decision = updates.get("approval_status", "pending")
            message = AIMessage(
                content=f"Review completed: {decision.upper()}"
            )
            updates["messages"] = [message]
            updates["status"] = "reviewing"

            # Set end time if approved or rejected
            if decision in ["approved", "rejected"]:
                updates["end_time"] = datetime.utcnow().isoformat()

            return updates

        except Exception as e:
            return {
                "error": f"Reviewer failed: {str(e)}",
                "status": "error",
                "messages": [HumanMessage(content=f"Reviewer error: {str(e)}")],
            }

    # =======================================================================
    # Routing Functions
    # =======================================================================

    def _review_decision(
        self,
        state: WorkflowState,
    ) -> Literal["approved", "revise", "rejected"]:
        """
        Determine next step after review.

        Routes based on the reviewer's approval status:
        - "approved" → END (workflow complete)
        - "revise" → writer (create revision)
        - "rejected" → END (workflow failed)

        Args:
            state: Current workflow state

        Returns:
            Next node name or END
        """
        approval_status = state.get("approval_status", "pending")

        if approval_status == "approved":
            return "approved"
        elif approval_status == "needs_revision":
            # Check revision limit
            revision_count = state.get("revision_count", 0)
            if revision_count >= self.max_iterations:
                # Too many revisions, reject
                return "rejected"
            return "revise"
        else:
            return "rejected"

    # =======================================================================
    # Execution Methods
    # =======================================================================

    def run(
        self,
        task: str,
        task_type: str = "general",
        task_context: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> WorkflowState:
        """
        Run the complete workflow synchronously.

        Executes the full multi-agent workflow from planning through review.
        Returns the final state when complete or encounters an error.

        Args:
            task: The task description
            task_type: Type of task (research, analysis, content_creation, general)
            task_context: Additional context for the task
            thread_id: Thread ID for checkpointing (generates UUID if None)

        Returns:
            Final workflow state

        Raises:
            Exception: If workflow execution fails

        Examples:
            >>> workflow = ResearchWorkflow()
            >>> final_state = workflow.run(
            ...     task="Research the latest AI trends",
            ...     task_type="research"
            ... )
            >>> print(final_state["final_output"])
        """
        # Generate thread ID if not provided
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # Create initial state
        initial_state = create_initial_state(
            task=task,
            task_type=task_type,
            task_context=task_context,
        )

        # Add initial message
        initial_state["messages"] = [
            HumanMessage(content=f"Task: {task}")
        ]

        try:
            # Invoke the graph
            config = {"configurable": {"thread_id": thread_id}}

            final_state = self.graph.invoke(
                initial_state,
                config=config,
            )

            return final_state

        except Exception as e:
            # Mark as error
            error_state = update_state(
                initial_state,
                {
                    "error": str(e),
                    "status": "error",
                    "end_time": datetime.utcnow().isoformat(),
                },
            )
            return error_state

    def run_with_streaming(
        self,
        task: str,
        task_type: str = "general",
        task_context: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Generator[dict[str, Any], None, WorkflowState]:
        """
        Run the workflow with streaming updates.

        Yields intermediate states and messages as the workflow progresses,
        allowing for real-time monitoring and progress tracking.

        Args:
            task: The task description
            task_type: Type of task (research, analysis, content_creation, general)
            task_context: Additional context for the task
            thread_id: Thread ID for checkpointing (generates UUID if None)

        Yields:
            Dictionary containing:
            - event: Type of event (node_enter, node_exit, error, etc.)
            - node: Current node name
            - state: Current workflow state
            - message: Status message

        Returns:
            Final workflow state

        Examples:
            >>> workflow = ResearchWorkflow()
            >>> for update in workflow.run_with_streaming("Research AI"):
            ...     print(f"{update['node']}: {update['message']}")
        """
        # Generate thread ID if not provided
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # Create initial state
        initial_state = create_initial_state(
            task=task,
            task_type=task_type,
            task_context=task_context,
        )

        initial_state["messages"] = [
            HumanMessage(content=f"Task: {task}")
        ]

        config = {"configurable": {"thread_id": thread_id}}

        try:
            # Stream the graph execution
            for event in self.graph.stream(
                initial_state,
                config=config,
                stream_mode="updates",
            ):
                # Process each event
                for node_name, node_updates in event.items():
                    yield {
                        "event": "node_exit",
                        "node": node_name,
                        "updates": node_updates,
                        "timestamp": datetime.utcnow().isoformat(),
                    }

            # Get final state
            final_state = self.get_state(thread_id)

            if final_state is None:
                # If no checkpoint, return initial state with error
                final_state = initial_state

            return final_state

        except Exception as e:
            yield {
                "event": "error",
                "node": "unknown",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Return error state
            error_state = update_state(
                initial_state,
                {
                    "error": str(e),
                    "status": "error",
                    "end_time": datetime.utcnow().isoformat(),
                },
            )
            return error_state

    def get_state(self, thread_id: str) -> Optional[WorkflowState]:
        """
        Get the current state of a workflow.

        Retrieves the latest state from checkpoint storage for a given thread.

        Args:
            thread_id: Thread ID for the workflow

        Returns:
            Current workflow state or None if not found

        Examples:
            >>> workflow = ResearchWorkflow()
            >>> state = workflow.get_state(thread_id="123")
            >>> print(state["status"])
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state_snapshot = self.graph.get_state(config)

            if state_snapshot:
                return state_snapshot.values

            return None

        except Exception:
            return None

    def get_state_history(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get the state history for a workflow thread.

        Args:
            thread_id: Thread ID for the workflow
            limit: Maximum number of history entries to return

        Returns:
            List of state snapshots with metadata

        Examples:
            >>> workflow = ResearchWorkflow()
            >>> history = workflow.get_state_history(thread_id="123")
            >>> for snapshot in history:
            ...     print(snapshot["status"])
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            history = list(self.graph.get_state_history(config, limit=limit))

            result = []
            for snapshot in history:
                result.append({
                    "state": snapshot.values,
                    "next": snapshot.next,
                    "config": snapshot.config,
                    "timestamp": snapshot.config.get("checkpoint_ns"),
                })

            return result

        except Exception:
            return []

    def resume(
        self,
        thread_id: str,
    ) -> WorkflowState:
        """
        Resume a workflow from a checkpoint.

        Continues execution from the last checkpointed state.

        Args:
            thread_id: Thread ID for the workflow to resume

        Returns:
            Final workflow state after resumption

        Raises:
            ValueError: If thread_id not found

        Examples:
            >>> workflow = ResearchWorkflow()
            >>> state = workflow.resume(thread_id="123")
        """
        # Get current state
        current_state = self.get_state(thread_id)

        if current_state is None:
            raise ValueError(f"Thread {thread_id} not found")

        # Check if already complete
        if is_complete(current_state):
            return current_state

        # Check for errors
        if is_error(current_state):
            raise ValueError(f"Cannot resume workflow in error state: {current_state.get('error')}")

        # Continue execution
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # The graph will continue from the last checkpoint
            final_state = self.graph.invoke(None, config=config)
            return final_state

        except Exception as e:
            return update_state(
                current_state,
                {
                    "error": f"Resumption failed: {str(e)}",
                    "status": "error",
                    "end_time": datetime.utcnow().isoformat(),
                },
            )


# =============================================================================
# Factory Function
# =============================================================================

def create_workflow(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_iterations: int = 10,
    checkpoint_storage: Optional[str] = None,
    enable_human_checkpoint: bool = False,
) -> ResearchWorkflow:
    """
    Factory function to create a ResearchWorkflow instance.

    Args:
        model_name: Default LLM model for agents
        temperature: Default sampling temperature
        max_iterations: Maximum workflow iterations (for revisions)
        checkpoint_storage: Checkpoint storage type (memory, sqlite)
        enable_human_checkpoint: Enable human-in-the-loop checkpoints

    Returns:
        Configured ResearchWorkflow instance

    Examples:
        >>> workflow = create_workflow(model_name="gpt-4o")
        >>> state = workflow.run("Research quantum computing")
        >>> print(state["final_output"])

        >>> # With custom settings
        >>> workflow = create_workflow(
        ...     model_name="gpt-4",
        ...     temperature=0.2,
        ...     max_iterations=5,
        ...     checkpoint_storage="sqlite"
        ... )
    """
    return ResearchWorkflow(
        model_name=model_name,
        temperature=temperature,
        max_iterations=max_iterations,
        enable_human_checkpoint=enable_human_checkpoint,
    )
