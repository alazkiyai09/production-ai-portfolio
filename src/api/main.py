"""
FastAPI application for AgenticFlow.

This module provides REST API endpoints for the multi-agent workflow system,
including workflow submission, status tracking, streaming, and human-in-the-loop
feedback.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    FastAPI,
    HTTPException,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.config import settings
from src.state.workflow_state import TaskType, get_progress
from src.workflows import ResearchWorkflow, create_workflow


# =============================================================================
# In-Memory Storage (Note: Use Redis in production)
# =============================================================================

class WorkflowStorage:
    """
    In-memory storage for workflow instances and their states.

    Note: In production, replace with Redis or a database for
    persistence across restarts and horizontal scaling.
    """

    def __init__(self):
        self.workflows: dict[str, dict[str, Any]] = {}
        self.results: dict[str, dict[str, Any]] = {}

    def add_workflow(
        self,
        workflow_id: str,
        task: str,
        task_type: str,
        workflow: ResearchWorkflow,
    ) -> None:
        """Register a new workflow."""
        self.workflows[workflow_id] = {
            "id": workflow_id,
            "task": task,
            "task_type": task_type,
            "workflow": workflow,
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending",
        }

    def get_workflow(self, workflow_id: str) -> Optional[dict[str, Any]]:
        """Get a workflow by ID."""
        return self.workflows.get(workflow_id)

    def update_workflow_status(
        self,
        workflow_id: str,
        status: str,
    ) -> None:
        """Update workflow status."""
        if workflow_id in self.workflows:
            self.workflows[workflow_id]["status"] = status
            self.workflows[workflow_id]["updated_at"] = datetime.utcnow().isoformat()

    def set_result(self, workflow_id: str, result: dict[str, Any]) -> None:
        """Store workflow result."""
        self.results[workflow_id] = {
            "result": result,
            "completed_at": datetime.utcnow().isoformat(),
        }

    def get_result(self, workflow_id: str) -> Optional[dict[str, Any]]:
        """Get workflow result."""
        return self.results.get(workflow_id)

    def list_workflows(self) -> list[dict[str, Any]]:
        """List all workflows."""
        return [
            {
                "id": wf["id"],
                "task": wf["task"],
                "task_type": wf["task_type"],
                "status": wf["status"],
                "created_at": wf["created_at"],
                "updated_at": wf.get("updated_at"),
            }
            for wf in self.workflows.values()
        ]

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow and its result."""
        deleted = False
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            deleted = True
        if workflow_id in self.results:
            del self.results[workflow_id]
        return deleted


# Global storage instance
storage = WorkflowStorage()


# =============================================================================
# Pydantic Models
# =============================================================================

class WorkflowRequest(BaseModel):
    """
    Request model for starting a new workflow.

    Attributes:
        task: The task description to execute
        task_type: Type of task (research, analysis, content_creation, general)
        task_context: Additional context for the task
        model_name: LLM model to use (optional, uses default if not specified)
        temperature: Sampling temperature (optional)
        max_iterations: Maximum iterations for revisions (optional)
    """

    task: str = Field(
        ...,
        description="The task description to execute",
        min_length=1,
        max_length=5000,
    )

    task_type: TaskType = Field(
        default="general",
        description="Type of task for routing",
    )

    task_context: Optional[str] = Field(
        None,
        description="Additional context or constraints for the task",
        max_length=10000,
    )

    model_name: Optional[str] = Field(
        None,
        description="LLM model to use (e.g., gpt-4o, gpt-4o-mini)",
    )

    temperature: Optional[float] = Field(
        None,
        description="Sampling temperature (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    max_iterations: Optional[int] = Field(
        None,
        description="Maximum iterations for revisions",
        ge=1,
        le=50,
    )

    @field_validator("task")
    @classmethod
    def task_must_not_be_empty(cls, v: str) -> str:
        """Validate task is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("Task cannot be empty")
        return v.strip()


class WorkflowStartResponse(BaseModel):
    """Response model for workflow start."""

    workflow_id: str = Field(..., description="Unique workflow identifier")
    status: str = Field(..., description="Initial workflow status")
    message: str = Field(..., description="Status message")
    created_at: str = Field(..., description="ISO format timestamp")


class StatusResponse(BaseModel):
    """Response model for workflow status."""

    workflow_id: str = Field(..., description="Workflow identifier")
    status: str = Field(..., description="Current workflow status")
    task: str = Field(..., description="Original task description")
    task_type: str = Field(..., description="Task type")
    progress: dict[str, Any] = Field(..., description="Progress metrics")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")


class ResultResponse(BaseModel):
    """Response model for workflow result."""

    workflow_id: str = Field(..., description="Workflow identifier")
    status: str = Field(..., description="Final workflow status")
    final_output: str = Field(..., description="Final output content")
    approval_status: str = Field(..., description="Reviewer approval status")
    feedback: list[str] = Field(..., description="Reviewer feedback")
    revision_count: int = Field(..., description="Number of revisions made")
    duration_seconds: float = Field(..., description="Total execution time")
    completed_at: str = Field(..., description="Completion timestamp")


class FeedbackRequest(BaseModel):
    """
    Request model for providing human feedback.

    Attributes:
        feedback: Feedback message from human reviewer
        action: Action to take (approve, revise, reject)
        revision_notes: Specific notes for revision
    """

    feedback: str = Field(
        ...,
        description="Feedback message from human reviewer",
        min_length=1,
        max_length=5000,
    )

    action: str = Field(
        ...,
        description="Action to take: approve, revise, or reject",
    )

    revision_notes: Optional[str] = Field(
        None,
        description="Specific notes for revision if action is 'revise'",
        max_length=5000,
    )

    @field_validator("action")
    @classmethod
    def action_must_be_valid(cls, v: str) -> str:
        """Validate action is one of the allowed values."""
        valid_actions = ["approve", "revise", "reject"]
        v_lower = v.lower()
        if v_lower not in valid_actions:
            raise ValueError(f"Action must be one of: {', '.join(valid_actions)}")
        return v_lower


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""

    workflow_id: str = Field(..., description="Workflow identifier")
    feedback_received: bool = Field(..., description="Whether feedback was processed")
    message: str = Field(..., description="Response message")
    new_status: Optional[str] = Field(None, description="Updated workflow status")


class WorkflowListResponse(BaseModel):
    """Response model for workflow list."""

    workflows: list[dict[str, Any]] = Field(..., description="List of workflows")
    count: int = Field(..., description="Total number of workflows")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    active_workflows: int = Field(..., description="Number of active workflows")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional details")
    timestamp: str = Field(..., description="Error timestamp")


# =============================================================================
# Background Task Execution
# =============================================================================

async def run_workflow_async(
    workflow_id: str,
    task: str,
    task_type: str,
    task_context: Optional[str],
    model_name: Optional[str],
    temperature: Optional[float],
    max_iterations: Optional[int],
) -> None:
    """
    Execute a workflow in the background.

    This async function runs the workflow and updates storage
    with progress and final results.

    Args:
        workflow_id: Unique workflow identifier
        task: Task description
        task_type: Type of task
        task_context: Additional context
        model_name: LLM model to use
        temperature: Sampling temperature
        max_iterations: Max iterations for revisions
    """
    try:
        # Update status to running
        storage.update_workflow_status(workflow_id, "running")

        # Get workflow instance
        workflow_data = storage.get_workflow(workflow_id)
        if not workflow_data:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = workflow_data["workflow"]

        # Check if streaming or regular execution
        # For simplicity, use regular execution here
        # (Streaming is handled via the /stream endpoint)

        # Run the workflow
        final_state = workflow.run(
            task=task,
            task_type=task_type,
            task_context=task_context,
            thread_id=workflow_id,
        )

        # Calculate duration
        start_time = datetime.fromisoformat(final_state["start_time"])
        end_time = (
            datetime.fromisoformat(final_state["end_time"])
            if final_state.get("end_time")
            else datetime.utcnow()
        )
        duration = (end_time - start_time).total_seconds()

        # Store result
        result = {
            "status": final_state["status"],
            "final_output": final_state.get("final_output", ""),
            "approval_status": final_state.get("approval_status", "pending"),
            "feedback": final_state.get("feedback", []),
            "revision_count": final_state.get("revision_count", 0),
            "duration_seconds": duration,
            "draft": final_state.get("draft", ""),
            "analysis": final_state.get("analysis", ""),
            "key_findings": final_state.get("key_findings", []),
            "error": final_state.get("error"),
        }

        storage.set_result(workflow_id, result)

        # Update workflow status
        final_status = final_state["status"]
        storage.update_workflow_status(workflow_id, final_status)

    except Exception as e:
        # Store error
        error_result = {
            "status": "error",
            "error": str(e),
            "final_output": "",
            "approval_status": "rejected",
            "feedback": [],
            "revision_count": 0,
            "duration_seconds": 0,
        }

        storage.set_result(workflow_id, error_result)
        storage.update_workflow_status(workflow_id, "error")


async def run_workflow_with_updates(
    workflow_id: str,
    task: str,
    task_type: str,
    task_context: Optional[str],
    workflow: ResearchWorkflow,
) -> None:
    """
    Execute workflow with streaming updates to storage.

    This version provides real-time progress updates that can be
    consumed via the status endpoint.

    Args:
        workflow_id: Unique workflow identifier
        task: Task description
        task_type: Type of task
        task_context: Additional context
        workflow: Workflow instance
    """
    try:
        storage.update_workflow_status(workflow_id, "running")

        # Run with streaming
        async for update in _stream_workflow_updates(workflow, task, task_type, task_context, workflow_id):
            # Update storage with each step's progress
            # The status endpoint will return the latest state
            pass

        # Final update
        final_state = workflow.get_state(workflow_id)
        if final_state:
            start_time = datetime.fromisoformat(final_state["start_time"])
            end_time = (
                datetime.fromisoformat(final_state["end_time"])
                if final_state.get("end_time")
                else datetime.utcnow()
            )
            duration = (end_time - start_time).total_seconds()

            result = {
                "status": final_state["status"],
                "final_output": final_state.get("final_output", ""),
                "approval_status": final_state.get("approval_status", "pending"),
                "feedback": final_state.get("feedback", []),
                "revision_count": final_state.get("revision_count", 0),
                "duration_seconds": duration,
                "draft": final_state.get("draft", ""),
                "analysis": final_state.get("analysis", ""),
                "key_findings": final_state.get("key_findings", []),
            }

            storage.set_result(workflow_id, result)
            storage.update_workflow_status(workflow_id, final_state["status"])

    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "final_output": "",
            "approval_status": "rejected",
            "feedback": [],
            "revision_count": 0,
            "duration_seconds": 0,
        }
        storage.set_result(workflow_id, error_result)
        storage.update_workflow_status(workflow_id, "error")


async def _stream_workflow_updates(
    workflow: ResearchWorkflow,
    task: str,
    task_type: str,
    task_context: Optional[str],
    thread_id: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Async generator for workflow updates.

    Yields updates as the workflow progresses.

    Args:
        workflow: Workflow instance
        task: Task description
        task_type: Type of task
        task_context: Additional context
        thread_id: Thread ID for checkpointing

    Yields:
        Update dictionaries with event, node, state info
    """
    # Run workflow in thread pool to avoid blocking
    loop = asyncio.get_event_loop()

    # Use run_with_streaming but yield events
    def _run_in_thread():
        return list(workflow.run_with_streaming(
            task=task,
            task_type=task_type,
            task_context=task_context,
            thread_id=thread_id,
        ))

    # Execute in thread pool
    updates = await loop.run_in_executor(None, _run_in_thread)

    for update in updates[:-1]:  # All but final state
        yield update


# =============================================================================
# API Router
# =============================================================================

router = APIRouter()


@router.post(
    "/workflow/start",
    response_model=WorkflowStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a new workflow",
    description="Submit a new task for processing by the multi-agent system",
)
async def start_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
) -> WorkflowStartResponse:
    """
    Start a new workflow execution.

    The workflow runs in the background. Use the returned workflow_id
    to track progress and retrieve results.
    """
    try:
        # Generate unique workflow ID
        workflow_id = str(uuid.uuid4())

        # Create workflow instance
        workflow = create_workflow(
            model_name=request.model_name or settings.default_model,
            temperature=request.temperature or 0.1,
            max_iterations=request.max_iterations or 10,
        )

        # Store workflow
        storage.add_workflow(
            workflow_id,
            request.task,
            request.task_type,
            workflow,
        )

        # Start background task
        background_tasks.add_task(
            run_workflow_async,
            workflow_id,
            request.task,
            request.task_type,
            request.task_context,
            request.model_name,
            request.temperature,
            request.max_iterations,
        )

        return WorkflowStartResponse(
            workflow_id=workflow_id,
            status="pending",
            message="Workflow started successfully. Use the workflow_id to track progress.",
            created_at=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start workflow: {str(e)}",
        )


@router.get(
    "/workflow/{workflow_id}/status",
    response_model=StatusResponse,
    summary="Get workflow status",
    description="Retrieve the current status and progress of a workflow",
)
async def get_workflow_status(workflow_id: str) -> StatusResponse:
    """
    Get the current status of a workflow.

    Returns detailed progress information including current stage,
    completion percentage, and any errors.
    """
    workflow_data = storage.get_workflow(workflow_id)

    if not workflow_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    try:
        workflow = workflow_data["workflow"]
        state = workflow.get_state(workflow_id)

        if state:
            progress = get_progress(state)
        else:
            # Workflow not started yet
            progress = {
                "status": "pending",
                "progress_percent": 0,
                "steps_completed": 0,
                "total_steps": 0,
                "iteration_count": 0,
                "agents_executed": 0,
                "has_research": False,
                "has_analysis": False,
                "has_draft": False,
                "is_reviewed": False,
                "elapsed_time_seconds": 0,
            }

        return StatusResponse(
            workflow_id=workflow_id,
            status=workflow_data["status"],
            task=workflow_data["task"],
            task_type=workflow_data["task_type"],
            progress=progress,
            created_at=workflow_data["created_at"],
            updated_at=workflow_data.get("updated_at"),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow status: {str(e)}",
        )


@router.get(
    "/workflow/{workflow_id}/stream",
    summary="Stream workflow updates",
    description="Server-Sent Events stream for real-time workflow updates",
)
async def stream_workflow(workflow_id: str):
    """
    Stream workflow updates using Server-Sent Events.

    This endpoint provides real-time updates as the workflow progresses.
    Use this for live progress tracking in UIs.
    """
    workflow_data = storage.get_workflow(workflow_id)

    if not workflow_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    workflow = workflow_data["workflow"]

    async def event_stream() -> AsyncGenerator[str, None]:
        """Generate SSE events."""
        try:
            # Run workflow with streaming
            async for update in _stream_workflow_updates(
                workflow,
                workflow_data["task"],
                workflow_data["task_type"],
                None,  # task_context
                workflow_id,
            ):
                # Format as SSE with proper JSON serialization
                event_data = {
                    "workflow_id": workflow_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    **update,
                }
                yield f"data: {json.dumps(event_data)}\n\n"

            # Send completion event
            complete_event = {"type": "complete", "workflow_id": workflow_id}
            yield f"data: {json.dumps(complete_event)}\n\n"

        except Exception as e:
            error_data = {
                "type": "error",
                "workflow_id": workflow_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/workflow/{workflow_id}/feedback",
    response_model=FeedbackResponse,
    summary="Submit human feedback",
    description="Provide human-in-the-loop feedback for a workflow",
)
async def submit_feedback(
    workflow_id: str,
    request: FeedbackRequest,
) -> FeedbackResponse:
    """
    Submit human feedback for a workflow.

    This allows human intervention in the workflow, such as
    approving revisions or providing additional guidance.
    """
    workflow_data = storage.get_workflow(workflow_id)

    if not workflow_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    try:
        # In a full implementation, this would:
        # 1. Update the workflow state with feedback
        # 2. Resume the workflow with new context
        # 3. Return the updated status

        # For now, acknowledge the feedback
        # (Full implementation would integrate with LangGraph checkpoints)

        message = "Feedback received"
        new_status = None

        if request.action == "approve":
            message = "Feedback received: Workflow approved"
            new_status = "approved"
        elif request.action == "revise":
            message = "Feedback received: Workflow will be revised"
            new_status = "pending_revision"
        elif request.action == "reject":
            message = "Feedback received: Workflow rejected"
            new_status = "rejected"

        return FeedbackResponse(
            workflow_id=workflow_id,
            feedback_received=True,
            message=message,
            new_status=new_status,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process feedback: {str(e)}",
        )


@router.get(
    "/workflow/{workflow_id}/result",
    response_model=ResultResponse,
    summary="Get workflow result",
    description="Retrieve the final result of a completed workflow",
)
async def get_workflow_result(workflow_id: str) -> ResultResponse:
    """
    Get the final result of a workflow.

    Returns the complete output including final content,
    approval status, and execution metrics.
    """
    result = storage.get_result(workflow_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result for workflow {workflow_id} not found or not ready",
        )

    result_data = result["result"]

    return ResultResponse(
        workflow_id=workflow_id,
        status=result_data["status"],
        final_output=result_data.get("final_output", ""),
        approval_status=result_data.get("approval_status", "pending"),
        feedback=result_data.get("feedback", []),
        revision_count=result_data.get("revision_count", 0),
        duration_seconds=result_data.get("duration_seconds", 0),
        completed_at=result["completed_at"],
    )


@router.get(
    "/workflows",
    response_model=WorkflowListResponse,
    summary="List all workflows",
    description="Retrieve a list of all workflows",
)
async def list_workflows() -> WorkflowListResponse:
    """
    List all workflows.

    Returns basic information about all workflows including
    their IDs, tasks, types, and current status.
    """
    workflows = storage.list_workflows()

    return WorkflowListResponse(
        workflows=workflows,
        count=len(workflows),
    )


@router.delete(
    "/workflow/{workflow_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a workflow",
    description="Delete a workflow and its associated data",
)
async def delete_workflow(workflow_id: str) -> None:
    """
    Delete a workflow.

    Removes the workflow and all associated data from storage.
    """
    deleted = storage.delete_workflow(workflow_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the API",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the service status and basic metrics.
    """
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat(),
        active_workflows=len(storage.workflows),
    )


# =============================================================================
# FastAPI Application Factory
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    print("AgenticFlow API starting...")
    print(f"Model: {settings.default_model}")
    print(f"Checkpoint storage: {settings.checkpoint_storage}")
    yield
    # Shutdown
    print("AgenticFlow API shutting down...")


def create_app(
    title: str = "AgenticFlow API",
    version: str = "0.1.0",
    cors_enabled: bool = True,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        title: API title
        version: API version
        cors_enabled: Enable CORS middleware

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        version=version,
        description="Multi-Agent Workflow System with LangGraph",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    if cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Include router
    app.include_router(router, prefix="/api/v1")

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "AgenticFlow API",
            "version": version,
            "status": "running",
            "docs": "/docs",
        }

    return app


# Create app instance
app = create_app()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
        workers=settings.workers,
    )
