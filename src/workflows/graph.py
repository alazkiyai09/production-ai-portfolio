"""
Legacy alias for backward compatibility.

This module re-exports the ResearchWorkflow for backward compatibility.
New code should import from src.workflows.research_workflow.
"""

from src.workflows.research_workflow import ResearchWorkflow, create_workflow

__all__ = ["ResearchWorkflow", "create_workflow"]
