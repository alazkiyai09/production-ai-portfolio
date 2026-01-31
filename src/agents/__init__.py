"""
Agent definitions for AgenticFlow.

This module contains all agent implementations including:
- BaseAgent: Abstract base class for all agents
- PlannerAgent: Creates step-by-step execution plans
- ResearcherAgent: Gathers information from web and files
- AnalyzerAgent: Analyzes data and extracts insights
- WriterAgent: Creates written content
- ReviewerAgent: Reviews and evaluates output
"""

# Base class
from src.agents.specialized_agents import BaseAgent

# Specialized agents
from src.agents.specialized_agents import (
    PlannerAgent,
    ResearcherAgent,
    AnalyzerAgent,
    WriterAgent,
    ReviewerAgent,
)

# Factory functions
from src.agents.specialized_agents import (
    create_agent,
    create_all_agents,
)

__all__ = [
    # Base class
    "BaseAgent",

    # Specialized agents
    "PlannerAgent",
    "ResearcherAgent",
    "AnalyzerAgent",
    "WriterAgent",
    "ReviewerAgent",

    # Factory functions
    "create_agent",
    "create_all_agents",
]
