"""
Base agent implementation for AgenticFlow.

This module was previously a placeholder. The actual BaseAgent class
is implemented in specialized_agents.py. This file is kept for
backward compatibility but imports from the main module.
"""

# Re-export BaseAgent from specialized_agents for backward compatibility
from src.agents.specialized_agents import BaseAgent

__all__ = ["BaseAgent"]

