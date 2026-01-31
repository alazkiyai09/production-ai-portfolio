"""
LLMOps-Eval: Production-ready LLM evaluation and deployment pipeline.

A comprehensive framework for evaluating, testing, and monitoring LLM systems
with support for multiple providers, custom metrics, and automated testing.
"""

__version__ = "0.1.0"
__author__ = "AI Engineer"
__license__ = "MIT"

from src.config import get_settings, settings

__all__ = ["__version__", "get_settings", "settings"]
