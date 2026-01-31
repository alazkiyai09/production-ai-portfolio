"""
Prompt template management system.

This module provides comprehensive template management with versioning,
variable extraction, validation, and Jinja2-based rendering.
"""

from src.prompt_optimizer.templates.template_manager import (
    PromptTemplate,
    RenderedPrompt,
    TemplateValidationResult,
    TemplateManager,
    create_template_manager,
    TEMPLATE_PATTERNS,
)
from src.prompt_optimizer.templates.jinja_env import (
    create_environment,
    get_template_component,
    TEMPLATE_COMPONENTS,
    COMMON_VARIABLES,
)

__all__ = [
    "PromptTemplate",
    "RenderedPrompt",
    "TemplateValidationResult",
    "TemplateManager",
    "create_template_manager",
    "TEMPLATE_PATTERNS",
    "create_environment",
    "get_template_component",
    "TEMPLATE_COMPONENTS",
    "COMMON_VARIABLES",
]
