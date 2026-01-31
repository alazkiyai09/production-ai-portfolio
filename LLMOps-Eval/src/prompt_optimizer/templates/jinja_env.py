"""
Jinja2 environment configuration for prompt templates.

This module provides a configured Jinja2 environment with custom filters
and functions for prompt rendering.
"""

from jinja2 import Environment, BaseLoader
from typing import Any, Dict, List
import json
import re


def format_datetime(value: Any, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime value."""
    if hasattr(value, "strftime"):
        return value.strftime(format_str)
    return str(value)


def format_json(value: Any, indent: int = 2) -> str:
    """Format value as JSON string."""
    return json.dumps(value, indent=indent, ensure_ascii=False)


def format_number(value: Any, precision: int = 2) -> str:
    """Format numeric value with precision."""
    try:
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return str(value)


def truncate_words(value: str, num_words: int = 20) -> str:
    """Truncate text to specified number of words."""
    words = str(value).split()
    if len(words) <= num_words:
        return str(value)
    return " ".join(words[:num_words]) + "..."


def clean_whitespace(value: str) -> str:
    """Clean up extra whitespace in text."""
    return re.sub(r"\s+", " ", value.strip())


def escape_quotes(value: str) -> str:
    """Escape quotes for use in prompts."""
    return value.replace('"', '\\"')


def format_examples(examples: List[Dict[str, str]]) -> str:
    """Format few-shot examples for prompts."""
    parts = []
    for i, example in enumerate(examples, 1):
        if "input" in example and "output" in example:
            parts.append(f"{i}. Input: {example['input']}\n   Output: {example['output']}")
        elif "question" in example and "answer" in example:
            parts.append(f"{i}. Q: {example['question']}\n   A: {example['answer']}")
        else:
            parts.append(f"{i}. {example}")
    return "\n".join(parts)


def get_token_count_estimate(value: str) -> int:
    """Estimate token count (rough approximation: ~4 chars per token)."""
    return len(str(value)) // 4


def create_environment(
    loader: BaseLoader = None,
    trim_blocks: bool = True,
    lstrip_blocks: bool = True,
) -> Environment:
    """
    Create a configured Jinja2 environment for prompt templates.

    Args:
        loader: Template loader
        trim_blocks: Trim whitespace from blocks
        lstrip_blocks: Strip whitespace from left side of blocks

    Returns:
        Configured Jinja2 Environment
    """
    env = Environment(
        loader=loader or BaseLoader(),
        trim_blocks=trim_blocks,
        lstrip_blocks=lstrip_blocks,
    )

    # Add custom filters
    env.filters["datetime"] = format_datetime
    env.filters["json"] = format_json
    env.filters["number"] = format_number
    env.filters["truncate"] = truncate_words
    env.filters["clean"] = clean_whitespace
    env.filters["escape_quotes"] = escape_quotes
    env.filters["examples"] = format_examples
    env.filters["token_estimate"] = get_token_count_estimate

    return env


# Pre-defined template components
TEMPLATE_COMPONENTS = {
    "system_instruction": """You are {{ role }}.

Your task is to {{ task }}.

{% if constraints %}
Constraints:
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}""",

    "output_instruction": """Provide your response as {{ format }}.

{% if format == 'json' %}
Use the following JSON format:
```json
{{ schema }}
```
{% elif format == 'code' %}
Use the {{ language }} programming language.
{% endif %}""",

    "cot_instruction": """Let's approach this step by step:

1. First, understand what's being asked
2. Gather the relevant information
3. Apply the appropriate reasoning
4. Formulate a clear response

{{ reasoning_prompt }}""",

    "few_shot_header": """Here are some examples to guide your response:

{{ examples }}

Now, for your task:""",

    "safety_reminder": """Remember to:
- Avoid harmful content
- Don't provide information that could be used maliciously
- If the request is unsafe, politely decline""",

    "confidence_request": """Provide your confidence level in your answer.
Use the format: [Your answer]

Confidence: X% (where X is 0-100)""",
}


# Common template variables
COMMON_VARIABLES = {
    "question": "The question to answer",
    "instruction": "The instruction or task",
    "context": "Background information or context",
    "output_format": "Desired output format",
    "examples": "Few-shot examples",
    "language": "Programming language",
    "schema": "JSON schema",
    "role": "AI assistant role",
    "task": "The task to perform",
    "constraints": "List of constraints",
    "reasoning_prompt": "Custom reasoning prompt",
}


def get_template_component(name: str, variables: Dict[str, Any] = None) -> str:
    """
    Get a pre-defined template component with variables filled in.

    Args:
        name: Component name
        variables: Variables to substitute

    Returns:
        Rendered component string
    """
    if name not in TEMPLATE_COMPONENTS:
        raise ValueError(f"Unknown component: {name}")

    component = TEMPLATE_COMPONENTS[name]

    # Quick and dirty variable substitution (for components without Jinja2)
    if variables:
        for var_name, var_value in variables.items():
            component = component.replace(f"{{{{{var_name}}}}}", str(var_value))

    return component


__all__ = [
    "create_environment",
    "get_template_component",
    "TEMPLATE_COMPONENTS",
    "COMMON_VARIABLES",
]
