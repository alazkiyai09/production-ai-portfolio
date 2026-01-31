"""
Tool utilities and helper functions.

This module provides utilities for managing tool registries,
creating custom tools, and handling tool execution.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool


def create_tool_registry() -> dict[str, BaseTool]:
    """
    Create a registry of all available tools.

    Returns:
        Dictionary mapping tool names to tool instances
    """
    from .customer_tools import (
        get_customer_profile_tool,
        get_customer_risk_history_tool,
    )
    from .device_tools import check_ip_reputation_tool, get_device_fingerprint_tool
    from .transaction_tools import (
        analyze_transaction_patterns_tool,
        get_transaction_by_id_tool,
        get_transaction_history_tool,
    )

    return {
        # Transaction tools
        "get_transaction_history": get_transaction_history_tool,
        "get_transaction_by_id": get_transaction_by_id_tool,
        "analyze_transaction_patterns": analyze_transaction_patterns_tool,
        # Customer tools
        "get_customer_profile": get_customer_profile_tool,
        "get_customer_risk_history": get_customer_risk_history_tool,
        # Device tools
        "get_device_fingerprint": get_device_fingerprint_tool,
        "check_ip_reputation": check_ip_reputation_tool,
    }


def get_tool_descriptions() -> dict[str, str]:
    """
    Get descriptions of all available tools.

    Returns:
        Dictionary mapping tool names to descriptions
    """
    registry = create_tool_registry()
    return {name: tool.description for name, tool in registry.items()}


def get_tools_by_category() -> dict[str, list[str]]:
    """
    Group tools by category.

    Returns:
        Dictionary mapping categories to lists of tool names
    """
    return {
        "transaction": [
            "get_transaction_history",
            "get_transaction_by_id",
            "analyze_transaction_patterns",
        ],
        "customer": [
            "get_customer_profile",
            "get_customer_risk_history",
        ],
        "device": [
            "get_device_fingerprint",
            "check_ip_reputation",
        ],
    }


async def execute_tool_safely(
    tool: BaseTool,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Execute a tool with error handling.

    Args:
        tool: Tool instance to execute
        **kwargs: Tool arguments

    Returns:
        Tool result with error handling
    """
    try:
        result = await tool.ainvoke(kwargs)
        return {"success": True, "result": result}
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool_name": tool.name,
        }
