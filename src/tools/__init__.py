"""
Tool implementations for AgenticFlow.

This module contains all tool implementations including:
- Web search (Tavily, DuckDuckGo)
- File I/O operations
- Code execution
- Calculator
- Utility tools
"""

# Import StructuredTool objects for LangChain agents
from src.tools.agent_tools import (
    web_search as _web_search_tool,
    read_file as _read_file_tool,
    write_file as _write_file_tool,
    list_files as _list_files_tool,
    run_python_code as _run_python_code_tool,
    calculator as _calculator_tool,
    get_current_time as _get_current_time_tool,
    get_current_timestamp as _get_current_timestamp_tool,
    ALL_TOOLS,
    WEB_TOOLS,
    FILE_TOOLS,
    CODE_TOOLS,
    MATH_TOOLS,
    UTILITY_TOOLS,
    AGENT_TOOLS,
    get_tools_for_agent,
    get_tool_by_name,
    list_all_tools,
)

# Helper to call a tool with direct function syntax
def _call_tool(tool, args_dict):
    """Helper to call a StructuredTool with a dictionary of arguments."""
    return tool.invoke(args_dict)

# Create callable wrappers with the same signatures as the original functions
# These allow tests to call: calculator("2+2") instead of calculator.invoke({"expression": "2+2"})
def calculator(expression: str) -> float:
    """Calculator tool - callable for testing."""
    return _call_tool(_calculator_tool, {"expression": expression})

def read_file(file_path: str) -> str:
    """Read file tool - callable for testing."""
    return _call_tool(_read_file_tool, {"file_path": file_path})

def write_file(file_path: str, content: str) -> str:
    """Write file tool - callable for testing."""
    return _call_tool(_write_file_tool, {"file_path": file_path, "content": content})

def list_files(directory: str = ".") -> list:
    """List files tool - callable for testing."""
    return _call_tool(_list_files_tool, {"directory": directory})

def run_python_code(code: str) -> str:
    """Run Python code tool - callable for testing."""
    return _call_tool(_run_python_code_tool, {"code": code})

def web_search(query: str, num_results: int = 5) -> list:
    """Web search tool - callable for testing."""
    return _call_tool(_web_search_tool, {"query": query, "num_results": num_results})

def get_current_time(format: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """Get current time tool - callable for testing."""
    return _call_tool(_get_current_time_tool, {"format": format})

def get_current_timestamp() -> str:
    """Get current timestamp tool - callable for testing."""
    return _call_tool(_get_current_timestamp_tool, {})

__all__ = [
    # Callable functions (for testing and direct use)
    "web_search",
    "read_file",
    "write_file",
    "list_files",
    "run_python_code",
    "calculator",
    "get_current_time",
    "get_current_timestamp",

    # Tool collections
    "ALL_TOOLS",
    "WEB_TOOLS",
    "FILE_TOOLS",
    "CODE_TOOLS",
    "MATH_TOOLS",
    "UTILITY_TOOLS",
    "AGENT_TOOLS",

    # Helper functions
    "get_tools_for_agent",
    "get_tool_by_name",
    "list_all_tools",
]
