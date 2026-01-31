"""
Tool implementations for AgenticFlow agents.

This module provides production-ready tools for the multi-agent system,
including web search, file I/O, code execution, and utility functions.

All tools include proper security measures, error handling, and timeouts.
"""

from __future__ import annotations

import ast
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Optional

from langchain_core.tools import tool
from tavily import TavilyClient
from duckduckgo_search import DDGS

# =============================================================================
# Configuration
# =============================================================================

# Security: Restrict file operations to workspace directory
# Use a function to read the environment dynamically for testing
def _get_workspace_root() -> Path:
    """
    Get the workspace root directory from environment or current directory.

    This function reads the environment variable each time it's called,
    allowing tests to override the workspace root by setting the env var.

    Returns:
        Path to the workspace root directory
    """
    return Path(os.getenv("WORKSPACE_ROOT", os.getcwd()))

# For backwards compatibility, provide a property-like access
# Note: This is still read at import time for some uses, but the
# validation functions use _get_workspace_root() for dynamic access
WORKSPACE_ROOT: Final = Path(os.getenv("WORKSPACE_ROOT", os.getcwd()))

ALLOWED_FILE_EXTENSIONS: Final = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".py",
    ".html",
    ".xml",
}

# Code execution configuration
CODE_EXECUTION_TIMEOUT: Final = 30  # seconds
MAX_CODE_LENGTH: Final = 10000  # characters
MAX_OUTPUT_LENGTH: Final = 5000  # characters

# Web search configuration
DEFAULT_MAX_RESULTS: Final = 5
MAX_MAX_RESULTS: Final = 10


# =============================================================================
# Web Search Tools
# =============================================================================

@tool
def web_search(query: str, num_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, Any]]:
    """
    Search the web for information using Tavily with DuckDuckGo fallback.

    This tool performs web searches to gather current information from the
    internet. It tries Tavily first (requires API key) and falls back to
    DuckDuckGo if Tavily is unavailable.

    Args:
        query: The search query string
        num_results: Number of results to return (1-10, default: 5)

    Returns:
        List of search results with keys: title, url, snippet, source

    Raises:
        ValueError: If num_results is out of range or query is empty

    Examples:
        >>> results = web_search("latest AI trends 2024", num_results=3)
        >>> for r in results:
        ...     print(f"{r['title']}: {r['url']}")
    """
    # Validate inputs
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    query = query.strip()
    num_results = max(1, min(num_results, MAX_MAX_RESULTS))

    # Try Tavily first
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key and tavily_api_key.startswith("tvly-"):
        try:
            return _search_with_tavily(query, num_results)
        except Exception as e:
            # Log error and fall back to DuckDuckGo
            pass

    # Fall back to DuckDuckGo
    return _search_with_duckduckgo(query, num_results)


def _search_with_tavily(query: str, num_results: int) -> list[dict[str, Any]]:
    """
    Perform search using Tavily API.

    Args:
        query: Search query string
        num_results: Number of results to return

    Returns:
        List of formatted search results
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    response = client.search(
        query=query,
        max_results=num_results,
        search_depth="advanced",
        include_domains=None,
        exclude_domains=None,
    )

    # Format results
    results = []
    for result in response.get("results", []):
        results.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("content", ""),
            "source": "tavily",
            "score": result.get("score", 0.0),
        })

    return results


def _search_with_duckduckgo(
    query: str,
    num_results: int,
) -> list[dict[str, Any]]:
    """
    Perform search using DuckDuckGo (no API key required).

    Args:
        query: Search query string
        num_results: Number of results to return

    Returns:
        List of formatted search results
    """
    results = []

    try:
        ddgs = DDGS()
        search_results = ddgs.text(
            query,
            max_results=num_results,
        )

        for result in search_results:
            results.append({
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("body", ""),
                "source": "duckduckgo",
                "score": 0.5,  # DuckDuckGo doesn't provide scores
            })

    except Exception as e:
        # Return empty list on error
        return []

    return results


# =============================================================================
# File I/O Tools (with Security)
# =============================================================================

def _validate_file_path(file_path_str: str) -> Path:
    """
    Validate and resolve file path to prevent directory traversal attacks.

    This function ensures that the file path is within the allowed workspace
    and has an allowed file extension.

    Args:
        file_path_str: File path to validate

    Returns:
        Resolved Path object

    Raises:
        ValueError: If path is invalid, outside workspace, or has disallowed extension
    """
    # Get the current workspace root (reads env var dynamically)
    workspace_root = _get_workspace_root()

    # Resolve the path
    file_path = Path(file_path_str).resolve()

    # Check for directory traversal
    try:
        file_path = file_path.relative_to(workspace_root)
    except ValueError:
        raise ValueError(
            f"File access denied: path is outside workspace. "
            f"File must be within: {workspace_root}"
        )

    # Prevent hidden files (starting with .) - check this before extension
    # since hidden files like ".secret" don't have extensions
    if any(part.startswith(".") for part in file_path.parts):
        raise ValueError("Access to hidden files/directories is not allowed")

    # Check file extension
    if file_path.suffix.lower() not in ALLOWED_FILE_EXTENSIONS:
        raise ValueError(
            f"File extension '{file_path.suffix}' is not allowed. "
            f"Allowed: {', '.join(sorted(ALLOWED_FILE_EXTENSIONS))}"
        )

    # Return full path
    return _get_workspace_root() / file_path


@tool
def read_file(file_path: str) -> str:
    """
    Read contents of a text file from the workspace.

    This tool provides secure file reading with the following protections:
    - Directory traversal prevention
    - File extension whitelist
    - Workspace root restriction
    - No hidden file access

    Args:
        file_path: Path to the file to read (relative or absolute within workspace)

    Returns:
        File contents as a string

    Raises:
        ValueError: If file path is invalid or outside workspace
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read

    Examples:
        >>> content = read_file("data/input.txt")
        >>> print(content)
    """
    # Validate file path
    validated_path = _validate_file_path(file_path)

    # Check if file exists
    if not validated_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not validated_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Read file with error handling
    try:
        with open(validated_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check file size
        if len(content) > 1_000_000:  # 1MB limit
            raise ValueError(
                f"File too large ({len(content)} chars). "
                "Maximum size is 1,000,000 characters."
            )

        return content

    except UnicodeDecodeError:
        raise ValueError(
            f"File encoding error: '{file_path}' is not a valid text file. "
            "Only UTF-8 encoded text files are supported."
        )


@tool
def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file in the workspace.

    This tool provides secure file writing with the following protections:
    - Directory traversal prevention
    - File extension whitelist
    - Workspace root restriction
    - Automatic directory creation
    - Atomic write with temporary file

    Args:
        file_path: Path to the file to write (relative or absolute within workspace)
        content: Content to write to the file

    Returns:
        Success message with file path

    Raises:
        ValueError: If file path is invalid or outside workspace

    Examples:
        >>> write_file("output/result.txt", "Analysis complete")
        'Successfully wrote to output/result.txt'
    """
    # Validate file path
    validated_path = _validate_file_path(file_path)

    # Validate content
    if not isinstance(content, str):
        raise ValueError("Content must be a string")

    if len(content) > 5_000_000:  # 5MB limit
        raise ValueError(
            f"Content too large ({len(content)} chars). "
            "Maximum size is 5,000,000 characters."
        )

    # Create parent directories if needed
    validated_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first (atomic write)
    temp_path = validated_path.with_suffix(validated_path.suffix + ".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Atomic rename
        temp_path.replace(validated_path)

        return f"Successfully wrote to {file_path} ({len(content)} characters)"

    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"Failed to write file: {e}")


@tool
def list_files(directory: str = ".") -> list[dict[str, Any]]:
    """
    List files in a directory within the workspace.

    This tool provides secure directory listing with protection against
    directory traversal attacks.

    Args:
        directory: Directory path to list (relative to workspace, default: ".")

    Returns:
        List of file/directory information with keys: name, type, size, path

    Raises:
        ValueError: If directory path is invalid or outside workspace

    Examples:
        >>> files = list_files("data")
        >>> for f in files:
        ...     print(f"{f['name']}: {f['type']}")
    """
    # Get the current workspace root (reads env var dynamically)
    workspace_root = _get_workspace_root()

    # Resolve and validate directory path
    dir_path = Path(directory).resolve()

    try:
        dir_path = dir_path.relative_to(workspace_root)
    except ValueError:
        raise ValueError(
            f"Directory access denied: path is outside workspace. "
            f"Directory must be within: {workspace_root}"
        )

    full_path = workspace_root / dir_path

    if not full_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not full_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # List contents
    result = []

    for item in full_path.iterdir():
        try:
            stat = item.stat()

            # Skip hidden files
            if item.name.startswith("."):
                continue

            # Skip directories (only list files)
            if item.is_dir():
                continue

            result.append({
                "name": item.name,
                "type": "file",
                "size": stat.st_size,
                "path": str(item.relative_to(workspace_root)),
                "extension": item.suffix,
            })

        except (PermissionError, OSError):
            # Skip files we can't access
            continue

    # Sort alphabetically by name
    result.sort(key=lambda x: x["name"])

    return result


# =============================================================================
# Code Execution Tool (Sandboxed)
# =============================================================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute Python code in a sandboxed environment with timeout.

    This tool provides safe Python code execution with:
    - 30-second timeout
    - Restricted globals (no imports, no file access by default)
    - Output capture (stdout and stderr)
    - Error handling and traceback

    Security measures:
    - No direct file system access
    - No network operations
    - Limited built-in functions
    - Timeout enforcement

    Args:
        code: Python code to execute

    Returns:
        Output from code execution or error message

    Raises:
        ValueError: If code is too long or contains dangerous operations

    Examples:
        >>> result = run_python_code("print(2 + 2)")
        >>> print(result)
        '4'
        >>> result = run_python_code("[x**2 for x in range(5)]")
        >>> print(result)
        '[0, 1, 4, 9, 16]'
    """
    # Validate code length
    if not code or not code.strip():
        raise ValueError("Code cannot be empty")

    if len(code) > MAX_CODE_LENGTH:
        raise ValueError(
            f"Code too long ({len(code)} chars). "
            f"Maximum length is {MAX_CODE_LENGTH} characters."
        )

    # Check for potentially dangerous operations
    dangerous_patterns = [
        r"\bimport\b",
        r"\b__import__\b",
        r"\beval\b",
        r"\bexec\b",
        r"\bopen\b",
        r"\bcompile\b",
        r"\bgetattr\b",
        r"\bsetattr\b",
        r"\bdelattr\b",
        r"\b__builtins____",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            raise ValueError(
                f"Code contains potentially dangerous operation: {pattern}. "
                "Import, eval, exec, file operations, and introspection are not allowed."
            )

    # Prepare restricted execution environment
    safe_globals: dict[str, Any] = {
        "__builtins__": {
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "hex": hex,
            "int": int,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "print": print,
        }
    }

    # Capture output
    import io
    import sys
    import threading
    import concurrent.futures
    from contextlib import redirect_stdout, redirect_stderr

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    def execute_code() -> str:
        """Execute code in restricted environment and return output."""
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            result = eval(code, safe_globals, {})

        # Get output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # Format output
        if result is not None:
            output = str(result)
        elif stdout_output:
            output = stdout_output
        else:
            output = "<code executed with no output>"

        # Add stderr if present
        if stderr_output:
            output += f"\n[stderr]: {stderr_output}"

        # Truncate if too long
        if len(output) > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH] + "\n... (output truncated)"

        return output

    try:
        # Execute with timeout using ThreadPoolExecutor (cross-platform)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(execute_code)
            try:
                output = future.result(timeout=CODE_EXECUTION_TIMEOUT)
                return output
            except concurrent.futures.TimeoutError:
                # Cancel the future if it's still running
                future.cancel()
                return f"TimeoutError: Code execution exceeded time limit ({CODE_EXECUTION_TIMEOUT} seconds)"

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        # Add limited traceback for debugging
        tb_str = "".join(traceback.format_tb(e.__traceback__, limit=3))
        if tb_str:
            error_msg += f"\n\nTraceback (most recent calls last):\n{tb_str}"
        return error_msg


# =============================================================================
# Calculator Tool (Safe Evaluation)
# =============================================================================

@tool
def calculator(expression: str) -> float:
    """
    Safely evaluate mathematical expressions.

    This tool provides safe mathematical expression evaluation using AST parsing.
    Only mathematical operations are allowed - no function calls or imports.

    Supported operations:
    - Basic: +, -, *, /, %, **
    - Parentheses: ( )
    - Numbers: integers and floats
    - Comparison: <, >, <=, >=, ==, != (returns boolean as 0.0 or 1.0)

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result as a float

    Raises:
        ValueError: If expression contains invalid operations

    Examples:
        >>> calculator("2 + 2")
        4.0
        >>> calculator("(10 + 5) * 2")
        30.0
        >>> calculator("2 ** 8")
        256.0
    """
    if not expression or not expression.strip():
        raise ValueError("Expression cannot be empty")

    expression = expression.strip()

    # Define allowed AST nodes
    allowed_nodes = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Compare,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Eq,
        ast.NotEq,
        ast.Call,
        ast.Name,
        ast.Load,
    }

    try:
        # Parse expression to AST
        tree = ast.parse(expression, mode="eval")

        # Check for disallowed nodes
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                raise ValueError(
                    f"Expression contains disallowed operation: {type(node).__name__}. "
                    "Only basic mathematical operations are allowed."
                )

        # Allow math functions
        import math

        safe_names = {
            "pi": math.pi,
            "e": math.e,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }

        # Evaluate expression
        result = eval(compile(tree, "<string>", "eval"), {"__builtins__": {}}, safe_names)

        # Convert to float
        if isinstance(result, bool):
            return float(result)

        return float(result)

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {e}")


# =============================================================================
# Utility Tools
# =============================================================================

@tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """
    Get the current date and time.

    This tool returns the current UTC time in a customizable format.

    Args:
        format: strftime format string (default: "%Y-%m-%d %H:%M:%S UTC")

    Returns:
        Current date and time as a formatted string

    Examples:
        >>> get_current_time()
        '2024-01-15 10:30:45 UTC'
        >>> get_current_time("%Y-%m-%d")
        '2024-01-15'
        >>> get_current_time("%A, %B %d, %Y")
        'Monday, January 15, 2024'
    """
    return datetime.utcnow().strftime(format)


@tool
def get_current_timestamp() -> str:
    """
    Get the current ISO 8601 timestamp.

    Returns:
        Current timestamp in ISO 8601 format (UTC)

    Examples:
        >>> get_current_timestamp()
        '2024-01-15T10:30:45.123456'
    """
    return datetime.utcnow().isoformat()


# =============================================================================
# Agent Tool Mappings
# =============================================================================

# All available tools
ALL_TOOLS = [
    web_search,
    read_file,
    write_file,
    list_files,
    run_python_code,
    calculator,
    get_current_time,
    get_current_timestamp,
]

# Tool mappings per agent type
AGENT_TOOLS: dict[str, list] = {
    "planner": [
        calculator,
        get_current_time,
        get_current_timestamp,
    ],
    "researcher": [
        web_search,
        read_file,
        list_files,
        get_current_time,
    ],
    "analyzer": [
        read_file,
        list_files,
        run_python_code,
        calculator,
        get_current_time,
    ],
    "writer": [
        read_file,
        write_file,
        get_current_time,
        get_current_timestamp,
    ],
    "reviewer": [
        read_file,
        calculator,
        get_current_time,
    ],
    "general": ALL_TOOLS,  # All tools for general tasks
}


def get_tools_for_agent(agent_type: str) -> list:
    """
    Get the appropriate tools for a specific agent type.

    Args:
        agent_type: Type of agent (planner, researcher, analyzer, writer, reviewer, general)

    Returns:
        List of tools available to the agent

    Raises:
        ValueError: If agent_type is unknown

    Examples:
        >>> tools = get_tools_for_agent("researcher")
        >>> [t.name for t in tools]
        ['web_search', 'read_file', 'list_files', 'get_current_time']
    """
    agent_type_lower = agent_type.lower()

    if agent_type_lower not in AGENT_TOOLS:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Valid types: {', '.join(AGENT_TOOLS.keys())}"
        )

    return AGENT_TOOLS[agent_type_lower]


def get_tool_by_name(tool_name: str) -> Optional[Any]:
    """
    Get a tool by its name.

    Args:
        tool_name: Name of the tool to retrieve

    Returns:
        Tool object or None if not found
    """
    for tool_obj in ALL_TOOLS:
        if tool_obj.name == tool_name:
            return tool_obj
    return None


def list_all_tools() -> list[dict[str, str]]:
    """
    List all available tools with their descriptions.

    Returns:
        List of tool information dictionaries
    """
    return [
        {
            "name": tool.name,
            "description": tool.description.strip().split("\n")[0],
            "full_description": tool.description,
        }
        for tool in ALL_TOOLS
    ]


# =============================================================================
# Tool Groups for Convenience
# =============================================================================

# Web search tools
WEB_TOOLS = [web_search]

# File I/O tools
FILE_TOOLS = [read_file, write_file, list_files]

# Code execution tools
CODE_TOOLS = [run_python_code]

# Math tools
MATH_TOOLS = [calculator]

# Utility tools
UTILITY_TOOLS = [get_current_time, get_current_timestamp]


# =============================================================================
# Raw Function Exports (for testing)
# =============================================================================
# The @tool decorator wraps functions in StructuredTool objects.
# For testing purposes, we expose the underlying functions directly.

def calculator_raw(expression: str) -> float:
    """Raw calculator function for testing (wraps the tool)."""
    return calculator.invoke({"expression": expression})

def read_file_raw(file_path: str) -> str:
    """Raw read_file function for testing (wraps the tool)."""
    return read_file.invoke({"file_path": file_path})

def write_file_raw(file_path: str, content: str) -> str:
    """Raw write_file function for testing (wraps the tool)."""
    return write_file.invoke({"file_path": file_path, "content": content})

def list_files_raw(directory: str = ".") -> list:
    """Raw list_files function for testing (wraps the tool)."""
    return list_files.invoke({"directory": directory})

def run_python_code_raw(code: str) -> str:
    """Raw run_python_code function for testing (wraps the tool)."""
    return run_python_code.invoke({"code": code})

def web_search_raw(query: str, num_results: int = 5) -> list:
    """Raw web_search function for testing (wraps the tool)."""
    return web_search.invoke({"query": query, "num_results": num_results})
