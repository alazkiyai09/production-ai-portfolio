# AgenticFlow Project Structure

```
agenticflow/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── pyproject.toml            # Project metadata and tool config
├── requirements.txt          # Python dependencies
├── PROJECT_STRUCTURE.md      # This file
├── README.md                 # Project documentation
│
├── data/                     # Runtime data directory
│   └── checkpoints.db        # SQLite checkpoint storage (if using sqlite)
│
├── logs/                     # Application logs
│   └── agenticflow.log
│
├── src/                      # Source code
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management with Pydantic
│   │
│   ├── agents/              # Agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py    # Base agent class
│   │   ├── planner.py       # Planner agent
│   │   ├── researcher.py    # Researcher agent
│   │   ├── analyzer.py      # Analyzer agent
│   │   ├── writer.py        # Writer agent
│   │   └── reviewer.py      # Reviewer agent
│   │
│   ├── tools/               # Tool implementations
│   │   ├── __init__.py
│   │   ├── search.py        # Web search tools (Tavily, DuckDuckGo)
│   │   ├── file_io.py       # File I/O tools
│   │   ├── code_executor.py # Code execution tool
│   │   └── custom_tools.py  # Custom utility tools
│   │
│   ├── workflows/           # LangGraph workflow definitions
│   │   ├── __init__.py
│   │   ├── graph.py         # Main workflow graph
│   │   ├── router.py        # Conditional routing logic
│   │   └── checkpoints.py   # Checkpoint configuration
│   │
│   ├── state/               # State definitions
│   │   ├── __init__.py
│   │   ├── workflow_state.py # Workflow state (TypedDict)
│   │   ├── agent_state.py    # Individual agent states
│   │   └── messages.py       # Message types
│   │
│   ├── api/                 # FastAPI endpoints
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI app factory
│   │   ├── routes/          # API route modules
│   │   │   ├── __init__.py
│   │   │   ├── workflows.py # Workflow submission/status
│   │   │   └── health.py    # Health check endpoints
│   │   └── models/          # Pydantic request/response models
│   │       ├── __init__.py
│   │       └── schemas.py
│   │
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── logging.py       # Logging configuration
│       ├── errors.py        # Error handlers
│       ├── retry.py         # Retry logic
│       └── formatters.py    # Output formatting
│
└── tests/                   # Test suite
    ├── __init__.py
    ├── conftest.py          # Pytest configuration and fixtures
    ├── test_agents/         # Agent tests
    │   ├── __init__.py
    │   ├── test_planner.py
    │   ├── test_researcher.py
    │   ├── test_analyzer.py
    │   ├── test_writer.py
    │   └── test_reviewer.py
    ├── test_tools/          # Tool tests
    │   ├── __init__.py
    │   ├── test_search.py
    │   ├── test_file_io.py
    │   └── test_code_executor.py
    ├── test_workflows/      # Workflow tests
    │   ├── __init__.py
    │   └── test_graph.py
    ├── test_api/            # API tests
    │   ├── __init__.py
    │   └── test_main.py
    └── test_utils/          # Utility tests
        ├── __init__.py
        └── test_retry.py
```

## Key Files Description

### Configuration
- **`.env.example`**: Template for environment variables. Copy to `.env` and fill in API keys.
- **`pyproject.toml`**: Project metadata, dependencies, and tool configurations (black, isort, mypy, pytest).
- **`requirements.txt`**: Pinned versions of all Python dependencies.
- **`src/config.py`**: Pydantic BaseSettings for configuration management.

### Core Components

#### 1. **Agents** (`src/agents/`)
- `base_agent.py`: Base class with common agent functionality
- `planner.py`: Breaks down tasks into steps
- `researcher.py`: Web search and document retrieval
- `analyzer.py`: Data analysis and pattern identification
- `writer.py`: Content generation
- `reviewer.py`: Quality evaluation and feedback

#### 2. **Tools** (`src/tools/`)
- `search.py`: Tavily and DuckDuckGo search integration
- `file_io.py`: Read/write operations
- `code_executor.py`: Safe Python code execution
- `custom_tools.py`: Domain-specific utilities

#### 3. **Workflows** (`src/workflows/`)
- `graph.py`: LangGraph StateGraph definition with edges and nodes
- `router.py`: Conditional routing logic between agents
- `checkpoints.py`: Checkpoint configuration for persistence

#### 4. **State** (`src/state/`)
- `workflow_state.py`: TypedDict for workflow state
- `agent_state.py`: Individual agent state schemas
- `messages.py`: Message type definitions

#### 5. **API** (`src/api/`)
- `main.py`: FastAPI application factory
- `routes/`: API endpoint implementations
- `models/`: Pydantic schemas for requests/responses

#### 6. **Utils** (`src/utils/`)
- `logging.py`: Loguru configuration
- `errors.py`: Custom exception classes
- `retry.py`: Exponential backoff retry logic
- `formatters.py`: Output formatting helpers

### Testing
- Unit tests for each agent, tool, and workflow component
- Integration tests for API endpoints
- Pytest fixtures for common test data
- Coverage reporting configured
