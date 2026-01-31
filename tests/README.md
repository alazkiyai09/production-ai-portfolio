# AgenticFlow Test Suite

Comprehensive test suite for the AgenticFlow multi-agent system.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_agenticflow.py      # Main test suite (40+ tests)
└── README.md                # This file
```

## Test Categories

### 1. State Tests (13 tests)
- `test_create_initial_state_*` - State creation variations
- `test_update_state_*` - State updates with agent tracking
- `test_is_complete_*` - Completion status checks
- `test_is_error_*` - Error status checks
- `test_needs_revision_*` - Revision status checks
- `test_get_progress_*` - Progress metrics
- `test_advance_step` - Step progression
- `test_mark_complete` - Workflow completion
- `test_mark_error` - Error marking

### 2. Tool Tests (16 tests)
- **Calculator** (3 tests)
  - Basic arithmetic operations
  - Complex expressions
  - Invalid expression rejection
- **File Operations** (7 tests)
  - Read file success
  - File not found
  - Directory traversal prevention
  - Hidden file blocking
  - Write file success
  - Directory traversal prevention (write)
  - Auto-directory creation
  - List files
  - Hidden file exclusion
- **Code Execution** (3 tests)
  - Basic Python code
  - Print statements
  - Timeout handling
  - Security (import blocking)
  - Security (eval blocking)
- **Search** (1 test)
  - Tavily integration
- **Tool Discovery** (2 tests)
  - Valid agent tools
  - Invalid agent type

### 3. Agent Tests (12 tests)
- **Planner** (2 tests)
  - Agent creation
  - System prompt
- **Researcher** (2 tests)
  - Agent creation
  - Search tools
- **Analyzer** (2 tests)
  - Agent creation
  - Computational tools
- **Writer** (2 tests)
  - Agent creation
  - File I/O tools
- **Reviewer** (1 test)
  - Agent creation
- **Factory** (2 tests)
  - Valid agent types
  - Invalid agent type
- **Execution** (1 test)
  - Agent invocation with mock LLM
- **LLM Creation** (1 test)
  - OpenAI LLM initialization

### 4. Workflow Tests (10 tests)
- **Creation** (2 tests)
  - Basic workflow
  - Custom settings
- **Graph Structure** (2 tests)
  - All nodes present
  - Entry point
- **Execution** (1 test)
  - Run with mocks
- **Checkpointing** (1 test)
  - Checkpoint saver creation
- **Routing** (4 tests)
  - Approved → END
  - Needs revision → Writer
  - Rejected → END
  - Max revisions exceeded → Reject

### 5. API Tests (6 tests)
- **Storage** (6 tests)
  - Add workflow
  - Get workflow
  - Update status
  - Set result
  - List workflows
  - Delete workflow
  - Get non-existent
  - Delete non-existent

### 6. Integration Tests (3 tests)
- State to agent flow
- Tool to agent integration
- Complete workflow structure
- Research workflow with search

**Total: 60+ test cases**

## Running Tests

### Run All Tests
```bash
# Basic run
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Verbose output
pytest tests/ -v

# With short tracebacks
pytest tests/ -v --tb=short
```

### Run Specific Test Categories
```bash
# State tests only
pytest tests/test_agenticflow.py::TestStateManagement -v

# Tool tests only
pytest tests/test_agenticflow.py::TestTools -v

# Agent tests only
pytest tests/test_agenticflow.py::TestAgents -v

# Workflow tests only
pytest tests/test_agenticflow.py::TestWorkflow -v

# API tests only
pytest tests/test_agenticflow.py::TestAPI -v

# Integration tests only
pytest tests/test_agenticflow.py::TestIntegration -v
```

### Run Specific Test
```bash
# Single test
pytest tests/test_agenticflow.py::TestStateManagement::test_create_initial_state_basic -v

# All tests matching pattern
pytest tests/test_agenticflow.py -k "test_create_initial_state" -v
```

### Run with Markers
```bash
# Unit tests only
pytest tests/ -m unit -v

# Integration tests only
pytest tests/ -m integration -v

# Skip slow tests
pytest tests/ -m "not slow" -v

# API tests only
pytest tests/ -m api -v
```

## Test Fixtures

Available fixtures in `conftest.py`:

| Fixture | Description |
|---------|-------------|
| `temp_workspace` | Temporary directory for file operations |
| `sample_task` | Sample task description |
| `sample_state` | Sample workflow state |
| `mock_llm` | Mock LLM instance |
| `mock_tavily_response` | Mock Tavily API response |
| `populated_state` | State with sample data |
| `completed_state` | Completed workflow state |
| `error_state` | Error workflow state |
| `mock_openai_response` | Mock OpenAI API response |
| `sample_research_data` | Sample research results |
| `sample_agent_output` | Sample agent output |
| `mock_llm_response` | Helper for creating mock responses |

## Coverage

Generate coverage report:

```bash
# HTML report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Terminal report
pytest tests/ --cov=src --cov-report=term-missing

# Combined
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov pytest-asyncio
      - run: pytest tests/ --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v3
```

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: Run tests
        entry: pytest tests/ -v
        language: system
        pass_filenames: false
        always_run: true
```

## Test Configuration

Pytest configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = [
    "-ra",
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing",
]
```

## Writing New Tests

1. **Add test to appropriate class** in `test_agenticflow.py`
2. **Use descriptive names**: `test_what_is_being_tested`
3. **Follow AAA pattern**: Arrange, Act, Assert
4. **Use fixtures** for common setup
5. **Mock external dependencies** (APIs, LLMs)
6. **Add docstrings** explaining what's being tested

Example:

```python
def test_my_feature(self, sample_state):
    """
    Test that my feature works correctly.

    Given: A sample state
    When: My feature is called
    Then: The result should be as expected
    """
    # Arrange
    input_data = sample_state

    # Act
    result = my_feature(input_data)

    # Assert
    assert result is not None
    assert result.status == "success"
```

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: Make sure you're running from the project root and src is in your PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### API Key Errors
```
OPENAI_API_KEY not found
```
**Solution**: Tests use mock API keys from `conftest.py`. Make sure it's being loaded:
```bash
pytest tests/ -c tests/conftest.py
```

### Fixture Not Found
```
fixture 'temp_workspace' not found
```
**Solution**: Ensure `conftest.py` is in the tests directory.

### Slow Tests
Some tests are marked as `slow`. Skip them:
```bash
pytest tests/ -m "not slow"
```

## Test Statistics

Current test count:
- State Tests: 13
- Tool Tests: 16
- Agent Tests: 12
- Workflow Tests: 10
- API Tests: 7
- Integration Tests: 4
- **Total: 62+ tests**

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure all tests pass
3. Add integration tests for workflows
4. Update this README
5. Maintain >80% code coverage
