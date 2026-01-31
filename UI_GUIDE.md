# AgenticFlow UI - Quick Start Guide

## Starting the Application

### 1. Start the Backend API

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Start the FastAPI backend
python src/api/main.py
# Or: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 2. Start the Streamlit UI

In a new terminal:

```bash
# Run Streamlit
streamlit run src/ui/app.py

# Or with custom port
streamlit run src/ui/app.py --server.port 8501
```

The UI will be available at `http://localhost:8501`

## Features

### Sidebar
- **Task Input**: Enter your task description
- **Task Type Selector**: Choose general, research, analysis, or content_creation
- **Advanced Settings**:
  - Model selection (gpt-4o-mini, gpt-4o, etc.)
  - Temperature control
  - Max iterations for revisions
  - Additional context
- **API Status**: Real-time connection status
- **Recent Workflows**: Quick access to previous workflows

### Main Area
- **Progress Bar**: Visual workflow progress (0-100%)
- **Metrics Dashboard**:
  - Steps completed
  - Iterations
  - Agents executed
  - Research status
  - Elapsed time
- **Agent Timeline**: Visual execution timeline
- **Agent Output Cards**: Expandable sections for each agent:
  - 📋 Planner - Execution plan
  - 🔍 Researcher - Gathered information
  - 📊 Analyzer - Insights and findings
  - ✍️ Writer - Content drafts
  - 👁️ Reviewer - Evaluation and feedback
- **Final Result**: Complete output with download option
- **Feedback Section**: Reviewer feedback items

## Example Workflows

### Research Task
```
Task: Research the latest developments in quantum computing
Type: research
Model: gpt-4o-mini
```

### Analysis Task
```
Task: Analyze the benefits and drawbacks of remote work
Type: analysis
Model: gpt-4o
```

### Content Creation
```
Task: Write a blog post about AI in healthcare
Type: content_creation
Model: gpt-4o
Temperature: 0.3
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + Enter` | Submit form |
| `R` | Refresh page |
| `C` | Clear cache |

## Troubleshooting

### API Connection Error
```
❌ Cannot connect to API
```
**Solution**: Ensure the backend is running on `http://localhost:8000`

### Import Errors
```
ModuleNotFoundError: No module named 'langgraph'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

### API Key Errors
```
OPENAI_API_KEY not found
```
**Solution**: Add your API key to `.env` file

## File Structure

```
src/ui/
├── __init__.py
└── app.py          # Main Streamlit application
```

## Customization

### Styling
Edit `CUSTOM_CSS` in `src/ui/app.py` to customize colors, fonts, and layouts.

### API Endpoint
Change `API_BASE_URL` in `src/ui/app.py` if running on a different host/port.

```python
API_BASE_URL = "http://your-server:8000/api/v1"
```

## Screenshots

### Main View
- Sidebar with task input
- Progress visualization
- Agent execution timeline

### Results View
- Final output display
- Download button
- Feedback section

### Agent Details
- Expandable cards for each agent
- Individual outputs
- Execution times
