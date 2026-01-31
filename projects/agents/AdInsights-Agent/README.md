# AdInsights-Agent

Autonomous LangGraph agent for AdTech analytics and insights generation.

## Overview

AdInsights-Agent is an intelligent analytics system that automatically analyzes advertising campaign data, detects trends and anomalies, and generates actionable insights with visualizations.

### Key Features

- **Autonomous Analysis Planning**: LangGraph agent breaks down analytical tasks automatically
- **Time-Series Trend Detection**: Statistical analysis of campaign metrics over time
- **Anomaly Detection**: Identifies outliers and unusual patterns in performance data
- **Automated Insights**: Natural language explanations of findings
- **Rich Visualizations**: Interactive charts using matplotlib and plotly
- **REST API**: FastAPI backend for integration
- **Target Domain**: Healthcare AdTech (CTR, CVR, spend, conversions)

## Tech Stack

- **LangGraph** - Agent orchestration
- **LangChain** - LLM integration
- **pandas/numpy** - Data manipulation
- **scipy/statsmodels** - Statistical analysis
- **matplotlib/plotly** - Visualizations
- **FastAPI** - REST API
- **Python 3.11+**

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd AdInsights-Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### Configuration

Edit `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### Running the Application

```bash
# Start API server
adinsights-api

# Or run directly
uvicorn src.api.main:app --reload --port 8000
```

### Run Agent CLI

```bash
# Run agent on sample data
adinsights-agent --data data/sample_campaigns.csv

# Or with Python
python -m src.agents.analysis_agent --data data/sample_campaigns.csv
```

## Project Structure

```
AdInsights-Agent/
├── src/
│   ├── agents/          # LangGraph agent implementation
│   ├── tools/           # Analysis tools (data, trends, anomalies, insights)
│   ├── analytics/       # Statistical functions
│   ├── visualization/   # Chart generation
│   └── api/             # FastAPI application
├── tests/               # Unit and integration tests
├── data/                # Sample campaign data
└── notebooks/           # Analysis notebooks
```

## API Usage

### Analyze Campaign Data

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/analyze",
        json={
            "data_source": "campaign_metrics.csv",
            "time_column": "date",
            "metrics": ["ctr", "cvr", "spend"],
            "analysis_type": "comprehensive"
        }
    )
    result = response.json()
```

### Generate Report

```python
response = await client.post(
    "http://localhost:8000/api/v1/report",
    json={
        "analysis_id": "analysis-123",
        "format": "html",
        "include_charts": True
    }
)
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_analytics.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `APP_ENV` | Environment | `development` |
| `API_PORT` | API server port | `8000` |
| `ANOMALY_THRESHOLD` | Standard deviations for anomaly detection | `2.5` |
| `TREND_CONFIDENCE` | Confidence level for trends | `0.95` |

## Example Analysis

The agent can:

1. **Load campaign data** from CSV/JSON
2. **Detect trends** in CTR, CVR, spend over time
3. **Identify anomalies** (sudden drops/spikes in performance)
4. **Generate insights** with natural language explanations
5. **Create visualizations** (line charts, heatmaps, scatter plots)
6. **Produce reports** in HTML/PDF format

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
