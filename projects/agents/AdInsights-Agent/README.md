# AdInsights-Agent

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-FF6B6B?style=flat)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker)](https://www.docker.com/)

**Autonomous LangGraph Agent for AdTech Analytics & Insights**

[Features](#features) • [Quick Start](#quick-start) • [API Documentation](#api-documentation) • [Deployment](#deployment)

</div>

---

## Overview

**AdInsights-Agent** is an intelligent analytics system that automatically analyzes advertising campaign data, detects trends and anomalies, and generates actionable insights with visualizations.

Built for healthcare AdTech with support for pharmaceuticals, hospitals, telehealth, and insurance campaigns.

### Why This Matters

- **Autonomous Analysis**: LangGraph agent plans and executes analysis without manual intervention
- **Statistical Rigor**: Uses scipy/statsmodels for statistically sound anomaly detection and forecasting
- **Healthcare Expertise**: Pre-loaded benchmarks for healthcare verticals
- **Production Ready**: Docker deployment, health checks, monitoring, and comprehensive testing

---

## Features

### Core Analytics

| Feature | Description |
|---------|-------------|
| **Trend Detection** | Linear regression, Mann-Kendall test, seasonal decomposition |
| **Anomaly Detection** | Z-score, IQR, Isolation Forest methods with severity classification |
| **Forecasting** | Exponential smoothing, ARIMA, Prophet-style predictions |
| **A/B Testing** | Statistical significance testing with lift calculation |
| **Benchmarking** | Healthcare industry benchmarks with percentile rankings |

### Agent Capabilities

| Capability | Description |
|------------|-------------|
| **Autonomous Planning** | LangGraph breaks down complex analytical requests |
| **Multi-Step Workflows** | Data gathering → Analysis → Insights → Reporting |
| **Tool Calling** | Dynamic tool selection based on request analysis |
| **State Management** | Tracks progress through long-running analyses |
| **Error Recovery** | Handles failures gracefully with informative errors |

### API Features

| Feature | Description |
|---------|-------------|
| **Background Jobs** | Submit analysis, retrieve results later |
| **Streaming Responses** | Server-Sent Events for real-time progress |
| **Quick Insights** | Fast pre-computed metrics for dashboard display |
| **Campaign Comparison** | Compare multiple campaigns side-by-side |
| **Benchmark API** | Industry benchmarks for reference |

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Agent Framework** | LangGraph, LangChain |
| **Data Processing** | pandas, numpy, scipy, statsmodels |
| **Visualization** | matplotlib, plotly |
| **API Framework** | FastAPI, uvicorn |
| **Testing** | pytest, pytest-cov |
| **Deployment** | Docker, docker-compose |

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and navigate
cd projects/agents/AdInsights-Agent

# Copy environment file
cp .env.example .env
# Edit .env and add OPENAI_API_KEY

# Start services
docker-compose up -d

# Access API
open http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run API server
uvicorn src.api.main:app --reload --port 8000
```

### Option 3: Using Makefile

```bash
make help      # List all commands
make build     # Build Docker image
make up        # Start services
make test      # Run tests
make dev       # Development mode with hot reload
```

---

## API Documentation

### Submit Analysis Job

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_id": "CAMP-001",
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "campaign_type": "healthcare_pharma",
    "analysis_types": ["detect_anomalies", "analyze_trends"],
    "include_charts": true
  }'
```

### Get Results

```bash
curl "http://localhost:8000/analyze/{job_id}"
```

### Stream Analysis (Real-time)

```bash
curl -X POST "http://localhost:8000/analyze/stream" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"campaign_id": "CAMP-001", "start_date": "2024-01-01", "end_date": "2024-01-31"}'
```

### Quick Insights

```bash
curl "http://localhost:8000/quick-insights?campaign_id=CAMP-001&days=7"
```

### Compare Campaigns

```bash
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_ids": ["CAMP-001", "CAMP-002"],
    "metric": "ctr",
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
  }'
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `APP_ENV` | Environment (development/production) | `development` |
| `API_PORT` | API server port | `8000` |
| `ANOMALY_THRESHOLD` | Standard deviations for anomaly detection | `2.5` |
| `TREND_CONFIDENCE` | Confidence level for trends | `0.95` |
| `CORS_ORIGINS` | Allowed CORS origins | `*` |

### Healthcare Benchmarks

Pre-loaded benchmarks for:

| Industry | Median CTR | Median CVR | Median CPA |
|----------|------------|------------|------------|
| **Pharma** | 1.2% | 2.8% | $280 |
| **Hospitals** | 0.9% | 1.5% | $450 |
| **Telehealth** | 1.8% | 4.2% | $120 |
| **Insurance** | 0.7% | 1.1% | $380 |

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test class
pytest tests/test_insights_agent.py::TestStatistics -v
```

### Code Quality

```bash
make lint      # Run ruff and mypy
make format    # Format with black and ruff
```

### Project Structure

```
AdInsights-Agent/
├── src/
│   ├── agents/           # LangGraph agent implementation
│   │   └── insights_agent.py
│   ├── tools/            # Analysis tools (LangChain tools)
│   │   └── analysis_tools.py
│   ├── analytics/        # Statistical functions
│   │   ├── statistics.py
│   │   ├── time_series.py
│   │   └── cohort.py
│   ├── data/             # Ad platform clients
│   │   └── ad_platform_client.py
│   ├── visualization/    # Report generation
│   │   └── report_generator.py
│   └── api/              # FastAPI application
│       └── main.py
├── tests/                # Comprehensive test suite
│   └── test_insights_agent.py
├── data/                 # Sample data and outputs
├── notebooks/            # Jupyter notebooks
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

---

## Deployment

### Docker Production

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Scale
docker-compose up -d --scale adinsights-api=3

# Stop
docker-compose down
```

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00",
  "components": {
    "api": "healthy",
    "agent": "healthy",
    "job_store": "healthy"
  },
  "uptime_seconds": 3600.5
}
```

---

## Example Output

### Analysis Report

```markdown
# Campaign Analysis Report: CAMP-001

## Executive Summary
Campaign CAMP-001 performed **ABOVE INDUSTRY BENCHMARK** over the 30-day period,
with an overall ROI of 3.2x (68th percentile).

## Key Metrics
| Metric | Value | Benchmark | Performance |
|--------|-------|-----------|-------------|
| CTR | 1.45% | 1.2% | +21% ✓ |
| CVR | 2.95% | 2.8% | +5% ✓ |
| CPA | $265 | $280 | -5% ✓ |
| ROI | 3.2x | 2.5x | +28% ✓ |

## Anomalies Detected
1. **Jan 15**: CTR spike to 2.8% (HIGH severity)
2. **Jan 22**: CVR dropped to 1.2% (MEDIUM severity)

## Trends
- **CTR**: Moderate upward trend (r² = 0.65)
- **CVR**: Stable with slight decline
- **ROI**: Strong upward trajectory

## Recommendations
1. Investigate creative/assets used on Jan 15 for replication
2. Review landing page performance after Jan 22 decline
3. Consider increasing budget for top-performing segments
```

---

## License

MIT License - see LICENSE file for details.

---

<div align="center">

**Built for AI Engineers, by AI Engineers**

⭐ Star this repo if it helps you land your dream role!

</div>
