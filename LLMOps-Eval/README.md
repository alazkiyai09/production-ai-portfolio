# LLMOps-Eval: Production-Ready LLM Evaluation & Deployment Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production--ready-success.svg)

**A comprehensive framework for evaluating, testing, and deploying Large Language Models with enterprise-grade monitoring.**

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Deployment](#deployment)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Supported Metrics](#supported-metrics)
- [Supported Providers](#supported-providers)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Creating Datasets](#creating-datasets)
- [API Documentation](#api-documentation)
- [Dashboard Usage](#dashboard-usage)
- [Deployment](#deployment)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**LLMOps-Eval** is a production-ready evaluation system designed for assessing LLM performance across multiple dimensions. Built for AI Engineers targeting remote roles, it demonstrates deep understanding of MLOps principles, evaluation methodologies, and production deployment.

### Why This Matters for Jobs

Industry requirements from EY, Meta, Google, and more:
- **Model evaluation metrics** (perplexity, hallucination, factual consistency)
- **Rigorous model testing, validation, and performance optimization**
- **MLOps maturity** that many AI Engineers lack
- **Production monitoring and observability**

### Key Capabilities

- 🧪 **Multi-Model Testing**: Compare OpenAI, Anthropic, Cohere, and local models side-by-side
- 📊 **9 Evaluation Metrics**: Accuracy, latency, cost, hallucination detection, safety checks, and more
- 🤖 **LLM-as-Judge**: Use powerful models to evaluate response quality
- 📈 **Automated Reporting**: Markdown and HTML reports with interactive Plotly charts
- 🚀 **Async Execution**: Parallel evaluation with configurable concurrency
- 🔍 **Real-time Monitoring**: Prometheus metrics with Grafana dashboards
- 🐳 **Production-Ready**: Docker, Kubernetes, and cloud deployment configs
- 🎯 **Comprehensive Testing**: Unit, integration, and end-to-end tests

---

## Features

### Core Evaluation System

| Feature | Description |
|---------|-------------|
| **Multi-Model Support** | Evaluate OpenAI, Anthropic, Cohere, and local Ollama models |
| **9 Built-in Metrics** | Accuracy, similarity, latency, cost, hallucination, toxicity, format compliance, LLM-judge |
| **Async Execution** | Parallel test execution with semaphore-based concurrency control |
| **Retry Logic** | Exponential backoff for failed requests |
| **Token Counting** | Accurate token counting using tiktoken (OpenAI) |
| **Cost Tracking** | Per-1K token pricing for accurate cost estimation |
| **Dataset Management** | YAML/JSON datasets with filtering, sampling, and versioning |

### Reporting & Visualization

| Feature | Description |
|---------|-------------|
| **Markdown Reports** | Plain text reports with tables and statistics |
| **HTML Reports** | Styled HTML with embedded Plotly charts |
| **CSV/JSON Export** | Data export for further analysis |
| **Model Comparison** | Statistical comparison with rankings |
| **Interactive Charts** | Bar charts, histograms, pie charts, heatmaps |
| **Trend Analysis** | Track performance over time |

### API & Dashboard

| Feature | Description |
|---------|-------------|
| **FastAPI REST API** | Full REST API with OpenAPI documentation |
| **Streamlit Dashboard** | Interactive web UI for running evaluations |
| **Background Tasks** | Async evaluation with status tracking |
| **Progress Monitoring** | Real-time progress updates with ETA |
| **Health Checks** | Component health monitoring |
| **CORS Support** | Configurable CORS origins |

### Monitoring & Observability

| Feature | Description |
|---------|-------------|
| **Prometheus Metrics** | 20+ metrics for evaluations, LLM requests, system health |
| **Alert Rules** | Pre-configured alerts for failures, high latency, cost spikes |
| **Grafana Dashboards** | Pre-built dashboards for visualization |
| **HPA Support** | Horizontal Pod Autoscaling for Kubernetes |
| **Resource Limits** | CPU/memory limits and requests |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LLMOps-Eval Architecture                     │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌───────────────┐      ┌──────────────┐
│              │      │               │      │              │
│  Streamlit   │──────│  FastAPI      │──────│  Prometheus  │
│  Dashboard   │      │  Backend      │      │   + Grafana  │
│  :8501       │      │  :8000        │      │  :9090/:3000 │
│              │      │               │      │              │
└──────────────┘      └───────┬───────┘      └──────────────┘
                               │
                      ┌────────┴────────┐
                      │                 │
               ┌──────▼──────┐   ┌─────▼─────┐
               │ Evaluation  │   │   Dataset  │
               │  Runner     │   │  Manager   │
               └──────┬──────┘   └─────┬─────┘
                      │                │
        ┌─────────────┼────────────────┼──────────────┐
        │             │                │              │
  ┌─────▼─────┐ ┌───▼────┐  ┌────────▼──┐  ┌─────▼─────┐
  │  Metrics  │ │ Models │  │ Providers │  │ Reporting │
  │  Engine   │ │        │  │           │  │           │
  └───────────┘ └────────┘  └────────────┘  └───────────┘
        │
        └──────────┬───────────────────────────┐
                   │                           │
        ┌──────────▼──────────┐    ┌──────────▼─────────┐
        │  LLM Providers      │    │   Storage          │
        │                    │    │                    │
        │  • OpenAI           │    │  • PostgreSQL/DB   │
        │  • Anthropic        │    │  • S3              │
        │  • Cohere           │    │  • Local Files     │
        │  • Ollama (local)   │    │  • Redis (cache)   │
        └─────────────────────┘    └────────────────────┘
```

### Data Flow

1. **Dataset Definition**: Define test cases in YAML/JSON
2. **Configuration**: Select models, metrics, and parameters
3. **Execution**: Runner parallelizes test execution
4. **Evaluation**: Metrics assess each response
5. **Aggregation**: Results are collected and summarized
6. **Reporting**: Generate reports and visualizations
7. **Monitoring**: Prometheus tracks all operations

---

## Supported Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Exact Match** | String equality comparison | Code, numbers, specific answers |
| **Contains** | Keyword/phrase presence checking | Factual verification |
| **Semantic Similarity** | Embedding-based cosine similarity | Paraphrase, meaning comparison |
| **LLM Judge** | GPT-4/Claude evaluates response quality | Complex reasoning, open-ended |
| **Hallucination Detection** | Fact-checking against context | Accuracy validation |
| **Toxicity** | Harmful content detection | Safety checks |
| **Format Compliance** | JSON/schema validation | Structured output |
| **Latency** | Response time measurement | Performance testing |
| **Cost** | API cost calculation | Budget tracking |

### Metric Configuration

```python
from src.evaluation.metrics import create_metric

# Simple usage
metric = create_metric("exact_match")

# With parameters
metric = create_metric(
    "semantic_similarity",
    threshold=0.85,
    model="all-mpnet-base-v2",
)
```

---

## Supported Providers

| Provider | Models | Pricing Tracking | Local |
|----------|--------|------------------|-------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo | ✅ | ❌ |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku | ✅ | ❌ |
| **Cohere** | Command, Command Light | ✅ | ❌ |
| **Ollama** | Llama 3.2, Mistral, Qwen, etc. | ❌ | ✅ |

### Pricing (per 1K tokens)

| Model | Input | Output |
|-------|-------|--------|
| GPT-4o | $0.005 | $0.015 |
| GPT-4o-mini | $0.00015 | $0.0006 |
| Claude 3.5 Sonnet | $0.003 | $0.015 |
| Claude 3 Haiku | $0.00025 | $0.00125 |
| Ollama (local) | Free | Free |

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llmops-eval.git
cd llmops-eval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# OPENAI_API_KEY=sk-your-key-here
# ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Start Services

#### Option A: Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Access:
# - API: http://localhost:8002
# - Dashboard: http://localhost:8502
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3001
```

#### Option B: Local Development

```bash
# Terminal 1: Start API
uvicorn src.api.main:app --reload --port 8002

# Terminal 2: Start Dashboard
streamlit run src/dashboard/app.py --server.port 8502
```

### 4. Run Your First Evaluation

```python
from src.runners import run_evaluation

result = await run_evaluation(
    name="my_first_eval",
    dataset="qa_evaluation",
    models=[
        {"provider": "openai", "model": "gpt-4o-mini"},
    ],
    metrics=["exact_match", "semantic_similarity"],
)

print(f"Success Rate: {result.summary['success_rate']:.1f}%")
print(f"Total Cost: ${result.summary['total_cost_usd']:.6f}")
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key |
| `ANTHROPIC_API_KEY` | Optional | Anthropic API key |
| `API_PORT` | 8002 | API server port |
| `DASHBOARD_PORT` | 8502 | Dashboard port |
| `MAX_CONCURRENT_EVALUATIONS` | 10 | Max parallel tests |
| `REQUEST_TIMEOUT` | 120 | Request timeout (seconds) |
| `ENABLE_CACHE` | true | Enable response caching |
| `LOG_LEVEL` | INFO | Logging verbosity |

### Configuration File (`.env`)

```bash
# API Keys
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# API Settings
API_HOST=0.0.0.0
API_PORT=8002
CORS_ORIGINS=http://localhost:8502

# Evaluation Settings
MAX_CONCURRENT_EVALUATIONS=10
REQUEST_TIMEOUT=120
MAX_RETRIES=3

# Cache
ENABLE_CACHE=true
CACHE_TTL=3600

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

---

## Creating Datasets

### Dataset Format

```yaml
name: my_evaluation
version: "1.0"
description: "My custom evaluation dataset"
default_metrics:
  - semantic_similarity
  - llm_judge
  - latency
test_cases:
  - id: test_001
    prompt: "What is the capital of France?"
    expected: "Paris"
    category: "factual"
    tags: ["geography", "simple"]
    metrics: ["exact_match"]
    metadata:
      difficulty: "easy"
```

### Programmatic Creation

```python
from src.datasets import DatasetManager, TestDataset, TestCase

manager = DatasetManager()

# Create dataset
dataset = TestDataset(
    name="my_dataset",
    version="1.0",
    description="My custom dataset",
    test_cases=[
        TestCase(
            id="test_001",
            prompt="Test prompt",
            expected="Expected answer",
            category="factual",
            tags=["test"],
        )
    ],
    default_metrics=["exact_match"],
)

# Save dataset
manager.save_dataset(dataset)
```

### Filtering & Sampling

```python
# Load dataset
dataset = manager.load_dataset("qa_evaluation")

# Filter by category
factual_only = manager.filter_by_category(dataset, "factual")

# Filter by tags
python_only = manager.filter_by_tags(dataset, ["python"])

# Random sample
sample = manager.sample(dataset, n=10, random_seed=42)
```

---

## API Documentation

### REST API Endpoints

#### Evaluations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/evaluate` | Start evaluation |
| `GET` | `/evaluate/{id}` | Get results |
| `GET` | `/evaluate/{id}/status` | Get status/progress |
| `GET` | `/evaluations` | List all evaluations |
| `DELETE` | `/evaluations/{id}` | Delete evaluation |

#### Datasets

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/datasets` | List datasets |
| `GET` | `/datasets/{name}` | Get dataset details |
| `POST` | `/datasets` | Upload dataset |

#### Reports & Metrics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/reports/{id}` | Generate report (HTML/MD) |
| `GET` | `/metrics` | List available metrics |
| `GET` | `/prometheus` | Prometheus metrics endpoint |

### Example: Start Evaluation

```bash
curl -X POST http://localhost:8002/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpt4_eval",
    "dataset": "qa_evaluation",
    "models": [
      {"provider": "openai", "model": "gpt-4o-mini"}
    ],
    "metrics": ["exact_match", "semantic_similarity"],
    "parallel": 3
  }'
```

### Example: Check Status

```bash
curl http://localhost:8002/evaluate/{evaluation_id}/status

# Response:
{
  "evaluation_id": "...",
  "status": "running",
  "progress": 45.5,
  "current_test": "qa_005",
  "total_tests": 30,
  "completed_tests": 14,
  "estimated_time_remaining": 35.2
}
```

### Interactive API Docs

- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

---

## Dashboard Usage

### Features

| Tab | Description |
|-----|-------------|
| **Overview** | Summary statistics and recent evaluations |
| **Results** | Detailed results with interactive charts |
| **Comparison** | Compare multiple evaluations side-by-side |
| **History** | Browse historical evaluation runs |
| **Progress** | Real-time monitoring of running evaluations |

### Quick Start with Dashboard

1. Navigate to http://localhost:8502
2. Configure evaluation in sidebar:
   - Enter evaluation name
   - Select dataset
   - Choose models (or use defaults)
   - Select metrics
3. Click "▶️ Start Evaluation"
4. Monitor progress in "Progress" tab
5. View results in "Results" tab
6. Download reports in "History" tab

### Dashboard Features

- **Real-time Updates**: Auto-refresh for running evaluations
- **Interactive Charts**: Plotly charts with zoom, pan, hover
- **Model Comparison**: Side-by-side statistics
- **Report Generation**: One-click HTML/Markdown reports
- **Historical Analysis**: Track performance over time

---

## Deployment

### Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f eval-api

# Scale API
docker-compose up -d --scale eval-api=3

# Stop all
docker-compose down
```

### Kubernetes

```bash
# Deploy to Kubernetes
./k8s/deploy.sh deploy

# Check status
kubectl get all -l app=llmops-eval

# Port-forward to access API
kubectl port-forward svc/llmops-eval-api 8002:8000

# View logs
./k8s/deploy.sh logs

# Delete deployment
./k8s/deploy.sh delete
```

### Resource Requirements

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| API (per replica) | 0.5-2 CPU | 512Mi-2Gi | - |
| Dashboard | 0.25-1 CPU | 256Mi-1Gi | - |
| Prometheus | 0.25-1 CPU | 256Mi-1Gi | 30Gi |
| Grafana | 0.1-0.5 CPU | 128Mi-512Mi | 10Gi |

### High Availability

- **Replicas**: 2+ API pods with HPA (2-10 autoscaling)
- **PodDisruptionBudget**: 1 minimum available during updates
- **Pod AntiAffinity**: Spread pods across nodes
- **Resource Limits**: Prevent resource exhaustion

---

## Example Reports

### Sample Output

```
# Evaluation Report: model_comparison

## Metadata
- **Dataset:** qa_evaluation
- **Duration:** 125.3s
- **Models Evaluated:** 3

## Summary
- **Total Tests:** 30
- **Success Rate:** 93.3%
- **Total Cost:** $0.015234
- **Avg Latency:** 1,234ms

## Model Comparison
| Model | Tests | Success Rate | Avg Score | Avg Latency | Cost |
|-------|-------|--------------|-----------|-------------|------|
| gpt-4o-mini | 10 | 95% | 0.87 | 1,200ms | $0.005 |
| claude-3-haiku | 10 | 90% | 0.82 | 800ms | $0.003 |
| llama3.2:3b | 10 | 85% | 0.68 | 500ms | $0.000 |
```

### HTML Report Features

- **Styled Header** with gradient
- **Summary Cards** with key metrics
- **Interactive Charts** (zoom, hover)
- **Color-coded Tables** (pass/fail)
- **Detailed Results** with pagination
- **Responsive Design**

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_llmops_eval.py -v

# Run with debug output
pytest tests/ -vv -s
```

### Code Formatting

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
LLMOps-Eval/
├── src/
│   ├── models/              # LLM provider interfaces
│   ├── evaluation/          # Metrics and evaluators
│   ├── datasets/            # Test dataset management
│   ├── runners/             # Test execution
│   ├── reporting/           # Results and visualization
│   ├── api/                 # FastAPI endpoints
│   ├── dashboard/           # Streamlit UI
│   ├── monitoring/          # Prometheus metrics
│   ├── utils/               # Utilities
│   └── config.py            # Configuration
├── tests/
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── fixtures/            # Test data
├── data/
│   ├── datasets/            # Test datasets
│   ├── results/             # Evaluation results
│   ├── models/              # Local models
│   └── cache/               # Response cache
├── k8s/                     # Kubernetes manifests
├── docker/                  # Docker configs
├── prometheus/              # Prometheus config
├── configs/                 # Configuration files
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project metadata
├── Dockerfile               # Container image
├── docker-compose.yml       # Local development
└── README.md               # This file
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: Follow PEP 8, use Black formatter
2. **Type Hints**: All functions must have type hints
3. **Tests**: Add tests for new features
4. **Docs**: Update documentation for API changes
5. **Commits**: Use conventional commit messages

### Commit Message Format

```
feat: add support for new LLM provider
fix: resolve race condition in evaluation runner
docs: update deployment instructions
test: add tests for semantic similarity metric
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Streamlit](https://streamlit.io/) - Interactive dashboards
- [Prometheus](https://prometheus.io/) - Metrics collection
- [Plotly](https://plotly.com/) - Interactive charts
- [sentence-transformers](https://www.sbert.net/) - Semantic similarity

---

## Contact

**Author**: AI Engineer Portfolio Project

**Issues**: [GitHub Issues](https://github.com/yourusername/llmops-eval/issues)

---

<div align="center">

**Built for AI Engineers, by AI Engineers**

⭐ Star this repo if it helps you land your dream role!

</div>
