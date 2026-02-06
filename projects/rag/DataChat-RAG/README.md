# DataChat-RAG

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0C4C97?style=flat&logo=langchain)](https://langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker)](https://www.docker.com/)

**Natural Language to SQL Analytics with RAG**

[Features](#features) • [Quick Start](#quick-start) • [API Documentation](#api-documentation) • [Deployment](#deployment)

</div>

---

## Overview

**DataChat-RAG** is a production-grade analytics system that combines natural language query understanding with hybrid retrieval (RAG) and SQL generation for database analytics.

### Why This Matters

- **Query Routing**: Automatically classifies queries into SQL, document search, or hybrid
- **Text-to-SQL**: Generates SQL queries from natural language with schema understanding
- **Conversation Memory**: Maintains context across multi-turn conversations
- **Query Caching**: Redis-backed caching for fast repeated queries
- **Production Ready**: Docker deployment, health monitoring, comprehensive tests

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Query Routing** | LLM-powered classification: SQL_QUERY, DOC_SEARCH, HYBRID |
| **Text-to-SQL** | Natural language → SQL with schema awareness |
| **Document RAG** | Semantic retrieval from policy/guideline documents |
| **Conversation Memory** | Context-aware multi-turn conversations |
| **Query Caching** | Redis caching with TTL for performance |
| **Streaming Responses** | Real-time SSE streaming for long queries |

### Security & Reliability

| Feature | Description |
|---------|-------------|
| **Authentication** | JWT-based auth with role-based access |
| **Rate Limiting** | Configurable rate limits per endpoint |
| **Input Validation** | Pydantic models with field validation |
| **Error Handling** | Comprehensive exception handling |
| **SQL Injection Prevention** | Parameterized queries only |

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **LLM Framework** | LangChain, LlamaIndex |
| **Database** | PostgreSQL, ChromaDB |
| **Cache** | Redis |
| **API Framework** | FastAPI, uvicorn |
| **Testing** | pytest |
| **Deployment** | Docker, docker-compose |

---

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Navigate to project
cd projects/rag/DataChat-RAG

# Copy environment file
cp .env.example .env
# Edit .env with your API keys and database config

# Start all services
docker-compose up -d

# Initialize database
docker-compose exec datachat-api python scripts/setup_database.py --seed

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

# Run API
uvicorn src.api.main:app --reload --port 8000
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI REST API |
| Streamlit UI | 8501 | Chat interface |
| PostgreSQL | 5432 | Analytics database |
| ChromaDB | 8001 | Vector database |
| Redis | 6379 | Query cache |

---

## API Documentation

### Chat Endpoint

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was our average CTR last week?",
    "conversation_id": null
  }'
```

### Stream Chat (SSE)

```bash
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me top performing campaigns"}'
```

### Get Schema

```bash
curl "http://localhost:8000/schema"
```

### Health Check

```bash
curl "http://localhost:8000/health"
```

### Cache Statistics

```bash
curl "http://localhost:8000/cache/stats"
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `DATABASE_URL` | PostgreSQL connection | `postgresql://...` |
| `REDIS_URL` | Redis connection | `redis://localhost:6379/0` |
| `CHROMA_PERSIST_DIR` | ChromaDB data directory | `./data/chromadb` |
| `CACHE_ENABLED` | Enable query caching | `true` |
| `CACHE_TTL` | Cache TTL in seconds | `3600` |
| `RERANK_ENABLED` | Enable reranking | `false` |

### Database Schema

**campaigns** table:
```sql
id          UUID PRIMARY KEY
name        VARCHAR(255)
client_name VARCHAR(255)
industry    VARCHAR(50)
start_date  DATE
end_date    DATE
budget      FLOAT
status      VARCHAR(50)
```

**daily_metrics** table:
```sql
date          DATE
campaign_id   UUID REFERENCES campaigns(id)
impressions   INTEGER
clicks        INTEGER
conversions   INTEGER
spend         FLOAT
ctr           FLOAT
cvr           FLOAT
cpa           FLOAT
```

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run integration tests
pytest tests/integration/ -v
```

### Database Setup

```bash
# Initialize database with schema
python scripts/setup_database.py --all

# Seed sample data (100 campaigns)
python scripts/setup_database.py --seed --campaigns 100

# Create custom data
python scripts/setup_database.py --seed --campaigns 50 --days 30
```

### Project Structure

```
DataChat-RAG/
├── src/
│   ├── api/              # FastAPI application
│   │   └── main.py
│   ├── core/             # RAG chain implementation
│   │   └── rag_chain.py
│   ├── routers/          # Query routing logic
│   │   └── query_router.py
│   ├── retrievers/       # Document retrieval
│   │   └── doc_retriever.py
│   ├── cache/            # Query cache (Redis)
│   │   └── query_cache.py
│   └── ui/               # Streamlit chat UI
│       └── chat_app.py
├── tests/                # Unit and integration tests
├── scripts/              # Database setup utilities
├── data/                 # Data persistence
├── Dockerfile
├── docker-compose.yml
└── README.docker.md      # Detailed Docker guide
```

---

## Example Queries

### SQL Queries (Database Analytics)

```
"What was our average CTR last week?"
"Show me top 5 campaigns by spend"
"Compare performance this month vs last month"
"Which campaigns have below-average CVR?"
```

### Document Queries (Policy/Guideline)

```
"What's the policy on sensitive data?"
"How do we handle GDPR compliance?"
"What are the benchmarks for healthcare campaigns?"
```

### Hybrid Queries

```
"Why is the BioGen campaign underperforming?"
"What metrics should we track for pharma clients?"
"Is our current spend within policy guidelines?"
```

---

## Response Format

```json
{
  "answer": "Based on the data, the average CTR last week was 1.2%, which is above our healthcare benchmark of 0.8-1.5%.",
  "query_type": "SQL_QUERY",
  "confidence": 0.92,
  "conversation_id": "conv_abc123",
  "sql_query": "SELECT AVG(ctr) FROM daily_metrics WHERE date >= CURRENT_DATE - INTERVAL '7 days'",
  "sql_results": {"avg_ctr": 1.2},
  "doc_sources": [],
  "suggested_followup": [
    "Would you like to see the trend over a longer time period?",
    "Would you like to compare these metrics to our benchmarks?"
  ],
  "processing_time_seconds": 1.23,
  "is_cached": false
}
```

---

## Deployment

### Production Docker

```bash
# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f datachat-api

# Scale API
docker-compose up -d --scale datachat-api=3
```

### Health Check Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00",
  "components": [
    {
      "name": "rag_chain",
      "status": "healthy",
      "message": "RAG chain initialized"
    },
    {
      "name": "document_retriever",
      "status": "healthy",
      "message": "150 chunks indexed"
    },
    {
      "name": "sql_retriever",
      "status": "degraded",
      "message": "Not yet implemented"
    }
  ],
  "uptime_seconds": 3600.5
}
```

---

## License

MIT License - see LICENSE file for details.

---

<div align="center">

**Built for AI Engineers, by AI Engineers**

⭐ Star this repo if it helps you land your dream role!

</div>
