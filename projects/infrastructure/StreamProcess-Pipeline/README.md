# StreamProcess-Pipeline

High-throughput data processing pipeline for streaming analytics with ML-based embeddings and vector storage.

## Overview

StreamProcess-Pipeline is a production-grade streaming data processing system built with:
- **FastAPI** for REST API endpoints
- **Celery** for distributed task processing
- **Redis** for message queuing and caching
- **PostgreSQL** for persistent storage
- **ChromaDB** for vector embeddings and similarity search

## Features

- High-throughput data ingestion and processing
- Real-time embedding generation using sentence transformers
- Vector similarity search with ChromaDB
- Distributed task processing with Celery workers
- RESTful API for data submission and querying
- Health monitoring and metrics endpoints

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+

### Using Docker Compose (Recommended)

```bash
# 1. Set environment variables
cp .env.example .env
# Edit .env and set strong passwords

# 2. Start all services
docker-compose up -d

# 3. Check service health
curl http://localhost:8000/health
```

### Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL and Redis (requires Docker)
docker-compose up -d postgres redis

# 4. Run migrations (if applicable)
python scripts/init_db.py

# 5. Start the API server
uvicorn src.api.main:app --reload

# 6. Start Celery worker (in another terminal)
celery -A src.tasks worker -l INFO
```

## API Endpoints

### Health Check
```
GET /health
```

### Submit Data for Processing
```
POST /api/v1/process
Content-Type: application/json

{
  "data": "your text data here",
  "metadata": {"key": "value"}
}
```

### Query Similar Documents
```
POST /api/v1/query
Content-Type: application/json

{
  "query": "search text",
  "top_k": 10
}
```

## Configuration

See `.env.example` for all configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_PASSWORD` | PostgreSQL password | REQUIRED |
| `POSTGRES_DB` | Database name | streamprocess |
| `POSTGRES_USER` | Database user | streamprocess_user |
| `REDIS_HOST` | Redis host | redis |
| `CHROMA_HOST` | ChromaDB host | chroma |
| `EMBEDDING_MODEL_NAME` | Transformer model | sentence-transformers/all-MiniLM-L6-v2 |

## Architecture

```
┌─────────────┐
│   FastAPI   │
│    Server   │
└──────┬──────┘
       │
       ├────────────────┐
       │                │
       ▼                ▼
┌──────────┐      ┌─────────┐
│  Celery  │      │  Redis  │
│  Worker  │◄─────┤  Queue  │
└─────┬────┘      └─────────┘
      │
      ├────────────────┐
      │                │
      ▼                ▼
┌──────────┐      ┌──────────┐
│PostgreSQL│      │ ChromaDB │
│  (Data)   │      │(Vectors) │
└──────────┘      └──────────┘
```

## Monitoring

- **Health Endpoint**: `GET /health`
- **Metrics Endpoint**: `GET /metrics` (Prometheus format)
- **Flower UI**: http://localhost:5555 (Celery monitoring)

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
black src/
isort src/
mypy src/
```

## Deployment

### Kubernetes

See `k8s/` directory for Kubernetes manifests:

```bash
kubectl apply -f k8s/
```

**IMPORTANT**: Update `k8s/02-secret.yaml` with your actual passwords before deploying.

## Troubleshooting

### Database Connection Issues
- Ensure PostgreSQL is running: `docker-compose ps postgres`
- Check environment variables are set correctly

### Celery Tasks Not Processing
- Check Redis is running: `docker-compose ps redis`
- View worker logs: `docker-compose logs worker`
- Check Flower UI: http://localhost:5555

### High Memory Usage
- Reduce `EMBEDDING_BATCH_SIZE` in .env
- Limit `TRANSFORMERS_CACHE` size

## License

MIT

## Author

**Ahmad Whafa Azka Al Azkiyai**
- GitHub: [@alazkiyai09](https://github.com/alazkiyai09)
- Email: [azka.alazkiyai@outlook.com](mailto:azka.alazkiyai@outlook.com)
