# Docker Deployment Guide for DataChat-RAG

Complete Docker containerization for the healthcare AdTech RAG system.

## Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+

### Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
# Optional: COHERE_API_KEY=your-cohere-api-key
```

### Production Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps
curl http://localhost:8000/health
```

### Development Mode

```bash
# Start with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Run tests in dev container
docker-compose run --rm dev-tools pytest tests/

# Access dev tools
docker-compose --profile tools up
```

## Services

| Service | Container | Ports | Description |
|---------|-----------|-------|-------------|
| PostgreSQL | `datachat-postgres` | 5432 | Campaign metrics database |
| ChromaDB | `datachat-chromadb` | 8001 | Vector database for documents |
| FastAPI | `datachat-api` | 8000 | REST API |
| Streamlit | `datachat-ui` | 8501 | Chat interface |
| Init | `datachat-init` | - | One-time setup (runs on startup) |

### Optional Services (Development)

| Service | Container | Ports | Description |
|---------|-----------|-------|-------------|
| pgAdmin | `datachat-pgadmin` | 5050 | Database admin UI |
| Redis | `datachat-redis` | 6379 | Caching layer |
| Dev Tools | `datachat-dev-tools` | - | Testing/development utilities |

## Volumes

| Volume | Purpose |
|--------|---------|
| `postgres_data` | PostgreSQL persistence |
| `chroma_data` | ChromaDB persistence |
| `redis_data` | Redis persistence (optional) |

## Configuration

### Environment Variables

```bash
# Database
DB_HOST=postgres
DB_PORT=5432
DB_NAME=datachat_rag
DB_USER=postgres
DB_PASSWORD=your-secure-password

# API
API_PORT=8000
STREAMLIT_PORT=8501

# OpenAI
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# RAG Settings
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=5
RERANK_ENABLED=false
```

## Commands

### Build Services

```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build datachat-api

# Rebuild without cache
docker-compose build --no-cache
```

### Start/Stop Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up datachat-api

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### View Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f datachat-api

# View last 100 lines
docker-compose logs --tail=100
```

### Execute Commands

```bash
# Run Python in API container
docker-compose exec datachat-api python -c "print('Hello')"

# Open shell in container
docker-compose exec datachat-api bash

# Run tests
docker-compose exec datachat-api pytest tests/

# Access PostgreSQL
docker-compose exec postgres psql -U postgres -d datachat_rag
```

### Database Operations

```bash
# Run database setup manually
docker-compose exec datachat-api python scripts/setup_database.py --all

# Seed sample data
docker-compose exec datachat-api python scripts/setup_database.py --seed --campaigns 100

# Access psql
docker-compose exec postgres psql -U postgres -d datachat_rag

# Create database backup
docker-compose exec postgres pg_dump -U postgres datachat_rag > backup.sql

# Restore database
docker-compose exec -T postgres psql -U postgres datachat_rag < backup.sql
```

## Health Checks

### Check Service Status

```bash
# API health
curl http://localhost:8000/health

# ChromaDB health
curl http://localhost:8001/api/v1/heartbeat

# PostgreSQL health
docker-compose exec postgres pg_isready -U postgres
```

### Expected Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-30T10:00:00",
  "components": [
    {
      "name": "rag_chain",
      "status": "healthy",
      "message": "RAG chain initialized"
    },
    {
      "name": "document_retriever",
      "status": "healthy",
      "message": "5 chunks indexed"
    },
    {
      "name": "sql_retriever",
      "status": "degraded",
      "message": "Not yet implemented"
    },
    {
      "name": "query_router",
      "status": "healthy",
      "message": "Query router initialized"
    }
  ],
  "uptime_seconds": 120.5
}
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs [service-name]

# Check resource usage
docker stats

# Restart service
docker-compose restart [service-name]
```

### Database Connection Issues

```bash
# Verify PostgreSQL is running
docker-compose ps postgres

# Check PostgreSQL logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U postgres -c "SELECT 1"
```

### ChromaDB Connection Issues

```bash
# Verify ChromaDB is running
docker-compose ps chromadb

# Check ChromaDB logs
docker-compose logs chromadb

# Test connection
curl http://localhost:8001/api/v1/heartbeat
```

### Permission Issues

```bash
# Fix volume permissions
docker-compose exec datachat-api chown -R appuser:appuser /app/data

# Rebuild with correct permissions
docker-compose down -v
docker-compose up -d
```

### Initialization Failures

```bash
# Re-run initialization
docker-compose up datachat-init

# Manual initialization
docker-compose exec datachat-api python scripts/init_docker.py
```

## Production Considerations

### Security

1. **Change default passwords**
   ```bash
   # Generate strong password
   openssl rand -base64 32
   ```

2. **Use secrets management**
   - Docker Secrets (Swarm)
   - Kubernetes Secrets
   - AWS Secrets Manager
   - Azure Key Vault

3. **Network isolation**
   - Services on private network
   - Only expose necessary ports
   - Use TLS/SSL for external access

4. **Rate limiting**
   - Implement API rate limits
   - Use nginx/traefik as reverse proxy

### Monitoring

1. **Health checks**
   - Container health checks enabled
   - External monitoring (Prometheus, Datadog)

2. **Logging**
   - Centralized logging (ELK, Loki)
   - Structured JSON logs
   - Log rotation

3. **Metrics**
   - API response times
   - Database query performance
   - Resource utilization

### Scaling

1. **Horizontal scaling**
   ```yaml
   # docker-compose.yml
   datachat-api:
     deploy:
       replicas: 3
   ```

2. **Load balancing**
   - nginx/traefik
   - AWS ALB/NLB
   - Kubernetes Services

3. **Database optimization**
   - Read replicas
   - Connection pooling
   - Query caching

## Backup and Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
docker-compose exec postgres pg_dump -U postgres datachat_rag > \
  "$BACKUP_DIR/datachat_$DATE.sql"
```

### ChromaDB Backup

```bash
# Backup ChromaDB volume
docker run --rm \
  -v datachat-chromadb-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/chromadb_$(date +%Y%m%d).tar.gz -C /data .
```

### Restore

```bash
# Restore PostgreSQL
docker-compose exec -T postgres psql -U postgres datachat_rag < backup.sql

# Restore ChromaDB
docker run --rm \
  -v datachat-chromadb-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/chromadb_backup.tar.gz -C /data
```

## Performance Tuning

### API Workers

```yaml
datachat-api:
  command: >
    gunicorn src.api.main:app
    --workers 4
    --worker-class uvicorn.workers.UvicornWorker
    --bind 0.0.0.0:8000
    --timeout 120
    --max-requests 1000
    --max-requests-jitter 100
```

### Database

```yaml
postgres:
  command: >
    postgres
    -c shared_buffers=256MB
    -c effective_cache_size=1GB
    -c maintenance_work_mem=64MB
    -c checkpoint_completion_target=0.9
    -c wal_buffers=16MB
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Docker Build

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build images
        run: docker-compose build
      - name: Run tests
        run: docker-compose run --rm datachat-api pytest tests/
```

## Support

For issues or questions:
- GitHub Issues: [repository-url]
- Documentation: [docs-url]
- Email: support@datachat.example.com
