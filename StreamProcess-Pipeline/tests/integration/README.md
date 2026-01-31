# Integration Tests for StreamProcess-Pipeline

End-to-end integration tests that validate the complete pipeline functionality.

## Running Tests

### Start Services
```bash
docker-compose up -d
```

### Run All Integration Tests
```bash
pytest tests/integration/ -v --cov=src
```

### Run Specific Test
```bash
pytest tests/integration/test_pipeline.py::test_happy_path_processing -v
```

## Test Descriptions

| Test | Description |
|------|-------------|
| happy_path_processing | Normal batch processing flow |
| large_batch_performance | 10,000 record batch |
| error_handling_invalid_records | Invalid data handling |
| duplicate_handling | Duplicate event_id detection |
| concurrent_batches | Multiple simultaneous ingestions |
| vector_store_query | Embedding and vector search |
| metrics_verification | Metrics collection |
| performance_statistics | Performance benchmarks |

## Debugging

```bash
# View logs
docker-compose logs -f api worker

# Check services
docker-compose ps

# Connect to database
docker-compose exec postgres psql -U streamprocess_user -d streamprocess
```
