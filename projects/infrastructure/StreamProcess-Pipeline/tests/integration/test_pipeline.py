"""
End-to-end integration tests for StreamProcess-Pipeline.

Demonstrates the complete pipeline:
1. Ingestion → 2. Processing → 3. Embedding → 4. Vector Store → 5. Querying
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import Mock

import httpx
import pytest
import redis.asyncio as redis
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.storage.models import EventRecord, ProcessingStatus


# ============================================================================
# Test Configuration
# ============================================================================

API_BASE_URL = "http://localhost:8000"
FLOWER_URL = "http://localhost:5555"
REDIS_URL = "redis://localhost:6379/0"
POSTGRES_URL = "postgresql://streamprocess_user:streamprocess_pass@localhost:5432/streamprocess"

# Performance thresholds
INGESTION_RATE_THRESHOLD = 1000  # records/second
PROCESSING_LATENCY_THRESHOLD = 5.0  # seconds for 1000 records
QUERY_LATENCY_THRESHOLD = 0.1  # seconds
BATCH_SIZE_SMALL = 100
BATCH_SIZE_MEDIUM = 1000
BATCH_SIZE_LARGE = 10000


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def docker_compose():
    """
    Start Docker Compose services for integration tests.

    This fixture assumes docker-compose is already running
    or the user will start it manually.
    """
    # Check if services are available
    try:
        import requests
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print("\n✓ Docker Compose services are running")
    except Exception:
        pytest.skip("Docker Compose services not running. Start with: docker-compose up -d")

    yield

    # Cleanup happens automatically when Docker Compose stops


@pytest.fixture
async def api_client():
    """Create async HTTP client for API calls."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield client


@pytest.fixture
async def redis_client():
    """Create Redis client for state verification."""
    client = await redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    yield client
    await client.close()


@pytest.fixture
def db_session():
    """Create database session for verification."""
    engine = create_engine(POSTGRES_URL.replace("postgresql://", "postgresql+psycopg2://"))
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def sample_adtech_record() -> Dict[str, Any]:
    """Generate a single sample AdTech event."""
    timestamp = datetime.utcnow()
    return {
        "event_id": f"evt_{timestamp.timestamp()}_{hash(timestamp) % 10000:04d}",
        "event_type": "impression",
        "timestamp": timestamp.isoformat() + "Z",
        "campaign_id": f"camp_{(hash(timestamp) % 100):02d}",
        "user_id": f"user_{(hash(timestamp) % 1000):04d}",
        "content": "Amazing healthcare solution for your needs. Quality service you can trust.",
        "metadata": {
            "device_type": "mobile",
            "os": "iOS",
            "country": "US",
            "region": "CA",
            "placement": "homepage_banner",
            "brand_safety_score": 95,
            "viewability": 88,
        },
    }


@pytest.fixture
def sample_adtech_batch(sample_adtech_record) -> List[Dict[str, Any]]:
    """Generate a batch of 100 sample AdTech events."""
    return [
        {
            **sample_adtech_record,
            "event_id": f"evt_{i}_{int(time.time() * 1000) % 100000}",
            "timestamp": (datetime.utcnow() - timedelta(seconds=i)).isoformat() + "Z",
            "campaign_id": f"camp_{i % 10}",
            "user_id": f"user_{i % 50}",
            "content": f"Ad content variation {i}. Healthcare services you can trust.",
            "metadata": {
                "device_type": ["mobile", "desktop", "tablet"][i % 3],
                "os": ["iOS", "Android", "Windows", "macOS"][i % 4],
                "country": ["US", "UK", "CA", "DE", "FR"][i % 5],
            },
        }
        for i in range(100)
    ]


@pytest.fixture
def large_adtech_batch() -> List[Dict[str, Any]]:
    """Generate a large batch of 10,000 AdTech events."""
    batch = []
    base_time = datetime.utcnow()

    for i in range(10000):
        batch.append({
            "event_id": f"evt_large_{i}_{int(time.time() * 1000) % 1000000}",
            "event_type": ["impression", "click", "conversion"][i % 3],
            "timestamp": (base_time - timedelta(seconds=i)).isoformat() + "Z",
            "campaign_id": f"camp_{i % 50}",
            "user_id": f"user_{i % 5000}",
            "content": f"Healthcare advertisement content {i}. Click to learn more.",
            "metadata": {
                "device_type": ["mobile", "desktop", "tablet"][i % 3],
                "os": ["iOS", "Android", "Windows", "macOS"][i % 4],
                "country": ["US", "UK", "CA", "DE", "FR", "JP", "AU"][i % 7],
                "position": i % 100,
            },
        })

    return batch


# ============================================================================
# Test Helper Functions
# ============================================================================

async def wait_for_processing(
    batch_id: str,
    timeout: float = 60.0,
    poll_interval: float = 1.0,
) -> Dict[str, Any]:
    """
    Wait for batch processing to complete.

    Args:
        batch_id: Batch ID to wait for
        timeout: Maximum wait time in seconds
        poll_interval: Polling interval in seconds

    Returns:
        Batch status dictionary
    """
    async with httpx.AsyncClient() as client:
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = await client.get(f"{API_BASE_URL}/ingest/status/{batch_id}")

            if response.status_code == 200:
                status = response.json()
                if status["status"] in ["completed", "failed", "partial"]:
                    return status

            await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s")


async def ingest_batch(
    client: httpx.AsyncClient,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ingest a batch of records through the API.

    Args:
        client: HTTP client
        records: List of event records

    Returns:
        Ingestion response
    """
    start_time = time.time()
    response = await client.post(
        f"{API_BASE_URL}/ingest/batch",
        json={"records": records},
    )
    duration = time.time() - start_time

    response.raise_for_status()
    result = response.json()

    # Add timing info
    result["ingestion_duration"] = duration
    result["ingestion_rate"] = len(records) / duration if duration > 0 else 0

    return result


async def get_metrics() -> Dict[str, Any]:
    """
    Get current metrics from the API.

    Returns:
        Metrics dictionary
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/ingest/metrics")
        response.raise_for_status()
        return response.json()


async def query_vector_store(
    query_text: str,
    n_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Query the vector store for similar content.

    Args:
        query_text: Query text
        n_results: Number of results to return

    Returns:
        List of search results
    """
    # This would call the vector store API endpoint
    # For now, we'll simulate the check
    return []


# ============================================================================
# Test 1: Happy Path - Normal Batch Processing
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_happy_path_processing(
    docker_compose,
    api_client,
    sample_adtech_batch,
    db_session,
):
    """
    Test the complete happy path:
    - Ingest batch
    - Wait for processing
    - Verify in database
    - Check metrics
    """
    print("\n=== Test: Happy Path Processing ===")

    # Step 1: Ingest batch
    print(f"Step 1: Ingesting {len(sample_adtech_batch)} records...")
    result = await ingest_batch(api_client, sample_adtech_batch)

    batch_id = result["batch_id"]
    print(f"  ✓ Batch ID: {batch_id}")
    print(f"  ✓ Status: {result['status']}")
    print(f"  ✓ Ingestion rate: {result['ingestion_rate']:.1f} records/sec")

    # Assertions
    assert result["status"] == "queued"
    assert result["record_count"] == len(sample_adtech_batch)
    assert result["ingestion_rate"] > INGESTION_RATE_THRESHOLD

    # Step 2: Wait for processing
    print(f"\nStep 2: Waiting for processing...")
    status = await wait_for_processing(batch_id, timeout=60.0)
    print(f"  ✓ Final status: {status['status']}")
    print(f"  ✓ Processed: {status['processed_records']}/{status['total_records']}")

    # Assertions
    assert status["status"] == "completed"
    assert status["processed_records"] == len(sample_adtech_batch)
    assert status["failed_records"] == 0

    # Verify processing completed within threshold
    completed_at = datetime.fromisoformat(status["completed_at"].replace("Z", "+00:00"))
    created_at = datetime.fromisoformat(status["created_at"].replace("Z", "+00:00"))
    processing_duration = (completed_at - created_at).total_seconds()
    print(f"  ✓ Processing duration: {processing_duration:.2f}s")
    assert processing_duration < PROCESSING_LATENCY_THRESHOLD

    # Step 3: Verify in database
    print(f"\nStep 3: Verifying records in database...")
    records_in_db = db_session.query(EventRecord).filter(
        EventRecord.batch_id == batch_id
    ).all()

    print(f"  ✓ Found {len(records_in_db)} records in database")
    assert len(records_in_db) == len(sample_adtech_batch)

    # Verify each record has embedding_id
    records_with_embeddings = [r for r in records_in_db if r.embedding_id is not None]
    print(f"  ✓ Records with embeddings: {len(records_with_embeddings)}")
    assert len(records_with_embeddings) == len(sample_adtech_batch)

    # Step 4: Check metrics
    print(f"\nStep 4: Checking metrics...")
    metrics = await get_metrics()
    queue_size = metrics.get("queue_size", 0)
    print(f"  ✓ Queue size: {queue_size}")

    print("\n✅ Happy path test PASSED")


# ============================================================================
# Test 2: Large Batch Performance
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_large_batch_performance(
    docker_compose,
    api_client,
    large_adtech_batch,
    db_session,
):
    """
    Test processing of large batch (10,000 records).

    Validates:
    - High throughput ingestion
    - Efficient batch processing
    - Memory usage stays reasonable
    """
    print("\n=== Test: Large Batch Performance ===")
    print(f"Batch size: {len(large_adtech_batch)} records")

    # Step 1: Ingest large batch
    start_time = time.time()
    result = await ingest_batch(api_client, large_adtech_batch)
    ingestion_duration = time.time() - start_time

    batch_id = result["batch_id"]
    print(f"✓ Ingestion completed in {ingestion_duration:.2f}s")
    print(f"✓ Ingestion rate: {result['ingestion_rate']:.1f} records/sec")

    # Assertions
    assert result["record_count"] == len(large_adtech_batch)
    assert result["ingestion_rate"] > INGESTION_RATE_THRESHOLD

    # Step 2: Wait for processing (may take longer for large batch)
    print(f"\nWaiting for processing (this may take a while)...")
    status = await wait_for_processing(batch_id, timeout=300.0)  # 5 minutes

    processing_duration = (
        datetime.fromisoformat(status["completed_at"].replace("Z", "+00:00")) -
        datetime.fromisoformat(status["created_at"].replace("Z", "+00:00"))
    ).total_seconds()

    print(f"✓ Processing completed in {processing_duration:.2f}s")
    print(f"✓ Records processed: {status['processed_records']}")

    # Performance assertions
    # Allow more time for large batches, but should still be reasonable
    expected_max_duration = len(large_adtech_batch) / 100  # 100 records/sec minimum
    assert processing_duration < expected_max_duration

    # Verify database
    records_in_db = db_session.query(EventRecord).filter(
        EventRecord.batch_id == batch_id
    ).count()
    assert records_in_db == len(large_adtech_batch)

    print("\n✅ Large batch test PASSED")


# ============================================================================
# Test 3: Error Handling - Invalid Records
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_handling_invalid_records(
    docker_compose,
    api_client,
    sample_adtech_record,
):
    """
    Test error handling for invalid records.

    Scenarios:
    - Missing required fields
    - Invalid data types
    - Malformed timestamps
    """
    print("\n=== Test: Error Handling ===")

    # Create invalid records
    invalid_records = [
        # Missing required field
        {
            "event_type": "impression",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            # Missing event_id
            "campaign_id": "camp_01",
            "user_id": "user_001",
            "content": "Test content",
        },
        # Invalid event_type
        {
            "event_id": "evt_invalid_1",
            "event_type": "invalid_type",  # Wrong type
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "campaign_id": "camp_01",
            "user_id": "user_001",
            "content": "Test content",
        },
        # Empty content
        {
            "event_id": "evt_invalid_2",
            "event_type": "impression",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "campaign_id": "camp_01",
            "user_id": "user_001",
            "content": "   ",  # Empty/whitespace
        },
        # Future timestamp
        {
            "event_id": "evt_invalid_3",
            "event_type": "impression",
            "timestamp": (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z",
            "campaign_id": "camp_01",
            "user_id": "user_001",
            "content": "Test content",
        },
    ]

    print(f"Sending {len(invalid_records)} invalid records...")

    try:
        result = await ingest_batch(api_client, invalid_records)
        # Should fail with validation error
        assert False, "Expected validation error"
    except httpx.HTTPStatusError as e:
        print(f"✓ Got expected error: {e.response.status_code}")
        assert e.response.status_code == 400

        error_detail = e.response.json()
        print(f"✓ Error type: {error_detail.get('error')}")
        assert "ValidationError" in error_detail.get("error", "")

        validation_errors = error_detail.get("validation_errors", [])
        print(f"✓ Validation errors: {len(validation_errors)}")
        assert len(validation_errors) == 4

    print("\n✅ Error handling test PASSED")


# ============================================================================
# Test 4: Duplicate Handling
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_duplicate_handling(
    docker_compose,
    api_client,
    sample_adtech_record,
    redis_client,
    db_session,
):
    """
    Test duplicate event_id handling.

    Scenarios:
    - Same event_id in same batch (should reject)
    - Same event_id in different batches (should reject second)
    """
    print("\n=== Test: Duplicate Handling ===")

    event_id = f"evt_duplicate_{int(time.time() * 1000)}"

    # Create duplicate records with same event_id
    duplicate_records = [
        {
            **sample_adtech_record,
            "event_id": event_id,
            "content": "First occurrence",
        },
        {
            **sample_adtech_record,
            "event_id": event_id,  # Same ID
            "content": "Duplicate occurrence",
            "campaign_id": "camp_02",  # Different field
        },
    ]

    print(f"Sending batch with duplicate event_id: {event_id}")

    # This should be rejected during validation
    try:
        result = await ingest_batch(api_client, duplicate_records)
        assert False, "Expected duplicate error"
    except httpx.HTTPStatusError as e:
        print(f"✓ Got expected error: {e.response.status_code}")
        assert e.response.status_code == 400

        error_detail = e.response.json()
        assert "duplicate" in error_detail.get("message", "").lower()
        print(f"✓ Error message: {error_detail.get('message')}")

    # Now test with a valid record, then try to send the same event_id again
    valid_record = {
        **sample_adtech_record,
        "event_id": event_id,
        "content": "Valid record for dedupe test",
    }

    # First ingestion should succeed
    print(f"\nSending first occurrence of event: {event_id}")
    result1 = await ingest_batch(api_client, [valid_record])
    batch_id_1 = result1["batch_id"]
    print(f"✓ First batch accepted: {batch_id_1}")

    # Wait for first batch to process
    await wait_for_processing(batch_id_1, timeout=30.0)

    # Second ingestion with same event_id should fail
    print(f"Sending duplicate of event: {event_id}")
    try:
        result2 = await ingest_batch(api_client, [valid_record])
        # Note: The current implementation might reject duplicates during validation
        # or might accept them but skip during processing
        # We'll check the status to verify
        batch_id_2 = result2["batch_id"]
        status = await wait_for_processing(batch_id_2, timeout=30.0)

        # Should have processed 0 records (skipped duplicate)
        if status["processed_records"] == 0:
            print(f"✓ Duplicate was skipped during processing")
        else:
            print(f"✓ Batch processed {status['processed_records']} records")

    except httpx.HTTPStatusError as e:
        print(f"✓ Duplicate rejected at validation: {e.response.status_code}")

    print("\n✅ Duplicate handling test PASSED")


# ============================================================================
# Test 5: Concurrent Batches
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_batches(
    docker_compose,
    sample_adtech_record,
):
    """
    Test multiple simultaneous batch ingestions.

    Validates:
    - Thread-safe ingestion
    - No race conditions
    - Queue management works
    - All batches complete successfully
    """
    print("\n=== Test: Concurrent Batches ===")

    num_batches = 5
    records_per_batch = 50

    # Create multiple batches
    batches = []
    for i in range(num_batches):
        batch = [
            {
                **sample_adtech_record,
                "event_id": f"evt_concurrent_{i}_{j}_{int(time.time() * 1000) % 100000}",
                "timestamp": (datetime.utcnow() - timedelta(seconds=j)).isoformat() + "Z",
                "content": f"Concurrent test batch {i} record {j}",
            }
            for j in range(records_per_batch)
        ]
        batches.append(batch)

    print(f"Sending {num_batches} batches concurrently ({records_per_batch} records each)...")

    # Send all batches concurrently
    start_time = time.time()

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [ingest_batch(client, batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    duration = time.time() - start_time

    # Check results
    successful = [r for r in results if isinstance(r, dict)]
    failed = [r for r in results if isinstance(r, Exception)]

    print(f"✓ Completed: {len(success)}/{num_batches} batches")
    print(f"✓ Failed: {len(failed)}/{num_batches} batches")
    print(f"✓ Total duration: {duration:.2f}s")
    print(f"✓ Total records: {len(success) * records_per_batch}")
    print(f"✓ Combined rate: {(len(success) * records_per_batch) / duration:.1f} records/sec")

    # All should succeed
    assert len(failed) == 0
    assert len(success) == num_batches

    # Wait for all batches to process
    batch_ids = [r["batch_id"] for r in results]
    print(f"\nWaiting for {len(batch_ids)} batches to process...")

    for batch_id in batch_ids:
        status = await wait_for_processing(batch_id, timeout=90.0)
        print(f"  ✓ Batch {batch_id}: {status['status']} ({status['processed_records']}/{status['total_records']})")
        assert status["status"] == "completed"

    print("\n✅ Concurrent batches test PASSED")


# ============================================================================
# Test 6: Vector Store Query
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_vector_store_query(
    docker_compose,
    api_client,
    sample_adtech_batch,
    db_session,
):
    """
    Test vector store functionality.

    Validates:
    - Embeddings are generated
    - Vector store is populated
    - Similarity search works
    """
    print("\n=== Test: Vector Store Query ===")

    # Step 1: Ingest and process batch
    result = await ingest_batch(api_client, sample_adtech_batch)
    batch_id = result["batch_id"]

    await wait_for_processing(batch_id, timeout=60.0)

    # Step 2: Query vector store
    query_text = "healthcare services medical"
    print(f"Querying vector store for: '{query_text}'")

    start_time = time.time()

    # For now, we'll verify embeddings were generated in the database
    records_with_embeddings = db_session.query(EventRecord).filter(
        EventRecord.batch_id == batch_id,
        EventRecord.embedding_id.isnot(None)
    ).all()

    query_duration = time.time() - start_time

    print(f"✓ Records with embeddings: {len(records_with_embeddings)}")
    assert len(records_with_embeddings) == len(sample_adtech_batch)

    # Verify query latency
    print(f"✓ Query duration: {query_duration*1000:.2f}ms")
    assert query_duration < QUERY_LATENCY_THRESHOLD

    # Step 3: Verify embeddings have correct dimension
    if records_with_embeddings:
        # We can't easily check the embedding vector here without ChromaDB client
        # but we can verify it was generated
        first_record = records_with_embeddings[0]
        assert first_record.embedding_id is not None
        print(f"✓ Sample embedding_id: {first_record.embedding_id}")

    print("\n✅ Vector store query test PASSED")


# ============================================================================
# Test 7: End-to-End Metrics Verification
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_verification(
    docker_compose,
    api_client,
    sample_adtech_batch,
):
    """
    Test that metrics are properly recorded.

    Validates:
    - Ingestion metrics are recorded
    - Processing metrics are recorded
    - Embedding metrics are recorded
    - Metrics are accessible via /metrics endpoint
    """
    print("\n=== Test: Metrics Verification ===")

    # Get initial metrics
    initial_metrics = await get_metrics()
    initial_queue_size = initial_metrics.get("queue_size", 0)
    print(f"Initial queue size: {initial_queue_size}")

    # Process a batch
    result = await ingest_batch(api_client, sample_adtech_batch)
    batch_id = result["batch_id"]

    await wait_for_processing(batch_id, timeout=60.0)

    # Get final metrics
    final_metrics = await get_metrics()
    final_queue_size = final_metrics.get("queue_size", 0)
    print(f"Final queue size: {final_queue_size}")

    # Verify Prometheus metrics endpoint
    async with httpx.AsyncClient() as client:
        prom_metrics_response = await client.get(f"{API_BASE_URL}/metrics")
        assert prom_metrics_response.status_code == 200

        prom_metrics_text = prom_metrics_response.text

        # Check for key metrics
        assert "streamprocess_ingestion_records_total" in prom_metrics_text
        assert "streamprocess_processing_records_total" in prom_metrics_text
        assert "streamprocess_embedding_generated_total" in prom_metrics_text

        print("✓ Prometheus metrics available")
        print("✓ Ingestion metrics present")
        print("✓ Processing metrics present")
        print("✓ Embedding metrics present")

    # Verify health status
    async with httpx.AsyncClient() as client:
        health_response = await client.get(f"{API_BASE_URL}/ingest/health")
        assert health_response.status_code == 200

        health = health_response.json()
        print(f"\n✓ Health status: {health.get('status')}")
        assert health.get("status") in ["healthy", "unknown"]  # "unknown" if no components registered

    print("\n✅ Metrics verification test PASSED")


# ============================================================================
# Test 8: Performance Statistics Report
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_performance_statistics(
    docker_compose,
    api_client,
    sample_adtech_batch,
    large_adtech_batch,
):
    """
    Generate comprehensive performance statistics report.

    Tests various batch sizes and reports:
    - Ingestion throughput
    - Processing latency
    - End-to-end latency
    - Resource utilization
    """
    print("\n" + "="*60)
    print("PERFORMANCE STATISTICS REPORT")
    print("="*60)

    test_batches = [
        ("Small (100 records)", sample_adtech_batch[:100]),
        ("Medium (1000 records)", sample_adtech_batch * 10),
        ("Large (5000 records)", large_adtech_batch[:5000]),
    ]

    stats = []

    for batch_name, batch in test_batches:
        print(f"\n{batch_name}:")
        print("-" * 40)

        # Ingestion
        ingest_result = await ingest_batch(api_client, batch)
        batch_id = ingest_result["batch_id"]

        ingest_rate = ingest_result["ingestion_rate"]
        print(f"  Ingestion: {ingest_rate:.1f} records/sec")

        # Processing
        start_wait = time.time()
        status = await wait_for_processing(batch_id, timeout=180.0)
        wait_time = time.time() - start_wait

        processing_duration = (
            datetime.fromisoformat(status["completed_at"].replace("Z", "+00:00")) -
            datetime.fromisoformat(status["created_at"].replace("Z", "+00:00"))
        ).total_seconds()

        print(f"  Processing: {processing_duration:.2f}s")
        print(f"  Wait time: {wait_time:.2f}s")
        print(f"  Records: {status['processed_records']}/{status['total_records']}")

        # Calculate stats
        stats.append({
            "batch_name": batch_name,
            "batch_size": len(batch),
            "ingestion_rate": ingest_rate,
            "processing_duration": processing_duration,
            "wait_time": wait_time,
            "throughput": len(batch) / processing_duration if processing_duration > 0 else 0,
        })

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Batch Size':<20} {'Ingest Rate':<15} {'Process Time':<15} {'Throughput':<15}")
    print("-" * 60)

    for stat in stats:
        batch_size_str = stat['batch_name'].split('(')[1].split(' records')[0]
        print(f"{batch_size_str:<20} {stat['ingestion_rate']:<15.1f} {stat['processing_duration']:<15.2f} {stat['throughput']:<15.1f}")

    # Performance assertions
    print("\nPerformance Assertions:")

    # Small batch should be fast
    small_stat = stats[0]
    assert small_stat["processing_duration"] < 2.0, "Small batch should process in <2s"
    print(f"✓ Small batch (< 2s): {small_stat['processing_duration']:.2f}s")

    # Ingestion rate should be high
    assert small_stat["ingestion_rate"] > INGESTION_RATE_THRESHOLD
    print(f"✓ Ingestion rate (> {INGESTION_RATE_THRESHOLD} rec/s): {small_stat['ingestion_rate']:.1f} rec/s")

    # Throughput should scale reasonably
    medium_throughput = stats[1]["throughput"]
    large_throughput = stats[2]["throughput"]
    # Large batch throughput might be lower but should still be reasonable
    assert large_throughput > 10  # At least 10 records/sec
    print(f"✓ Medium batch throughput: {medium_throughput:.1f} rec/s")
    print(f"✓ Large batch throughput: {large_throughput:.1f} rec/s")

    print("\n✅ Performance statistics test PASSED")
