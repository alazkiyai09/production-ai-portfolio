"""
Integration test conftest for StreamProcess-Pipeline.

Provides fixtures for Docker Compose management and test helpers.
"""

import os
import subprocess
import time
from typing import AsyncGenerator, List

import httpx
import pytest
import redis.asyncio as redis
from sqlalchemy import create_engine
from datetime import datetime, timedelta

from sqlalchemy.orm import sessionmaker, Session


# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://streamprocess_user:streamprocess_pass@localhost:5432/streamprocess"
)

DOCKER_COMPOSE_FILE = "docker-compose.yml"
DOCKER_COMPOSE_TEST_FILE = "docker-compose.test.yml"


# ============================================================================
# Docker Compose Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def docker_compose_up():
    """
    Start Docker Compose services for integration tests.

    This fixture:
    1. Checks if services are already running
    2. If not, starts them with docker-compose
    3. Yields once services are healthy
    4. Tears down after all tests complete

    Usage:
        pytest tests/integration/  # Will auto-start compose
        or
        export DOCKER_COMPOSE_UP=false  # Skip auto-start
        pytest tests/integration/
    """
    # Check if services are already running
    if services_healthy():
        print("\n✓ Docker Compose services already running")
        yield
        return

    # Check if auto-start is disabled
    if os.getenv("DOCKER_COMPOSE_UP", "false").lower() == "false":
        pytest.skip("Docker Compose auto-start disabled. Start manually with: docker-compose up -d")

    print("\nStarting Docker Compose services...")
    subprocess.run(
        ["docker-compose", "up", "-d"],
        check=True,
    )

    # Wait for services to be healthy
    print("Waiting for services to be healthy...")
    max_wait = 120  # 2 minutes
    start_time = time.time()

    while time.time() - start_time < max_wait:
        if services_healthy():
            print("✓ All services are healthy")
            break
        time.sleep(2)
    else:
        pytest.fail("Services did not become healthy in time")

    yield

    # Cleanup happens after all tests
    print("\nStopping Docker Compose services...")
    subprocess.run(
        ["docker-compose", "down", "-v"],
        check=False,  # Don't fail if already down
    )


def services_healthy() -> bool:
    """Check if all required services are healthy."""
    try:
        # Check API
        response = httpx.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            return False

        # Check Redis
        redis_client = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
        redis_client.ping()

        # Check PostgreSQL
        engine = create_engine(POSTGRES_URL.replace("postgresql://", "postgresql+psycopg2://"))
        conn = engine.connect()
        conn.execute("SELECT 1")
        conn.close()
        engine.dispose()

        return True
    except Exception:
        return False


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def db_session() -> Session:
    """
    Create a database session for testing.

    The session is automatically closed after the test.
    """
    engine = create_engine(
        POSTGRES_URL.replace("postgresql://", "postgresql+psycopg2://"),
        pool_pre_ping=True,
    )

    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()
    engine.dispose()


@pytest.fixture(scope="function")
def clean_db(db_session: Session):
    """
    Clean database before/after test.

    Removes all test data to ensure test isolation.
    """
    # Clean before test
    db_session.query(EventRecord).delete()
    db_session.query(ProcessingStatus).delete()
    db_session.commit()

    yield

    # Clean after test
    db_session.query(EventRecord).delete()
    db_session.query(ProcessingStatus).delete()
    db_session.commit()


# ============================================================================
# Redis Fixtures
# ============================================================================

@pytest.fixture(scope="function")
async def redis_client() -> redis.Redis:
    """
    Create Redis client for testing.

    The client is automatically closed after the test.
    """
    client = await redis.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )

    yield client

    # Flush test data
    await client.flushdb()
    await client.close()


# ============================================================================
# API Client Fixtures
# ============================================================================

@pytest.fixture(scope="function")
async def api_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    Create async HTTP client for API calls.

    The client is automatically closed after the test.
    """
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    timeout = httpx.Timeout(60.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        yield client


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_campaigns() -> List[dict]:
    """Sample campaign data for testing."""
    return [
        {"id": "camp_01", "name": "Healthcare Awareness", "budget": 50000},
        {"id": "camp_02", "name": "Medical Services", "budget": 75000},
        {"id": "camp_03", "name": "Pharma Products", "budget": 100000},
    ]


@pytest.fixture
def sample_content_variations() -> List[str]:
    """Sample content variations for embedding tests."""
    return [
        "Premium healthcare services for you and your family.",
        "Affordable medical insurance plans available now.",
        "Expert doctors and nurses ready to help you.",
        "State-of-the-art medical facilities near you.",
        "Trusted healthcare provider since 1990.",
        "Your health is our top priority.",
        "Comprehensive care for all ages.",
        "Advanced treatments with proven results.",
        "Compassionate healthcare professionals.",
    ]


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def performance_thresholds():
    """Define performance thresholds for testing."""
    return {
        "ingestion_rate_min": 1000,  # records/second
        "processing_latency_max": 5.0,  # seconds for 1000 records
        "query_latency_max": 0.1,  # seconds
        "memory_usage_max": 2 * 1024 * 1024 * 1024,  # 2GB
        "cpu_usage_max": 0.9,  # 90%
    }


# ============================================================================
# Test Data Generator
# ============================================================================

class TestDataGenerator:
    """
    Generate test data for StreamProcess-Pipeline.

    Creates realistic AdTech event data with various properties.
    """

    @staticmethod
    def generate_event(
        event_id: str = None,
        event_type: str = "impression",
        timestamp: datetime = None,
        **kwargs
    ) -> dict:
        """
        Generate a single test event.

        Args:
            event_id: Optional event ID (auto-generated if None)
            event_type: Event type
            timestamp: Event timestamp (now if None)
            **kwargs: Additional event fields

        Returns:
            Event dictionary
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        if event_id is None:
            event_id = f"evt_{timestamp.timestamp()}_{hash(str(timestamp)) % 10000:04d}"

        event = {
            "event_id": event_id,
            "event_type": event_type,
            "timestamp": timestamp.isoformat() + "Z",
            "campaign_id": kwargs.get("campaign_id", "camp_01"),
            "user_id": kwargs.get("user_id", f"user_{hash(str(timestamp)) % 1000:04d}"),
            "content": kwargs.get(
                "content",
                "Default healthcare advertisement content for testing purposes."
            ),
            "metadata": kwargs.get(
                "metadata",
                {
                    "device_type": "mobile",
                    "os": "iOS",
                    "country": "US",
                    "region": "CA",
                }
            ),
        }

        return event

    @staticmethod
    def generate_batch(
        size: int = 100,
        start_time: datetime = None,
        event_type: str = "impression",
        **kwargs
    ) -> List[dict]:
        """
        Generate a batch of test events.

        Args:
            size: Batch size
            start_time: Start timestamp (events go backwards in time)
            event_type: Default event type
            **kwargs: Additional event fields

        Returns:
            List of event dictionaries
        """
        if start_time is None:
            start_time = datetime.utcnow()

        batch = []
        for i in range(size):
            event_time = start_time - timedelta(seconds=i)
            batch.append(TestDataGenerator.generate_event(
                event_type=event_type,
                timestamp=event_time,
                **kwargs
            ))

        return batch

    @staticmethod
    def generate_varied_batch(size: int = 100) -> List[dict]:
        """
        Generate a batch with varied event types and metadata.

        Args:
            size: Batch size

        Returns:
            List of event dictionaries
        """
        event_types = ["impression", "click", "conversion"]
        device_types = ["mobile", "desktop", "tablet"]
        countries = ["US", "UK", "CA", "DE", "FR"]

        batch = []
        for i in range(size):
            batch.append(TestDataGenerator.generate_event(
                event_type=event_types[i % len(event_types)],
                metadata={
                    "device_type": device_types[i % len(device_types)],
                    "country": countries[i % len(countries)],
                    "position": i % 100,
                }
            ))

        return batch


# ============================================================================
# Import required models
# ============================================================================

# Import at end to avoid circular dependencies
from src.storage.models import EventRecord, ProcessingStatus
