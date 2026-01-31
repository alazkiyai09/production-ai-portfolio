"""
Pytest configuration and shared fixtures.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Set test environment before importing app
os.environ["ENVIRONMENT"] = "test"
os.environ["DEBUG"] = "true"
os.environ["LOG_LEVEL"] = "WARNING"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_db_url(temp_dir: Path) -> str:
    """Get test database URL."""
    return f"sqlite:///{temp_dir}/test.db"


@pytest.fixture
def test_redis_url() -> str:
    """Get test Redis URL."""
    return os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture
def sample_adtech_event() -> dict:
    """Sample AdTech event for testing."""
    return {
        "event_id": "evt_123456789",
        "timestamp": "2026-01-30T12:34:56Z",
        "event_type": "ad_impression",
        "user_id": "user_abc123",
        "session_id": "sess_xyz789",
        "campaign_id": "camp_456",
        "ad_group_id": "adgroup_789",
        "creative_id": "creative_101",
        "publisher_id": "pub_202",
        "placement": "homepage_banner",
        "device_type": "mobile",
        "os": "iOS",
        "browser": "Safari",
        "country": "US",
        "region": "CA",
        "city": "San Francisco",
        "referrer": "https://google.com",
        "url": "https://example.com/page",
        "impressions": 1,
        "clicks": 0,
        "cost_cents": 50,
        "revenue_cents": 0,
        "custom_attributes": {
            "brand_safety_score": 95,
            "viewability": 88,
        },
    }


@pytest.fixture
def sample_adtech_events() -> list[dict]:
    """Sample batch of AdTech events for testing."""
    events = []
    for i in range(100):
        events.append(
            {
                "event_id": f"evt_{i}",
                "timestamp": "2026-01-30T12:34:56Z",
                "event_type": "ad_impression",
                "user_id": f"user_{i % 10}",
                "session_id": f"sess_{i // 10}",
                "campaign_id": f"camp_{i % 5}",
                "ad_group_id": f"adgroup_{i % 10}",
                "creative_id": f"creative_{i % 20}",
                "publisher_id": f"pub_{i % 15}",
                "placement": f"placement_{i % 8}",
                "device_type": ["mobile", "desktop", "tablet"][i % 3],
                "os": ["iOS", "Android", "Windows", "macOS"][i % 4],
                "country": ["US", "UK", "CA", "DE", "FR"][i % 5],
                "impressions": 1,
                "clicks": i % 2,
                "cost_cents": 30 + (i % 10) * 10,
                "revenue_cents": (i % 3) * 50,
                "custom_attributes": {
                    "brand_safety_score": 80 + (i % 20),
                    "viewability": 70 + (i % 30),
                },
            }
        )
    return events


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing."""
    from src.api.main import app

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
