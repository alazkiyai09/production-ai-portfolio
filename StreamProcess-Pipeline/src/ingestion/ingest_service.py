"""
High-throughput ingestion service for StreamProcess-Pipeline.

Handles async ingestion of AdTech events with validation, queuing, and metrics.
"""

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import redis.asyncio as redis
from fastapi import APIRouter, Depends, HTTPException, Request, status
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from src.monitoring.metrics import get_metrics
from src.storage.database import get_db
from src.storage.models import EventRecord, ProcessingStatus


# ============================================================================
# Metrics
# ============================================================================

ingestion_requests_total = Counter(
    "ingestion_requests_total",
    "Total number of ingestion requests",
    ["endpoint", "status"]
)

ingestion_records_total = Counter(
    "ingestion_records_total",
    "Total number of records ingested",
    ["event_type"]
)

ingestion_batches_total = Counter(
    "ingestion_batches_total",
    "Total number of batches processed",
    ["status"]
)

ingestion_batch_size = Histogram(
    "ingestion_batch_size",
    "Batch size distribution",
    buckets=[1, 10, 50, 100, 500, 1000]
)

ingestion_rejection_rate = Counter(
    "ingestion_rejections_total",
    "Total number of rejected records",
    ["reason"]
)

processing_queue_size = Gauge(
    "processing_queue_size",
    "Current size of processing queue"
)

ingestion_duration_seconds = Histogram(
    "ingestion_duration_seconds",
    "Ingestion request duration",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)


# ============================================================================
# Enums and Constants
# ============================================================================

class EventType(str, Enum):
    """Supported event types."""
    IMPRESSION = "impression"
    CLICK = "click"
    CONVERSION = "conversion"


class BatchStatus(str, Enum):
    """Batch processing status."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class RejectionReason(str, Enum):
    """Reasons for record rejection."""
    INVALID_SCHEMA = "invalid_schema"
    DUPLICATE = "duplicate"
    MISSING_REQUIRED = "missing_required"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMITED = "rate_limited"


# ============================================================================
# Pydantic Models
# ============================================================================

class AdEvent(BaseModel):
    """AdTech event model for ingestion."""
    event_id: str = Field(..., min_length=1, max_length=100, description="Unique event identifier")
    event_type: EventType = Field(..., description="Type of event")
    timestamp: datetime = Field(..., description="Event timestamp")
    campaign_id: str = Field(..., min_length=1, max_length=50, description="Campaign identifier")
    user_id: str = Field(..., min_length=1, max_length=100, description="Hashed user identifier")
    content: str = Field(..., min_length=1, max_length=10000, description="Ad creative text for embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, v: str) -> str:
        """Validate event ID format."""
        if not v or v.strip() != v:
            raise ValueError("event_id cannot be empty or have leading/trailing whitespace")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Validate timestamp is not too far in the future or past."""
        now = datetime.utcnow()
        if v > now + timedelta(minutes=5):
            raise ValueError("timestamp cannot be more than 5 minutes in the future")
        if v < now - timedelta(days=30):
            raise ValueError("timestamp cannot be more than 30 days in the past")
        return v

    @field_validator("campaign_id")
    @classmethod
    def validate_campaign_id(cls, v: str) -> str:
        """Validate campaign ID."""
        if not v or not v.strip():
            raise ValueError("campaign_id cannot be empty")
        return v.strip()

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        """Validate user ID is hashed (no PII)."""
        if not v or len(v) < 10:
            raise ValueError("user_id appears to be invalid (too short)")
        # Basic check for hash-like format (hex or base64)
        if not all(c in "0123456789abcdefABCDEF-" for c in v):
            # Could be base64 or other encoding, just warn but allow
            pass
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("content cannot be empty or whitespace")
        return v.strip()

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "event_id": "evt_abc123xyz",
                "event_type": "impression",
                "timestamp": "2026-01-30T12:34:56Z",
                "campaign_id": "camp_456",
                "user_id": "a1b2c3d4e5f6",
                "content": "Amazing product - limited time offer!",
                "metadata": {
                    "device_type": "mobile",
                    "country": "US",
                    "placement": "homepage_banner"
                }
            }
        }


class BatchIngestRequest(BaseModel):
    """Batch ingestion request."""
    records: List[AdEvent] = Field(..., min_length=1, max_length=1000, description="Batch of events")

    @field_validator("records")
    @classmethod
    def validate_records(cls, v: List[AdEvent]) -> List[AdEvent]:
        """Validate batch size and check for duplicate event_ids within batch."""
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 records")

        # Check for duplicate event_ids within the batch
        event_ids = [r.event_id for r in v]
        duplicates = [eid for eid in event_ids if event_ids.count(eid) > 1]
        if duplicates:
            unique_duplicates = list(set(duplicates))
            raise ValueError(f"Duplicate event_ids found in batch: {unique_duplicates[:5]}")

        return v


class BatchIngestResponse(BaseModel):
    """Batch ingestion response."""
    batch_id: str = Field(..., description="Unique batch identifier")
    status: BatchStatus = Field(..., description="Batch status")
    record_count: int = Field(..., description="Number of records in batch")
    queued_at: datetime = Field(..., description="Timestamp when batch was queued")
    estimated_processing_time_seconds: Optional[float] = Field(
        None, description="Estimated time to complete processing"
    )


class SingleIngestRequest(BaseModel):
    """Single event ingestion request."""
    record: AdEvent = Field(..., description="Single event record")


class SingleIngestResponse(BaseModel):
    """Single event ingestion response."""
    event_id: str = Field(..., description="Event identifier")
    batch_id: str = Field(..., description="Batch identifier (single event batch)")
    status: BatchStatus = Field(..., description="Processing status")
    queued_at: datetime = Field(..., description="Timestamp when queued")


class BatchStatusResponse(BaseModel):
    """Batch status response."""
    batch_id: str = Field(..., description="Batch identifier")
    status: BatchStatus = Field(..., description="Current status")
    total_records: int = Field(..., description="Total records in batch")
    processed_records: int = Field(..., description="Number of records processed")
    failed_records: int = Field(..., description="Number of failed records")
    created_at: datetime = Field(..., description="Batch creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Processing start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class ValidationError(BaseModel):
    """Validation error details."""
    record_index: int = Field(..., description="Index of invalid record in batch")
    event_id: Optional[str] = Field(None, description="Event ID if available")
    reason: str = Field(..., description="Validation failure reason")
    details: Optional[str] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    validation_errors: Optional[List[ValidationError]] = Field(
        None, description="Validation error details"
    )


# ============================================================================
# Ingest Service
# ============================================================================

class IngestService:
    """
    High-throughput ingestion service for AdTech events.

    Features:
    - Batch validation with detailed error reporting
    - Deduplication check against existing events
    - Rate limiting per client
    - Queue messages to Redis for async processing
    - Comprehensive metrics collection
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        db_session: AsyncSession,
        queue_name: str = "ingestion:queue",
        dedupe_window_hours: int = 24,
        rate_limit_per_minute: int = 10000,
    ):
        """
        Initialize the ingestion service.

        Args:
            redis_client: Async Redis client
            db_session: Async database session
            queue_name: Redis queue name for processed events
            dedupe_window_hours: Hours to look back for duplicates
            rate_limit_per_minute: Max records per minute per client
        """
        self.redis = redis_client
        self.db = db_session
        self.queue_name = queue_name
        self.dedupe_window = timedelta(hours=dedupe_window_hours)
        self.rate_limit = rate_limit_per_minute

    async def validate_batch(
        self,
        records: List[AdEvent],
        client_id: Optional[str] = None,
    ) -> Tuple[bool, List[ValidationError]]:
        """
        Validate a batch of records.

        Checks:
        - Pydantic schema validation (done before this method)
        - Duplicate event_ids within database
        - Rate limiting
        - Required fields presence

        Args:
            records: List of event records to validate
            client_id: Optional client identifier for rate limiting

        Returns:
            Tuple of (is_valid, validation_errors)
        """
        validation_errors = []

        # Check rate limit if client_id provided
        if client_id:
            is_allowed, current_count = await self._check_rate_limit(client_id)
            if not is_allowed:
                validation_errors.append(
                    ValidationError(
                        record_index=0,
                        reason=RejectionReason.RATE_LIMITED,
                        details=f"Rate limit exceeded: {current_count}/{self.rate_limit} per minute"
                    )
                )
                return False, validation_errors

        # Check for duplicates in database
        existing_event_ids = await self._check_duplicates(records)
        if existing_event_ids:
            for idx, record in enumerate(records):
                if record.event_id in existing_event_ids:
                    validation_errors.append(
                        ValidationError(
                            record_index=idx,
                            event_id=record.event_id,
                            reason=RejectionReason.DUPLICATE,
                            details="Event ID already exists in database"
                        )
                    )

        # Additional custom validation
        for idx, record in enumerate(records):
            # Validate metadata doesn't contain sensitive keys
            sensitive_keys = ["email", "phone", "ssn", "credit_card", "password"]
            found_sensitive = []
            for key in record.metadata.keys():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    found_sensitive.append(key)

            if found_sensitive:
                validation_errors.append(
                    ValidationError(
                        record_index=idx,
                        event_id=record.event_id,
                        reason=RejectionReason.VALIDATION_ERROR,
                        details=f"Potentially sensitive fields in metadata: {found_sensitive}"
                    )
                )

        is_valid = len(validation_errors) == 0
        return is_valid, validation_errors

    async def queue_for_processing(
        self,
        records: List[AdEvent],
        client_id: Optional[str] = None,
    ) -> str:
        """
        Queue a batch of records for processing.

        Args:
            records: Validated list of event records
            client_id: Optional client identifier for tracking

        Returns:
            batch_id: Unique batch identifier
        """
        # Generate batch ID
        batch_id = str(uuid.uuid4())

        # Create batch metadata
        batch_metadata = {
            "batch_id": batch_id,
            "status": BatchStatus.QUEUED.value,
            "total_records": len(records),
            "processed_records": 0,
            "failed_records": 0,
            "client_id": client_id,
            "created_at": datetime.utcnow().isoformat(),
            "event_types": {r.event_type.value for r in records},
            "campaign_ids": {r.campaign_id for r in records},
        }

        # Store batch status in Redis (with 24h TTL)
        batch_key = f"batch:{batch_id}"
        await self.redis.setex(
            batch_key,
            86400,  # 24 hours
            json.dumps(batch_metadata)
        )

        # Queue each record for processing
        pipe = self.redis.pipeline()
        for record in records:
            # Create message payload
            message = {
                "batch_id": batch_id,
                "event_id": record.event_id,
                "event_data": record.model_dump(),
            }
            # Push to Redis list (queue)
            pipe.lpush(self.queue_name, json.dumps(message))

        # Execute pipeline
        await pipe.execute()

        # Update queue size metric
        queue_size = await self.redis.llen(self.queue_name)
        processing_queue_size.set(queue_size)

        # Store event IDs in dedupe cache (for 24h)
        dedupe_key = f"events:{datetime.utcnow().strftime('%Y%m%d')}"
        event_ids = [r.event_id for r in records]
        await self.redis.sadd(dedupe_key, *event_ids)
        await self.redis.expire(dedupe_key, int(self.dedupe_window.total_seconds()))

        # Update metrics
        ingestion_batches_total.labels(status=BatchStatus.QUEUED.value).inc()
        ingestion_batch_size.observe(len(records))
        for record in records:
            ingestion_records_total.labels(event_type=record.event_type.value).inc()

        return batch_id

    async def get_batch_status(self, batch_id: str) -> Optional[BatchStatusResponse]:
        """
        Get the processing status of a batch.

        Args:
            batch_id: Batch identifier

        Returns:
            BatchStatusResponse or None if not found
        """
        # Check Redis cache first
        batch_key = f"batch:{batch_id}"
        batch_data = await self.redis.get(batch_key)

        if batch_data:
            metadata = json.loads(batch_data)
            return BatchStatusResponse(
                batch_id=metadata["batch_id"],
                status=BatchStatus(metadata["status"]),
                total_records=metadata["total_records"],
                processed_records=metadata.get("processed_records", 0),
                failed_records=metadata.get("failed_records", 0),
                created_at=datetime.fromisoformat(metadata["created_at"]),
                started_at=datetime.fromisoformat(metadata["started_at"]) if metadata.get("started_at") else None,
                completed_at=datetime.fromisoformat(metadata["completed_at"]) if metadata.get("completed_at") else None,
                error_message=metadata.get("error_message"),
            )

        # If not in Redis, check database
        # This would query a batch tracking table (simplified here)
        return None

    async def _check_rate_limit(self, client_id: str) -> Tuple[bool, int]:
        """
        Check if client has exceeded rate limit.

        Uses sliding window counter in Redis.

        Args:
            client_id: Client identifier

        Returns:
            Tuple of (is_allowed, current_count)
        """
        now = datetime.utcnow()
        window_key = f"ratelimit:{client_id}:{now.strftime('%Y%m%d%H%M')}"

        # Increment counter
        current_count = await self.redis.incr(window_key)

        # Set expiration on first increment
        if current_count == 1:
            await self.redis.expire(window_key, 60)

        return current_count <= self.rate_limit, current_count

    async def _check_duplicates(self, records: List[AdEvent]) -> set[str]:
        """
        Check for existing event IDs in database.

        Uses Redis set for fast lookup (populated by recent events).

        Args:
            records: Records to check

        Returns:
            Set of duplicate event_ids
        """
        event_ids = {r.event_id for r in records}

        # Check today's dedupe cache
        today_key = f"events:{datetime.utcnow().strftime('%Y%m%d')}"
        existing = await self.redis.smembers(today_key)

        duplicates = set()
        for eid in event_ids:
            if eid.encode() in existing:
                duplicates.add(eid)

        return duplicates

    async def estimate_processing_time(self, batch_size: int) -> float:
        """
        Estimate processing time based on queue size and batch size.

        Simple heuristic: ~100 records/second per worker.

        Args:
            batch_size: Size of the batch

        Returns:
            Estimated processing time in seconds
        """
        queue_size = await self.redis.llen(self.queue_name)
        # Assume 4 workers processing 100 records/sec each
        processing_rate = 400  # records/second
        total_records = queue_size + batch_size
        estimated_seconds = total_records / processing_rate
        return estimated_seconds


# ============================================================================
# FastAPI Router
# ============================================================================

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Dependency to get ingest service
async def get_ingest_service(
    db: AsyncSession = Depends(get_db),
) -> IngestService:
    """Get IngestService instance."""
    from src.ingestion.consumer import get_redis_client

    redis_client = get_redis_client()
    return IngestService(redis_client, db)


def get_client_id(request: Request) -> Optional[str]:
    """Extract client ID from request headers or IP."""
    # Try API key header first
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"

    # Fall back to client IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"

    return f"ip:{request.client.host}"


@router.post(
    "/batch",
    response_model=BatchIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    summary="Ingest a batch of events",
    description="Accept up to 1000 AdTech events for async processing. Returns batch_id for tracking.",
)
async def ingest_batch(
    request: BatchIngestRequest,
    ingest_service: IngestService = Depends(get_ingest_service),
    client_id: str = Depends(get_client_id),
):
    """
    Ingest a batch of AdTech events.

    - **records**: List of event records (max 1000)
    - **Returns**: Batch ID for tracking processing status
    - **Validation**: Entire batch is validated; rejects if any record is invalid
    - **Deduplication**: Checks for duplicate event_ids
    - **Rate Limiting**: Enforced per client
    """
    import time
    start_time = time.time()

    try:
        # Validate batch
        is_valid, validation_errors = await ingest_service.validate_batch(
            request.records,
            client_id
        )

        if not is_valid:
            ingestion_requests_total.labels(endpoint="/ingest/batch", status="rejected").inc()
            ingestion_rejection_rate.labels(reason=validation_errors[0].reason).inc()

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "ValidationError",
                    "message": "Batch validation failed",
                    "validation_errors": [e.model_dump() for e in validation_errors]
                }
            )

        # Queue for processing
        batch_id = await ingest_service.queue_for_processing(
            request.records,
            client_id
        )

        # Estimate processing time
        estimated_time = await ingest_service.estimate_processing_time(len(request.records))

        # Record metrics
        duration = time.time() - start_time
        ingestion_duration_seconds.labels(endpoint="/ingest/batch").observe(duration)
        ingestion_requests_total.labels(endpoint="/ingest/batch", status="accepted").inc()

        return BatchIngestResponse(
            batch_id=batch_id,
            status=BatchStatus.QUEUED,
            record_count=len(request.records),
            queued_at=datetime.utcnow(),
            estimated_processing_time_seconds=estimated_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        ingestion_requests_total.labels(endpoint="/ingest/batch", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalError", "message": str(e)}
        )


@router.post(
    "/single",
    response_model=SingleIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    summary="Ingest a single event",
    description="Accept a single AdTech event for async processing.",
)
async def ingest_single(
    request: SingleIngestRequest,
    ingest_service: IngestService = Depends(get_ingest_service),
    client_id: str = Depends(get_client_id),
):
    """
    Ingest a single AdTech event.

    Convenience endpoint that wraps the single event in a batch.
    """
    import time
    start_time = time.time()

    try:
        # Validate
        is_valid, validation_errors = await ingest_service.validate_batch(
            [request.record],
            client_id
        )

        if not is_valid:
            ingestion_requests_total.labels(endpoint="/ingest/single", status="rejected").inc()
            ingestion_rejection_rate.labels(reason=validation_errors[0].reason).inc()

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "ValidationError",
                    "message": validation_errors[0].details,
                    "validation_errors": [validation_errors[0].model_dump()]
                }
            )

        # Queue for processing
        batch_id = await ingest_service.queue_for_processing(
            [request.record],
            client_id
        )

        # Record metrics
        duration = time.time() - start_time
        ingestion_duration_seconds.labels(endpoint="/ingest/single").observe(duration)
        ingestion_requests_total.labels(endpoint="/ingest/single", status="accepted").inc()

        return SingleIngestResponse(
            event_id=request.record.event_id,
            batch_id=batch_id,
            status=BatchStatus.QUEUED,
            queued_at=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        ingestion_requests_total.labels(endpoint="/ingest/single", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalError", "message": str(e)}
        )


@router.get(
    "/status/{batch_id}",
    response_model=BatchStatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Batch not found"},
    },
    summary="Get batch processing status",
    description="Query the current status of a previously submitted batch.",
)
async def get_batch_status(
    batch_id: str,
    ingest_service: IngestService = Depends(get_ingest_service),
):
    """
    Get the processing status of a batch.

    Returns detailed status including progress counts and timestamps.
    """
    try:
        status = await ingest_service.get_batch_status(batch_id)

        if status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "NotFound", "message": f"Batch {batch_id} not found"}
            )

        return status

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "InternalError", "message": str(e)}
        )


@router.get(
    "/metrics",
    summary="Get ingestion metrics",
    description="Returns current ingestion metrics (record counts, rates, etc.)",
)
async def get_ingestion_metrics(
    ingest_service: IngestService = Depends(get_ingest_service),
):
    """
    Get current ingestion metrics.

    Returns summary metrics about the ingestion service.
    """
    queue_size = await ingest_service.redis.llen(ingest_service.queue_name)

    return {
        "queue_size": queue_size,
        "metrics": {
            "requests_total": ingestion_requests_total._value.get(),
            "records_total": ingestion_records_total._value.get(),
            "batches_total": ingestion_batches_total._value.get(),
            "rejection_rate": ingestion_rejection_rate._value.get(),
        }
    }


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health", summary="Health check")
async def health_check():
    """Health check endpoint for the ingestion service."""
    return {"status": "healthy", "service": "ingestion"}
