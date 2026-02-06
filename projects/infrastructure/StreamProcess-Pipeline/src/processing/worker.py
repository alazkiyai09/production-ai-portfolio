"""
Celery processing worker for StreamProcess-Pipeline.

Handles async processing of ingested batches:
- Data transformation and cleaning
- Embedding generation
- Vector store updates
- Metadata database updates
"""

import hashlib
import json
import os
import signal
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from celery import Celery, group, shared_task, chain
from celery.signals import worker_init, worker_ready, worker_shutdown, task_prerun, task_postrun, task_failure
from prometheus_client import Counter, Gauge, Histogram
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.database import get_db_manager
from src.storage.models import EventRecord, ProcessingStatus
from src.storage.repositories import EventRepository, StatusRepository
from src.monitoring.metrics import get_metrics


# ============================================================================
# Configuration
# ============================================================================

# Celery Configuration
celery_app = Celery("streamprocess")

# Get configuration from environment
celery_app.conf.update(
    # Broker settings
    broker_url=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
    result_backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"),

    # Task settings
    task_track_started=True,
    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "3600")),
    task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "3000")),
    task_acks_late=True,  # Ack after task completes
    task_reject_on_worker_lost=True,  # Re-queue if worker dies

    # Worker settings
    worker_prefetch_multiplier=int(os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER", "4")),
    worker_max_tasks_per_child=int(os.getenv("CELERY_WORKER_MAX_TASKS_PER_CHILD", "1000")),
    worker_concurrency=int(os.getenv("CELERY_WORKER_CONCURRENCY", "4")),

    # Result settings
    result_expires=3600,  # 1 hour
    result_extended=True,  # Allow extended result tracking

    # Task routing
    task_routes={
        "src.processing.worker.process_batch": {"queue": "processing"},
        "src.processing.worker.generate_embeddings": {"queue": "embedding"},
        "src.processing.worker.update_vector_store": {"queue": "vector_db"},
        "src.processing.worker.update_metadata_db": {"queue": "database"},
    },

    # Retry settings
    task_autoretry_for=(Exception,),  # Retry on all exceptions
    task_retry_kwargs={"max_retries": 3, "countdown": 60},

    # Serializer
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezones
    timezone="UTC",
    enable_utc=True,
)


# ============================================================================
# Metrics
# ============================================================================

worker_tasks_total = Counter(
    "worker_tasks_total",
    "Total worker tasks",
    ["task_name", "status"],
)

worker_task_duration_seconds = Histogram(
    "worker_task_duration_seconds",
    "Worker task duration",
    ["task_name"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0],
)

worker_batch_size = Histogram(
    "worker_batch_size",
    "Worker batch size distribution",
    buckets=[1, 10, 50, 100, 500, 1000],
)

worker_success_rate = Gauge(
    "worker_success_rate",
    "Worker task success rate",
)

worker_queue_size = Gauge(
    "worker_queue_size",
    "Current worker queue size",
    ["queue"],
)


# ============================================================================
# Global State
# ============================================================================

_embedding_model: Optional[SentenceTransformer] = None
_db_manager = None
_metrics = get_metrics()


# ============================================================================
# Celery Signal Handlers
# ============================================================================

@worker_init.connect
def worker_init_handler(sender=None, **kwargs):
    """Handle worker initialization."""
    import os
    worker_name = sender.hostname if sender else "unknown"
    print(f"[{worker_name}] Worker initializing...")

    # Initialize embedding model
    global _embedding_model
    try:
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        device = os.getenv("EMBEDDING_MODEL_DEVICE", "cpu")

        print(f"[{worker_name}] Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name, device=device)
        print(f"[{worker_name}] Embedding model loaded successfully")

        # Set model info in metrics
        _metrics.set_app_info(
            name="streamprocess-worker",
            version=os.getenv("APP_VERSION", "1.0.0"),
            environment=os.getenv("ENVIRONMENT", "development"),
        )
    except Exception as e:
        print(f"[{worker_name}] ERROR: Failed to load embedding model: {e}")
        traceback.print_exc()


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready."""
    worker_name = sender.hostname if sender else "unknown"
    print(f"[{worker_name}] Worker ready and waiting for tasks...")


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Handle worker shutdown."""
    worker_name = sender.hostname if sender else "unknown"
    print(f"[{worker_name}] Worker shutting down...")

    # Cleanup resources
    global _db_manager
    if _db_manager and _db_manager._engine:
        import asyncio
        asyncio.run(_db_manager.disconnect())


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    """Handle task pre-run."""
    worker_tasks_total.labels(task_name=task.name, status="started").inc()


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, retval=None, **kwargs):
    """Handle task post-run."""
    worker_tasks_total.labels(task_name=task.name, status="success").inc()


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, einfo=None, **kwargs):
    """
    Handle task failure.

    Sends failed task information to the dead letter queue for later analysis
    and potential retry processing.
    """
    task_name = sender.name if sender else "unknown"
    worker_tasks_total.labels(task_name=task_name, status="failed").inc()
    print(f"[{task_name}] Task {task_id} failed: {exception}")

    # Send to dead letter queue
    try:
        traceback_str = traceback.format_exception(type(exception), exception, exception.__traceback__) if exception else ""
        handle_failed_task.delay(
            task_id=task_id or "unknown",
            exception=str(exception) if exception else "Unknown error",
            traceback_str="".join(traceback_str),
        )
        print(f"[{task_name}] Task {task_id} sent to dead letter queue")
    except Exception as dlq_error:
        print(f"[{task_name}] Failed to send task {task_id} to DLQ: {dlq_error}")


# ============================================================================
# Helper Functions
# ============================================================================

def get_embedding_model() -> SentenceTransformer:
    """Get or initialize embedding model."""
    global _embedding_model

    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        device = os.getenv("EMBEDDING_MODEL_DEVICE", "cpu")
        _embedding_model = SentenceTransformer(model_name, device=device)

    return _embedding_model


def get_db_manager_sync():
    """Get database manager (sync wrapper)."""
    global _db_manager
    if _db_manager is None:
        _db_manager = get_db_manager()
    return _db_manager


def hash_content(content: str) -> str:
    """
    Generate SHA-256 hash of content.

    Args:
        content: Content string to hash

    Returns:
        Hex digest hash
    """
    return hashlib.sha256(content.encode()).hexdigest()


def transform_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform and clean event data.

    Args:
        event: Raw event data

    Returns:
        Transformed event
    """
    # Clean and normalize content
    content = event.get("content", "")
    content = " ".join(content.split())  # Normalize whitespace
    content = content[:10000]  # Truncate if too long

    # Transform metadata
    metadata = event.get("metadata", {})

    # Add derived fields
    transformed = {
        "event_id": event["event_id"],
        "event_type": event["event_type"],
        "timestamp": event["timestamp"],
        "campaign_id": event["campaign_id"],
        "user_id": event["user_id"],
        "content": content,
        "content_hash": hash_content(content),
        "metadata": metadata,
        "derived": {
            "content_length": len(content),
            "word_count": len(content.split()),
        },
    }

    return transformed


# ============================================================================
# Celery Tasks
# ============================================================================

@shared_task(
    name="src.processing.worker.process_batch",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 60},
)
def process_batch(self, batch_id: str) -> Dict[str, Any]:
    """
    Main processing task for a batch.

    Workflow:
    1. Fetch batch records from database/queue
    2. Transform and clean data
    3. Generate embeddings
    4. Update vector store
    5. Update metadata database

    Args:
        self: Celery task instance
        batch_id: Batch identifier

    Returns:
        Processing result with stats
    """
    import time
    import asyncio

    start_time = time.time()
    print(f"[process_batch] Processing batch: {batch_id}")

    try:
        # Create async event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run async processing
        result = loop.run_until_complete(_process_batch_async(batch_id))
        loop.close()

        # Record metrics
        duration = time.time() - start_time
        worker_task_duration_seconds.labels(task_name="process_batch").observe(duration)
        worker_batch_size.observe(result.get("total_records", 0))

        print(f"[process_batch] Completed {batch_id}: {result['processed_records']}/{result['total_records']} records in {duration:.2f}s")

        return result

    except Exception as e:
        print(f"[process_batch] ERROR processing batch {batch_id}: {e}")
        traceback.print_exc()

        # Update status to failed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_update_batch_status(batch_id, "failed", error_message=str(e)))
        loop.close()

        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)


async def _process_batch_async(batch_id: str) -> Dict[str, Any]:
    """
    Async implementation of batch processing.

    Args:
        batch_id: Batch identifier

    Returns:
        Processing result
    """
    db_manager = get_db_manager_sync()

    async with db_manager.session() as session:
        # Get batch status
        status_repo = StatusRepository(session)
        batch_status = await status_repo.get_by_batch_id(batch_id)

        if not batch_status:
            print(f"[_process_batch_async] Batch {batch_id} not found")
            return {"status": "not_found", "batch_id": batch_id}

        # Update to processing
        await status_repo.update_status(batch_id, "processing")

        # Fetch records from queue/database
        # In production, these would be fetched from the ingestion queue
        records = await _fetch_batch_records(session, batch_id)

        if not records:
            print(f"[_process_batch_async] No records found for batch {batch_id}")
            await status_repo.update_status(batch_id, "completed")
            return {"status": "completed", "batch_id": batch_id, "total_records": 0}

        # Transform records
        transformed_records = [transform_event(r) for r in records]

        # Extract texts for embedding
        texts = [r["content"] for r in transformed_records]

        # Generate embeddings
        embeddings = await _generate_embeddings_async(texts)

        # Update vector store
        vector_ids = await _update_vector_store_async(transformed_records, embeddings)

        # Update metadata database
        await _update_metadata_db_async(session, batch_id, transformed_records, vector_ids, "completed")

        return {
            "status": "completed",
            "batch_id": batch_id,
            "total_records": len(records),
            "processed_records": len(records),
            "failed_records": 0,
        }


async def _fetch_batch_records(session: AsyncSession, batch_id: str) -> List[Dict[str, Any]]:
    """
    Fetch records for a batch.

    In production, this would fetch from the Redis queue.
    For now, returns mock records.

    Args:
        session: Database session
        batch_id: Batch identifier

    Returns:
        List of event records
    """
    # Try to fetch from database
    event_repo = EventRepository(session)
    records = await event_repo.get_by_batch_id(batch_id)

    if records:
        # Convert ORM objects to dicts
        return [
            {
                "event_id": r.event_id,
                "event_type": r.event_type,
                "timestamp": r.timestamp.isoformat(),
                "campaign_id": r.campaign_id,
                "user_id": r.user_id,
                "content": r.content,
                "metadata": r.metadata,
            }
            for r in records
        ]

    # If no records in DB, return empty (they should be in queue)
    return []


async def _generate_embeddings_async(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings

    Returns:
        List of embedding vectors
    """
    import time

    start_time = time.time()

    # Get embedding model (runs in thread pool for CPU-bound work)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        model = get_embedding_model()
        embeddings = list(executor.submit(lambda: model.encode(texts, batch_size=32)).result())

    duration = time.time() - start_time

    # Record metrics
    _metrics.record_embedding_generation(
        model="sentence-transformers",
        cache_hit=False,
        duration=duration / len(texts),
        status="success",
    )

    return embeddings.tolist()


@shared_task(
    name="src.processing.worker.generate_embeddings",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 30},
)
def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for texts using sentence-transformers.

    Optimized for batch processing with GPU/CPU support.

    Args:
        self: Celery task instance
        texts: List of text strings

    Returns:
        List of embedding vectors (lists of floats)
    """
    import time

    if not texts:
        return []

    start_time = time.time()

    try:
        # Get model
        model = get_embedding_model()

        # Get batch size from config
        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

        # Generate embeddings
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)

        duration = time.time() - start_time

        # Record metrics
        worker_task_duration_seconds.labels(task_name="generate_embeddings").observe(duration)
        _metrics.record_embedding_generation(
            model="sentence-transformers",
            cache_hit=False,
            duration=duration / len(texts),
            status="success",
        )

        print(f"[generate_embeddings] Generated {len(embeddings)} embeddings in {duration:.2f}s ({len(embeddings)/duration:.1f} embeddings/sec)")

        return embeddings.tolist()

    except Exception as e:
        print(f"[generate_embeddings] ERROR: {e}")
        traceback.print_exc()
        raise self.retry(exc=e)


@shared_task(
    name="src.processing.worker.update_vector_store",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 30},
)
def update_vector_store(self, records: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
    """
    Upsert records to ChromaDB vector store.

    Args:
        self: Celery task instance
        records: List of transformed event records
        embeddings: List of embedding vectors

    Returns:
        List of vector IDs inserted/updated
    """
    import time
    import asyncio

    start_time = time.time()

    try:
        # Run async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        vector_ids = loop.run_until_complete(
            _update_vector_store_async(records, embeddings)
        )
        loop.close()

        duration = time.time() - start_time

        # Record metrics
        worker_task_duration_seconds.labels(task_name="update_vector_store").observe(duration)
        _metrics.record_vector_store_operation(
            operation="add",
            status="success",
            duration=duration / len(records) if records else 0,
        )

        print(f"[update_vector_store] Upserted {len(vector_ids)} vectors in {duration:.2f}s")

        return vector_ids

    except Exception as e:
        print(f"[update_vector_store] ERROR: {e}")
        traceback.print_exc()
        raise self.retry(exc=e)


async def _update_vector_store_async(
    records: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> List[str]:
    """
    Async implementation of vector store update.

    Args:
        records: List of transformed event records
        embeddings: List of embedding vectors

    Returns:
        List of vector IDs
    """
    from src.storage.vector_store import get_vector_store

    # Get vector store
    vector_store = await get_vector_store()

    # Prepare data for insertion
    ids = [r["event_id"] for r in records]
    documents = [r["content"] for r in records]
    metadatas = [
        {
            "event_id": r["event_id"],
            "event_type": r["event_type"],
            "campaign_id": r["campaign_id"],
            "timestamp": r["timestamp"],
            "content_hash": r.get("content_hash", ""),
            **r.get("metadata", {}),
        }
        for r in records
    ]

    # Add to vector store
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "adtech_embeddings")
    await vector_store.add_texts(
        collection_name=collection_name,
        texts=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    return ids


@shared_task(
    name="src.processing.worker.update_metadata_db",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 30},
)
def update_metadata_db(self, batch_id: str, records: List[Dict[str, Any]], vector_ids: List[str], status: str) -> Dict[str, int]:
    """
    Update PostgreSQL metadata database with processing status.

    Args:
        self: Celery task instance
        batch_id: Batch identifier
        records: List of processed records
        vector_ids: List of vector store IDs
        status: Processing status

    Returns:
        Dictionary with counts
    """
    import time
    import asyncio

    start_time = time.time()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            _update_metadata_db_async(None, batch_id, records, vector_ids, status)
        )
        loop.close()

        duration = time.time() - start_time
        worker_task_duration_seconds.labels(task_name="update_metadata_db").observe(duration)

        print(f"[update_metadata_db] Updated {result.get('updated', 0)} records in {duration:.2f}s")

        return result

    except Exception as e:
        print(f"[update_metadata_db] ERROR: {e}")
        traceback.print_exc()
        raise self.retry(exc=e)


async def _update_metadata_db_async(
    session: Optional[AsyncSession],
    batch_id: str,
    records: List[Dict[str, Any]],
    vector_ids: List[str],
    status: str,
) -> Dict[str, int]:
    """
    Async implementation of metadata update.

    Args:
        session: Database session (optional)
        batch_id: Batch identifier
        records: List of processed records
        vector_ids: List of vector store IDs
        status: Processing status

    Returns:
        Dictionary with update counts
    """
    from src.storage.database import get_db_manager

    db_manager = get_db_manager()

    async with db_manager.session() as sess:
        event_repo = EventRepository(sess)
        status_repo = StatusRepository(sess)

        # Update each event record
        updated = 0
        for record, vector_id in zip(records, vector_ids):
            # Check if event exists
            event = await event_repo.get_by_event_id(record["event_id"])

            if event:
                # Update with embedding ID
                await event_repo.update_embedding(record["event_id"], vector_id)
                await event_repo.mark_processed(record["event_id"])
                updated += 1
            else:
                # Create new record
                await event_repo.create(
                    event_id=record["event_id"],
                    event_type=record["event_type"],
                    timestamp=datetime.fromisoformat(record["timestamp"]) if isinstance(record["timestamp"], str) else record["timestamp"],
                    campaign_id=record["campaign_id"],
                    user_id=record["user_id"],
                    content=record["content"],
                    metadata=record.get("metadata", {}),
                    batch_id=batch_id,
                )
                updated += 1

        # Update batch status
        await status_repo.update_status(
            batch_id,
            status,
            processed_records=len(records),
        )

        return {"updated": updated, "batch_id": batch_id}


async def _update_batch_status(
    batch_id: str,
    status: str,
    error_message: Optional[str] = None,
) -> None:
    """
    Update batch status in database.

    Args:
        batch_id: Batch identifier
        status: New status
        error_message: Optional error message
    """
    from src.storage.database import get_db_manager

    db_manager = get_db_manager()

    async with db_manager.session() as session:
        status_repo = StatusRepository(session)
        await status_repo.update_status(
            batch_id,
            status,
            error_message=error_message,
        )


# ============================================================================
# Dead Letter Queue Handler
# ============================================================================

@shared_task(
    name="src.processing.worker.handle_failed_task",
)
def handle_failed_task(task_id: str, exception: str, traceback_str: str) -> None:
    """
    Handle failed tasks by sending to dead letter queue.

    Args:
        task_id: Failed task ID
        exception: Exception message
        traceback_str: Traceback string
    """
    import redis.asyncio as redis

    print(f"[handle_failed_task] Task {task_id} failed: {exception}")

    # Store failure info in Redis
    failure_info = {
        "task_id": task_id,
        "exception": exception,
        "traceback": traceback_str,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Add to dead letter queue
    dlq_key = os.getenv("INGESTION_DEAD_LETTER_TOPIC", "adtech.dead.letter")

    # Store as async (but run in sync context for Celery)
    import asyncio
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    async def _push_to_dlq():
        client = await redis.from_url(redis_url)
        await client.lpush(dlq_key, json.dumps(failure_info))
        await client.close()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_push_to_dlq())
    loop.close()


# ============================================================================
# Graceful Shutdown
# ============================================================================

def setup_graceful_shutdown():
    """Setup graceful shutdown handlers."""
    def handle_shutdown(signum, frame):
        print(f"\n[SHUTDOWN] Received signal {signum}, shutting down gracefully...")
        # Celery handles the rest
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)


# ============================================================================
# Main Entry Points
# ============================================================================

def main():
    """Main entry point for running worker directly."""
    setup_graceful_shutdown()

    # Start worker
    worker = celery_app.worker
    worker.start()


def flower_main():
    """Main entry point for Flower monitoring."""
    from celery import flower

    flower_app = flower.Flower(
        app=celery_app,
        broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
        port=int(os.getenv("FLOWER_PORT", "5555")),
    )

    flower_app.start()


if __name__ == "__main__":
    main()
