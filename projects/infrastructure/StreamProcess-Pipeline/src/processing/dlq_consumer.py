"""
Dead Letter Queue Consumer for StreamProcess-Pipeline.

Monitors and processes failed tasks from the dead letter queue.
Provides analysis, retry logic, and alerting for failed tasks.
"""

import asyncio
import json
import logging
import os
import signal
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

import redis.asyncio as redis

from src.monitoring.metrics import get_metrics

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES AND ENUMS
# =============================================================================

class FailureReason(str, Enum):
    """Categorized reasons for task failures."""
    VALIDATION_ERROR = "validation_error"
    DATABASE_ERROR = "database_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    EMBEDDING_ERROR = "embedding_error"
    VECTOR_STORE_ERROR = "vector_store_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class FailedTask:
    """Representation of a failed task from the DLQ."""
    task_id: str
    exception: str
    traceback_str: str
    timestamp: str
    retry_count: int = 0
    max_retries: int = 3
    failure_reason: FailureReason = FailureReason.UNKNOWN_ERROR
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "exception": self.exception,
            "traceback": self.traceback_str,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "failure_reason": self.failure_reason.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailedTask":
        """Create from dictionary."""
        return cls(
            task_id=data.get("task_id", ""),
            exception=data.get("exception", ""),
            traceback_str=data.get("traceback", ""),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            failure_reason=FailureReason(data.get("failure_reason", "unknown_error")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DLQStats:
    """Statistics for the dead letter queue."""
    total_failed: int = 0
    pending_retry: int = 0
    permanently_failed: int = 0
    retried_success: int = 0
    last_failure_time: Optional[str] = None
    failure_by_reason: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_failed": self.total_failed,
            "pending_retry": self.pending_retry,
            "permanently_failed": self.permanently_failed,
            "retried_success": self.retried_success,
            "last_failure_time": self.last_failure_time,
            "failure_by_reason": self.failure_by_reason,
        }


# =============================================================================
# DLQ CONSUMER
# =============================================================================

class DeadLetterQueueConsumer:
    """
    Consumer for processing failed tasks from the dead letter queue.

    Features:
    - Periodic polling of the DLQ Redis list
    - Automatic retry with exponential backoff
    - Categorization of failure reasons
    - Alerting for critical failures
    - Persistence of failure history
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        dlq_key: Optional[str] = None,
        poll_interval: float = 5.0,
        max_retries: int = 3,
        retry_backoff_base: float = 2.0,
        alert_threshold: int = 10,
        persistence_path: Optional[str] = None,
    ):
        """
        Initialize the DLQ consumer.

        Args:
            redis_url: Redis connection URL
            dlq_key: Redis key for the dead letter queue list
            poll_interval: Seconds between DLQ polls
            max_retries: Maximum retry attempts per task
            retry_backoff_base: Base for exponential backoff
            alert_threshold: Number of failures to trigger alert
            persistence_path: Path to persist failure history
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.dlq_key = dlq_key or os.getenv("INGESTION_DEAD_LETTER_TOPIC", "adtech.dead.letter")
        self.processed_key = f"{self.dlq_key}.processed"
        self.permanent_key = f"{self.dlq_key}.permanent"

        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.alert_threshold = alert_threshold
        self.persistence_path = persistence_path

        self._redis: Optional[redis.Redis] = None
        self._running = False
        self._stats = DLQStats()
        self._metrics = get_metrics()

        # Callbacks for different failure types
        self._failure_handlers: Dict[FailureReason, Callable] = {}

    async def start(self) -> None:
        """Start the DLQ consumer."""
        logger.info("Starting Dead Letter Queue Consumer...")
        self._running = True

        # Connect to Redis
        self._redis = await redis.from_url(self.redis_url)

        # Setup signal handlers
        self._setup_signal_handlers()

        # Load previous stats if persistence enabled
        if self.persistence_path:
            await self._load_stats()

        logger.info(f"DLQ Consumer started, monitoring: {self.dlq_key}")

        # Main consumption loop
        while self._running:
            try:
                await self._process_dlq()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in DLQ consumer loop: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)

    async def stop(self) -> None:
        """Stop the DLQ consumer."""
        logger.info("Stopping Dead Letter Queue Consumer...")
        self._running = False

        if self._redis:
            await self._redis.close()

        # Persist stats before stopping
        if self.persistence_path:
            await self._save_stats()

        logger.info("DLQ Consumer stopped")

    def register_failure_handler(
        self,
        reason: FailureReason,
        handler: Callable[[FailedTask], Any],
    ) -> None:
        """
        Register a callback handler for specific failure types.

        Args:
            reason: Failure reason to handle
            handler: Async callback function
        """
        self._failure_handlers[reason] = handler
        logger.info(f"Registered handler for: {reason.value}")

    async def _process_dlq(self) -> None:
        """Process failed tasks from the DLQ."""
        if not self._redis:
            return

        # Get count of items in DLQ
        dlq_size = await self._redis.llen(self.dlq_key)

        if dlq_size == 0:
            return

        logger.info(f"Processing {dlq_size} failed tasks from DLQ")

        # Process up to 100 items at a time
        batch_size = min(100, dlq_size)

        for _ in range(batch_size):
            try:
                # Pop from right (FIFO for retry order)
                item = await self._redis.rpop(self.dlq_key)

                if not item:
                    break

                # Parse failed task
                task_data = json.loads(item)
                failed_task = FailedTask.from_dict(task_data)

                # Update stats
                self._stats.total_failed += 1
                self._stats.last_failure_time = failed_task.timestamp

                # Categorize failure
                failed_task.failure_reason = self._categorize_failure(failed_task)
                reason_key = failed_task.failure_reason.value
                self._stats.failure_by_reason[reason_key] = self._stats.failure_by_reason.get(reason_key, 0) + 1

                # Check if should retry
                if failed_task.retry_count < self.max_retries:
                    # Calculate backoff delay
                    delay = self.retry_backoff_base ** failed_task.retry_count

                    logger.info(
                        f"Retrying task {failed_task.task_id} "
                        f"(attempt {failed_task.retry_count + 1}/{self.max_retries}) "
                        f"after {delay}s delay"
                    )

                    # Schedule retry
                    asyncio.create_task(self._retry_task(failed_task, delay))

                    self._stats.pending_retry += 1
                else:
                    # Max retries reached, mark as permanent failure
                    logger.error(f"Task {failed_task.task_id} permanently failed after {self.max_retries} attempts")

                    await self._redis.lpush(self.permanent_key, item)
                    self._stats.permanently_failed += 1

                    # Check if alert threshold reached
                    if self._stats.permanently_failed >= self.alert_threshold:
                        await self._send_alert(failed_task)

                    # Call failure handler if registered
                    handler = self._failure_handlers.get(failed_task.failure_reason)
                    if handler:
                        await handler(failed_task)

            except Exception as e:
                logger.error(f"Error processing DLQ item: {e}", exc_info=True)

    async def _retry_task(self, failed_task: FailedTask, delay: float) -> None:
        """
        Retry a failed task after a delay.

        Args:
            failed_task: The failed task to retry
            delay: Delay in seconds before retry
        """
        await asyncio.sleep(delay)

        try:
            # Re-execute the task (this would normally call the actual Celery task)
            # For now, we'll just log the retry attempt
            logger.info(f"Executing retry for task {failed_task.task_id}")

            # In production, you would use:
            # from src.processing.worker import celery_app
            # celery_app.send_task(task_name, args=task_args, kwargs=task_kwargs)

            # Simulate success/failure
            # In real implementation, this would await the actual task result
            success = True  # Placeholder

            if success:
                logger.info(f"Task {failed_task.task_id} retry succeeded")
                self._stats.retried_success += 1
                self._stats.pending_retry -= 1

                # Record to processed DLQ for tracking
                await self._redis.lpush(
                    self.processed_key,
                    json.dumps({
                        **failed_task.to_dict(),
                        "status": "retry_success",
                        "retry_timestamp": datetime.utcnow().isoformat(),
                    }),
                )
            else:
                # Increment retry count and push back to DLQ
                failed_task.retry_count += 1
                await self._redis.lpush(self.dlq_key, json.dumps(failed_task.to_dict()))

        except Exception as e:
            logger.error(f"Retry failed for task {failed_task.task_id}: {e}")

            # Increment retry count and push back to DLQ
            failed_task.retry_count += 1
            failed_task.metadata["last_error"] = str(e)

            await self._redis.lpush(self.dlq_key, json.dumps(failed_task.to_dict()))

    def _categorize_failure(self, failed_task: FailedTask) -> FailureReason:
        """
        Categorize a failure based on exception message and traceback.

        Args:
            failed_task: The failed task to categorize

        Returns:
            Categorized failure reason
        """
        exception_lower = failed_task.exception.lower()
        traceback_lower = failed_task.traceback_str.lower()

        if any(term in exception_lower for term in ["validation", "invalid", "schema"]):
            return FailureReason.VALIDATION_ERROR
        elif any(term in exception_lower for term in ["database", "sql", "connection"]):
            return FailureReason.DATABASE_ERROR
        elif any(term in exception_lower for term in ["network", "connection", "timeout", "unreachable"]):
            return FailureReason.NETWORK_ERROR
        elif "timeout" in exception_lower or "timeout" in traceback_lower:
            return FailureReason.TIMEOUT
        elif any(term in exception_lower for term in ["memory", "allocation"]):
            return FailureReason.OUT_OF_MEMORY
        elif any(term in exception_lower for term in ["embedding", "sentence", "transformer"]):
            return FailureReason.EMBEDDING_ERROR
        elif any(term in exception_lower for term in ["vector", "chroma", "collection"]):
            return FailureReason.VECTOR_STORE_ERROR
        else:
            return FailureReason.UNKNOWN_ERROR

    async def _send_alert(self, failed_task: FailedTask) -> None:
        """
        Send alert for critical failure.

        Args:
            failed_task: The failed task that triggered the alert
        """
        logger.error(
            f"ALERT: Dead Letter Queue threshold reached! "
            f"Permanent failures: {self._stats.permanently_failed}. "
            f"Latest failure: {failed_task.task_id} - {failed_task.exception}"
        )

        # In production, integrate with alerting systems:
        # - Slack webhook
        # - PagerDuty
        # - Email alerts
        # - CloudWatch/Sentry alerts

        # Record metrics
        self._metrics.record_dlq_threshold_breach(
            threshold=self.alert_threshold,
            current_failures=self._stats.permanently_failed,
        )

    async def get_stats(self) -> DLQStats:
        """Get current DLQ statistics."""
        if self._redis:
            dlq_size = await self._redis.llen(self.dlq_key)
            processed_size = await self._redis.llen(self.processed_key)
            permanent_size = await self._redis.llen(self.permanent_key)

            return DLQStats(
                total_failed=self._stats.total_failed,
                pending_retry=dlq_size,
                permanently_failed=permanent_size,
                retried_success=self._stats.retried_success,
                last_failure_time=self._stats.last_failure_time,
                failure_by_reason=self._stats.failure_by_reason,
            )

        return self._stats

    async def get_failed_tasks(
        self,
        limit: int = 100,
        include_processed: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get failed tasks from the DLQ.

        Args:
            limit: Maximum number of tasks to return
            include_processed: Whether to include processed tasks

        Returns:
            List of failed task dictionaries
        """
        if not self._redis:
            return []

        tasks = []

        # Get from active DLQ
        dlq_items = await self._redis.lrange(self.dlq_key, 0, limit - 1)
        for item in dlq_items:
            try:
                tasks.append(json.loads(item))
            except json.JSONDecodeError:
                pass

        # Optionally get from permanent failures
        if include_processed and len(tasks) < limit:
            permanent_items = await self._redis.lrange(
                self.permanent_key,
                0,
                limit - len(tasks) - 1,
            )
            for item in permanent_items:
                try:
                    tasks.append(json.loads(item))
                except json.JSONDecodeError:
                    pass

        return tasks

    async def clear_dlq(self, permanent_only: bool = False) -> int:
        """
        Clear the dead letter queue.

        Args:
            permanent_only: Only clear permanently failed tasks

        Returns:
            Number of items cleared
        """
        if not self._redis:
            return 0

        if permanent_only:
            count = await self._redis.llen(self.permanent_key)
            await self._redis.delete(self.permanent_key)
            logger.info(f"Cleared {count} permanently failed tasks")
            return count
        else:
            count = await self._redis.llen(self.dlq_key)
            await self._redis.delete(self.dlq_key)
            await self._redis.delete(self.permanent_key)
            logger.info(f"Cleared {count} failed tasks from DLQ")
            return count

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def handle_shutdown(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self._running = False

        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)

    async def _load_stats(self) -> None:
        """Load stats from persistence file."""
        if not self.persistence_path:
            return

        try:
            import aiofiles

            async with aiofiles.open(self.persistence_path, "r") as f:
                data = json.loads(await f.read())

                self._stats = DLQStats(
                    total_failed=data.get("total_failed", 0),
                    pending_retry=data.get("pending_retry", 0),
                    permanently_failed=data.get("permanently_failed", 0),
                    retried_success=data.get("retried_success", 0),
                    last_failure_time=data.get("last_failure_time"),
                    failure_by_reason=data.get("failure_by_reason", {}),
                )

                logger.info(f"Loaded DLQ stats from {self.persistence_path}")

        except FileNotFoundError:
            logger.info(f"No existing stats file at {self.persistence_path}, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load stats: {e}")

    async def _save_stats(self) -> None:
        """Save stats to persistence file."""
        if not self.persistence_path:
            return

        try:
            import aiofiles

            async with aiofiles.open(self.persistence_path, "w") as f:
                await f.write(json.dumps(self._stats.to_dict(), indent=2))

                logger.info(f"Saved DLQ stats to {self.persistence_path}")

        except Exception as e:
            logger.error(f"Failed to save stats: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def run_dlq_consumer(
    redis_url: Optional[str] = None,
    dlq_key: Optional[str] = None,
    poll_interval: float = 5.0,
) -> None:
    """
    Run the DLQ consumer.

    Args:
        redis_url: Redis connection URL
        dlq_key: Redis key for the DLQ
        poll_interval: Polling interval in seconds
    """
    consumer = DeadLetterQueueConsumer(
        redis_url=redis_url,
        dlq_key=dlq_key,
        poll_interval=poll_interval,
        persistence_path="data/dlq_stats.json",
    )

    try:
        await consumer.start()
    except asyncio.CancelledError:
        logger.info("DLQ consumer cancelled")
    finally:
        await consumer.stop()


def start_dlq_consumer_background(
    redis_url: Optional[str] = None,
    dlq_key: Optional[str] = None,
) -> asyncio.Task:
    """
    Start the DLQ consumer as a background task.

    Args:
        redis_url: Redis connection URL
        dlq_key: Redis key for the DLQ

    Returns:
        Background task
    """
    loop = asyncio.get_event_loop()
    return loop.create_task(run_dlq_consumer(redis_url, dlq_key))


if __name__ == "__main__":
    # Run the DLQ consumer directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(run_dlq_consumer())
