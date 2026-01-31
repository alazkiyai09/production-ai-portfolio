"""
Message consumers for ingestion pipeline.

Supports both Redis and Kafka for message consumption.
Includes idempotent consumer pattern to prevent duplicate processing.
"""

import asyncio
import hashlib
import json
import logging
import time
from threading import RLock
from typing import AsyncIterator, Callable, Optional, Dict, Set

import redis.asyncio as redis
from pydantic import BaseModel

from src.ingestion.ingest_service import AdEvent

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class ConsumerConfig(BaseModel):
    """Consumer configuration."""
    broker_url: str = "redis://localhost:6379/0"
    queue_name: str = "ingestion:queue"
    consumer_group: str = "ingestion_consumers"
    consumer_name: str = "consumer_1"
    batch_size: int = 100
    timeout_ms: int = 5000
    max_retries: int = 3
    retry_delay_ms: int = 1000
    enable_idempotency: bool = True
    idempotency_ttl_seconds: int = 3600  # 1 hour default


# ============================================================================
# Idempotency Utilities
# ============================================================================

class ProcessedMessageCache:
    """
    Thread-safe cache for tracking processed message IDs.

    Uses time-based expiration to prevent memory leaks.
    """

    def __init__(self, ttl_seconds: int = 3600, cleanup_interval: int = 300):
        """
        Initialize the processed message cache.

        Args:
            ttl_seconds: Time-to-live for message IDs (default 1 hour)
            cleanup_interval: Interval between cleanup runs (default 5 minutes)
        """
        self._processed: Dict[str, float] = {}  # message_id -> timestamp
        self._lock = RLock()
        self._ttl = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cache entry has expired."""
        return time.time() - timestamp > self._ttl

    def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        now = time.time()

        # Only cleanup periodically
        if now - self._last_cleanup < self._cleanup_interval:
            return

        with self._lock:
            expired_keys = [
                msg_id for msg_id, timestamp in self._processed.items()
                if self._is_expired(timestamp)
            ]
            for msg_id in expired_keys:
                del self._processed[msg_id]

            self._last_cleanup = now

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired message IDs from cache")

    def is_processed(self, message_id: str) -> bool:
        """
        Check if a message has been processed.

        Args:
            message_id: Unique message identifier

        Returns:
            True if message was processed within TTL window
        """
        self._cleanup_expired()

        with self._lock:
            if message_id not in self._processed:
                return False

            # Check if entry is expired
            timestamp = self._processed[message_id]
            if self._is_expired(timestamp):
                del self._processed[message_id]
                return False

            return True

    def mark_processed(self, message_id: str) -> bool:
        """
        Mark a message as processed.

        Args:
            message_id: Unique message identifier

        Returns:
            True if message was newly marked (not previously processed)
        """
        self._cleanup_expired()

        with self._lock:
            # Check if already processed
            if message_id in self._processed:
                timestamp = self._processed[message_id]
                if not self._is_expired(timestamp):
                    return False  # Already processed

            # Mark as processed
            self._processed[message_id] = time.time()
            return True

    def get_size(self) -> int:
        """Get current cache size."""
        self._cleanup_expired()
        with self._lock:
            return len(self._processed)

    def clear(self) -> None:
        """Clear all processed message IDs."""
        with self._lock:
            self._processed.clear()


def generate_message_id(message: dict) -> str:
    """
    Generate a unique message ID from message content.

    Uses SHA-256 hash of message JSON for deterministic IDs.

    Args:
        message: Message dictionary

    Returns:
        Hex digest of SHA-256 hash
    """
    # Sort keys for deterministic serialization
    message_json = json.dumps(message, sort_keys=True)

    # Generate hash
    return hashlib.sha256(message_json.encode()).hexdigest()


def extract_message_id(message: dict, id_field: Optional[str] = None) -> Optional[str]:
    """
    Extract message ID from message.

    Args:
        message: Message dictionary
        id_field: Field name containing message ID (if None, uses hash)

    Returns:
        Message ID string or None
    """
    # Try to get explicit ID field
    if id_field:
        return str(message.get(id_field, "")) or None

    # Try common ID fields
    for field in ["message_id", "id", "event_id", "msg_id", "uuid"]:
        value = message.get(field)
        if value:
            return str(value)

    # Fallback to hash-based ID
    return generate_message_id(message)


# ============================================================================
# Idempotent Consumer
# ============================================================================

class IdempotentConsumer:
    """
    Wrapper for consumers that provides idempotent message processing.

    Prevents duplicate message processing by tracking processed message IDs.
    Uses time-based cache expiration to prevent memory leaks.
    """

    def __init__(
        self,
        consumer: "RedisConsumer",
        id_field: Optional[str] = None,
        ttl_seconds: int = 3600,
    ):
        """
        Initialize idempotent consumer.

        Args:
            consumer: Underlying consumer instance
            id_field: Field name containing message ID (None = auto-detect + hash)
            ttl_seconds: Time-to-live for processed message cache
        """
        self._consumer = consumer
        self._id_field = id_field
        self._cache = ProcessedMessageCache(ttl_seconds=ttl_seconds)

    async def connect(self):
        """Establish connection to underlying consumer."""
        return await self._consumer.connect()

    async def disconnect(self):
        """Close underlying consumer connection."""
        return await self._consumer.disconnect()

    async def consume(
        self,
        handler: Callable[[dict], None],
        batch_size: int = 1,
        timeout_ms: int = 5000,
    ) -> None:
        """
        Consume messages with idempotency guarantee.

        Args:
            handler: Async callback function to process each message
            batch_size: Number of messages to consume per batch
            timeout_ms: Timeout for blocking pop
        """
        # Wrap handler with idempotency check
        async def idempotent_handler(message: dict) -> None:
            message_id = extract_message_id(message, self._id_field)

            if message_id is None:
                logger.warning("Message has no ID, processing without idempotency check")
                await handler(message)
                return

            # Check if already processed
            if self._cache.is_processed(message_id):
                logger.debug(f"Skipping duplicate message: {message_id}")
                return

            # Mark as processed and handle
            is_new = self._cache.mark_processed(message_id)

            if is_new:
                try:
                    await handler(message)
                except Exception as e:
                    # On handler failure, remove from cache to allow retry
                    with self._cache._lock:
                        if message_id in self._cache._processed:
                            del self._cache._processed[message_id]
                    raise
            else:
                logger.debug(f"Skipping already processed message: {message_id}")

        # Delegate to underlying consumer
        await self._consumer.consume(idempotent_handler, batch_size, timeout_ms)

    async def consume_batch(
        self,
        batch_size: int = 100,
        timeout_ms: int = 5000,
    ) -> AsyncIterator[list[dict]]:
        """
        Consume messages as async iterator with idempotency filtering.

        Args:
            batch_size: Number of messages per batch
            timeout_ms: Timeout for each batch

        Yields:
            List of messages (duplicates removed)
        """
        async for batch in self._consumer.consume_batch(batch_size, timeout_ms):
            # Filter out duplicates
            filtered_batch = []
            for message in batch:
                message_id = extract_message_id(message, self._id_field)

                if message_id is None:
                    # No ID, include without filtering
                    filtered_batch.append(message)
                    continue

                # Check if already processed
                is_new = self._cache.mark_processed(message_id)
                if is_new:
                    filtered_batch.append(message)
                else:
                    logger.debug(f"Filtered duplicate message: {message_id}")

            if filtered_batch:
                yield filtered_batch

    async def read_message(self) -> Optional[dict]:
        """
        Read a single message (non-blocking) with duplicate check.

        Returns:
            Message dict or None if queue is empty or duplicate
        """
        message = await self._consumer.read_message()

        if message is None:
            return None

        message_id = extract_message_id(message, self._id_field)

        if message_id is None:
            return message

        # Check if already processed
        is_new = self._cache.mark_processed(message_id)
        if is_new:
            return message
        else:
            logger.debug(f"Skipping duplicate message: {message_id}")
            return None

    async def get_queue_size(self) -> int:
        """Get current queue size."""
        return await self._consumer.get_queue_size()

    def get_cache_size(self) -> int:
        """Get number of tracked message IDs in cache."""
        return self._cache.get_size()

    def clear_cache(self) -> None:
        """Clear all processed message IDs from cache."""
        self._cache.clear()


# ============================================================================
# Redis Consumer
# ============================================================================

class RedisConsumer:
    """
    Async Redis consumer for ingestion pipeline.

    Uses Redis lists as a simple queue (FIFO).
    """

    def __init__(
        self,
        broker_url: str = "redis://localhost:6379/0",
        queue_name: str = "ingestion:queue",
    ):
        """
        Initialize Redis consumer.

        Args:
            broker_url: Redis connection URL
            queue_name: Queue name to consume from
        """
        self.broker_url = broker_url
        self.queue_name = queue_name
        self._redis: Optional[redis.Redis] = None
        self._closed = False

    async def connect(self):
        """Establish Redis connection."""
        if self._redis is None:
            self._redis = await redis.from_url(
                self.broker_url,
                encoding="utf-8",
                decode_responses=True,
            )

    async def disconnect(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
        self._closed = True

    async def consume(
        self,
        handler: Callable[[dict], None],
        batch_size: int = 1,
        timeout_ms: int = 5000,
    ) -> None:
        """
        Consume messages from Redis queue.

        Args:
            handler: Async callback function to process each message
            batch_size: Number of messages to consume per batch
            timeout_ms: Timeout for blocking pop
        """
        await self.connect()

        while not self._closed:
            try:
                # Use BRPOP for blocking right-pop (FIFO)
                if batch_size == 1:
                    result = await self._redis.brpop(self.queue_name, timeout=timeout_ms // 1000)
                    if result:
                        _, message_json = result
                        try:
                            message = json.loads(message_json)
                            await handler(message)
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse message: {e}")
                else:
                    # Batch consumption
                    messages = []
                    for _ in range(batch_size):
                        result = await self._redis.brpop(self.queue_name, timeout=1)
                        if result:
                            _, message_json = result
                            try:
                                messages.append(json.loads(message_json))
                            except json.JSONDecodeError:
                                continue

                    if messages:
                        for message in messages:
                            await handler(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error consuming messages: {e}")
                await asyncio.sleep(1)

    async def consume_batch(
        self,
        batch_size: int = 100,
        timeout_ms: int = 5000,
    ) -> AsyncIterator[list[dict]]:
        """
        Consume messages as async iterator.

        Args:
            batch_size: Number of messages per batch
            timeout_ms: Timeout for each batch

        Yields:
            List of messages
        """
        await self.connect()

        while not self._closed:
            try:
                messages = []
                for _ in range(batch_size):
                    result = await self._redis.rpop(self.queue_name)
                    if result:
                        try:
                            messages.append(json.loads(result))
                        except json.JSONDecodeError:
                            continue

                if messages:
                    yield messages
                else:
                    # No messages, wait a bit
                    await asyncio.sleep(timeout_ms / 1000)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error consuming batch: {e}")
                await asyncio.sleep(1)

    async def read_message(self) -> Optional[dict]:
        """
        Read a single message (non-blocking).

        Returns:
            Message dict or None if queue is empty
        """
        await self.connect()
        result = await self._redis.rpop(self.queue_name)
        if result:
            return json.loads(result)
        return None

    async def get_queue_size(self) -> int:
        """Get current queue size."""
        await self.connect()
        return await self._redis.llen(self.queue_name)


# ============================================================================
# Kafka Consumer (Optional)
# ============================================================================

class KafkaConsumer:
    """
    Kafka consumer for ingestion pipeline.

    Requires kafka-python or aiokafka.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "adtech.raw.events",
        consumer_group: str = "ingestion_consumers",
    ):
        """
        Initialize Kafka consumer.

        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Topic to consume from
            consumer_group: Consumer group ID
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer_group = consumer_group
        self._consumer = None
        self._closed = False

    async def connect(self):
        """Establish Kafka connection."""
        # Placeholder for aiokafka implementation
        pass

    async def disconnect(self):
        """Close Kafka connection."""
        self._closed = True

    async def consume(self, handler: Callable[[dict], None]):
        """Consume messages from Kafka."""
        # Placeholder implementation
        raise NotImplementedError("Kafka consumer not fully implemented")


# ============================================================================
# Shared Client
# ============================================================================

_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> redis.Redis:
    """
    Get shared Redis client instance.

    Returns:
        Redis client

    Note:
        This is a simplified version. For production, use connection pooling.
    """
    global _redis_client

    import os

    redis_url = os.getenv(
        "REDIS_URL",
        os.getenv("REDIS_HOST", "redis://localhost:6379/0")
    )

    if _redis_client is None:
        _redis_client = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    return _redis_client


async def get_async_redis_client() -> redis.Redis:
    """
    Get async Redis client instance.

    Returns:
        Async Redis client
    """
    import os

    redis_url = os.getenv(
        "REDIS_URL",
        os.getenv("REDIS_HOST", "redis://localhost:6379/0")
    )

    return await redis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
    )


# ============================================================================
# Message Consumer Factory
# ============================================================================

class MessageConsumer:
    """
    Factory for creating message consumers.

    Supports both Redis and Kafka backends.
    Includes idempotent consumer support.
    """

    @staticmethod
    def create_redis_consumer(
        broker_url: str = "redis://localhost:6379/0",
        queue_name: str = "ingestion:queue",
        enable_idempotency: bool = False,
        id_field: Optional[str] = None,
        idempotency_ttl_seconds: int = 3600,
    ) -> RedisConsumer | IdempotentConsumer:
        """
        Create a Redis consumer.

        Args:
            broker_url: Redis connection URL
            queue_name: Queue name
            enable_idempotency: Enable idempotent message processing
            id_field: Field name for message ID (None = auto-detect)
            idempotency_ttl_seconds: TTL for processed message cache

        Returns:
            RedisConsumer or IdempotentConsumer instance
        """
        consumer = RedisConsumer(broker_url=broker_url, queue_name=queue_name)

        if enable_idempotency:
            return IdempotentConsumer(
                consumer=consumer,
                id_field=id_field,
                ttl_seconds=idempotency_ttl_seconds,
            )

        return consumer

    @staticmethod
    def create_kafka_consumer(
        bootstrap_servers: str = "localhost:9092",
        topic: str = "adtech.raw.events",
        consumer_group: str = "ingestion_consumers",
    ) -> KafkaConsumer:
        """
        Create a Kafka consumer.

        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Topic name
            consumer_group: Consumer group ID

        Returns:
            KafkaConsumer instance
        """
        return KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            topic=topic,
            consumer_group=consumer_group,
        )

    @staticmethod
    async def create_from_env(
        enable_idempotency: bool = False,
    ) -> RedisConsumer | KafkaConsumer | IdempotentConsumer:
        """
        Create consumer based on environment variables.

        Args:
            enable_idempotency: Enable idempotent message processing

        Returns:
            Consumer instance based on ENV configuration
        """
        import os

        backend = os.getenv("MESSAGE_QUEUE_BACKEND", "redis").lower()

        if backend == "kafka":
            return MessageConsumer.create_kafka_consumer(
                bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
                topic=os.getenv("KAFKA_TOPIC_RAW", "adtech.raw.events"),
                consumer_group=os.getenv("KAFKA_CONSUMER_GROUP", "ingestion_consumers"),
            )
        else:
            return MessageConsumer.create_redis_consumer(
                broker_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                queue_name=os.getenv("REDIS_QUEUE_NAME", "ingestion:queue"),
                enable_idempotency=enable_idempotency,
                id_field=os.getenv("MESSAGE_ID_FIELD"),
                idempotency_ttl_seconds=int(os.getenv("IDEMPOTENCY_TTL_SECONDS", "3600")),
            )


# ============================================================================
# Base Consumer Interface
# ============================================================================

class BaseConsumer:
    """Base interface for all consumers."""

    async def connect(self):
        """Establish connection."""
        raise NotImplementedError

    async def disconnect(self):
        """Close connection."""
        raise NotImplementedError

    async def consume(self, handler: Callable[[dict], None]):
        """Consume messages."""
        raise NotImplementedError


# Type alias for backward compatibility
MessageConsumer = RedisConsumer  # Default to Redis
