"""
Message producers for ingestion pipeline.

Supports both Redis and Kafka for message publishing.
"""

import json
from typing import List, Optional

import redis.asyncio as redis
from pydantic import BaseModel


# ============================================================================
# Configuration
# ============================================================================

class ProducerConfig(BaseModel):
    """Producer configuration."""
    broker_url: str = "redis://localhost:6379/0"
    topic_or_queue: str = "ingestion:queue"
    max_retries: int = 3
    timeout_ms: int = 5000


# ============================================================================
# Redis Producer
# ============================================================================

class RedisProducer:
    """
    Async Redis producer for publishing messages.
    """

    def __init__(
        self,
        broker_url: str = "redis://localhost:6379/0",
        queue_name: str = "ingestion:queue",
    ):
        """
        Initialize Redis producer.

        Args:
            broker_url: Redis connection URL
            queue_name: Queue name to publish to
        """
        self.broker_url = broker_url
        self.queue_name = queue_name
        self._redis: Optional[redis.Redis] = None

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

    async def publish(self, message: dict) -> bool:
        """
        Publish a single message.

        Args:
            message: Message dictionary to publish

        Returns:
            True if successful
        """
        await self.connect()
        try:
            message_json = json.dumps(message)
            await self._redis.lpush(self.queue_name, message_json)
            return True
        except Exception as e:
            print(f"Failed to publish message: {e}")
            return False

    async def publish_batch(self, messages: List[dict]) -> int:
        """
        Publish a batch of messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Number of messages successfully published
        """
        await self.connect()
        try:
            pipe = self._redis.pipeline()
            for message in messages:
                message_json = json.dumps(message)
                pipe.lpush(self.queue_name, message_json)

            results = await pipe.execute()
            return sum(1 for r in results if r)

        except Exception as e:
            print(f"Failed to publish batch: {e}")
            return 0


# ============================================================================
# Kafka Producer (Optional)
# ============================================================================

class KafkaProducer:
    """
    Kafka producer for publishing messages.

    Requires aiokafka or kafka-python.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "adtech.raw.events",
    ):
        """
        Initialize Kafka producer.

        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Topic to publish to
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self._producer = None

    async def connect(self):
        """Establish Kafka connection."""
        # Placeholder for aiokafka implementation
        pass

    async def disconnect(self):
        """Close Kafka connection."""
        pass

    async def publish(self, message: dict) -> bool:
        """Publish a single message."""
        raise NotImplementedError("Kafka producer not fully implemented")

    async def publish_batch(self, messages: List[dict]) -> int:
        """Publish a batch of messages."""
        raise NotImplementedError("Kafka producer not fully implemented")


# ============================================================================
# Message Producer Factory
# ============================================================================

class MessageProducer:
    """
    Factory for creating message producers.

    Supports both Redis and Kafka backends.
    """

    @staticmethod
    def create_redis_producer(
        broker_url: str = "redis://localhost:6379/0",
        queue_name: str = "ingestion:queue",
    ) -> RedisProducer:
        """Create a Redis producer."""
        return RedisProducer(broker_url=broker_url, queue_name=queue_name)

    @staticmethod
    def create_kafka_producer(
        bootstrap_servers: str = "localhost:9092",
        topic: str = "adtech.raw.events",
    ) -> KafkaProducer:
        """Create a Kafka producer."""
        return KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            topic=topic,
        )
