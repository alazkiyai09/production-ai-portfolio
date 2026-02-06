"""
Query result caching with Redis backend.

Provides efficient caching of RAG query results with:
- Redis-based storage with configurable TTL
- Cache key generation from query hash
- Statistics tracking (hit rate, miss rate)
- Graceful error handling with fallback to uncached
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from redis import Redis
from redis.exceptions import RedisError


# =============================================================================
# Logging Configuration
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CacheEntry:
    """A cached query result entry."""

    key: str
    value: Dict[str, Any]
    created_at: str
    ttl: int
    hit_count: int = 0

    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl <= 0:
            return False  # No expiration
        created = datetime.fromisoformat(self.created_at)
        return datetime.now() > created + timedelta(seconds=self.ttl)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "hit_count": self.hit_count,
        }


@dataclass
class CacheStats:
    """Cache statistics."""

    total_hits: int = 0
    total_misses: int = 0
    total_sets: int = 0
    total_deletes: int = 0
    last_hit_at: Optional[str] = None
    last_miss_at: Optional[str] = None

    @property
    def total_requests(self) -> int:
        """Total cache requests (hits + misses)."""
        return self.total_hits + self.total_misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_hits / self.total_requests) * 100

    @property
    def miss_rate(self) -> float:
        """Cache miss rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_misses / self.total_requests) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "total_requests": self.total_requests,
            "total_sets": self.total_sets,
            "total_deletes": self.total_deletes,
            "hit_rate": round(self.hit_rate, 2),
            "miss_rate": round(self.miss_rate, 2),
            "last_hit_at": self.last_hit_at,
            "last_miss_at": self.last_miss_at,
        }


# =============================================================================
# Query Cache
# =============================================================================

class QueryCache:
    """
    Redis-backed cache for RAG query results.

    Features:
    - Automatic cache key generation from query parameters
    - Configurable TTL (default 1 hour)
    - Cache statistics tracking
    - Graceful error handling (fallback to uncached)
    - Thread-safe operations

    Example:
        cache = QueryCache(
            redis_url="redis://localhost:6379/0",
            default_ttl=3600,
            prefix="datachat",
        )

        # Check cache
        result = cache.get(question="What is CTR?", filters={"doc_type": "policy"})

        # Store result
        cache.set(
            question="What is CTR?",
            filters={"doc_type": "policy"},
            response={"answer": "Click-through rate...", "confidence": 0.95},
        )
    """

    # Default cache key prefix
    DEFAULT_PREFIX = "datachat:query"

    # Default TTL in seconds (1 hour)
    DEFAULT_TTL = 3600

    # Cache key format
    KEY_FORMAT = "{prefix}:{hash}"

    def __init__(
        self,
        redis_url: Optional[str] = None,
        redis_host: Optional[str] = None,
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        default_ttl: int = DEFAULT_TTL,
        prefix: str = DEFAULT_PREFIX,
        enabled: bool = True,
        connection_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the query cache.

        Args:
            redis_url: Full Redis URL (e.g., "redis://localhost:6379/0")
                Overrides other connection parameters if provided.
            redis_host: Redis host (default: localhost)
            redis_port: Redis port (default: 6379)
            redis_db: Redis database number (default: 0)
            redis_password: Optional Redis password
            default_ttl: Default TTL for cache entries in seconds (default: 3600)
            prefix: Prefix for cache keys (default: "datachat:query")
            enabled: Whether caching is enabled (default: True)
            connection_kwargs: Additional kwargs for Redis connection
        """
        self.default_ttl = default_ttl
        self.prefix = prefix
        self.enabled = enabled
        self._stats = CacheStats()
        self._redis: Optional[Redis] = None

        if not self.enabled:
            logger.info("Query cache is disabled")
            return

        try:
            # Build Redis connection
            if redis_url:
                self._redis = Redis.from_url(
                    redis_url,
                    **(connection_kwargs or {}),
                )
            else:
                redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
                redis_port = int(os.getenv("REDIS_PORT", str(redis_port)))
                redis_db = int(os.getenv("REDIS_DB", str(redis_db)))
                redis_password = redis_password or os.getenv("REDIS_PASSWORD")

                self._redis = Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password,
                    decode_responses=True,
                    **(connection_kwargs or {}),
                )

            # Test connection
            self._redis.ping()
            logger.info(f"Query cache connected: {redis_host}:{redis_port}/{redis_db}")

        except RedisError as e:
            logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
            self._redis = None
            self.enabled = False

    # =========================================================================
    # Cache Key Generation
    # =========================================================================

    def generate_key(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a cache key from query parameters.

        The key is a hash of the question, filters, and recent conversation
        context to ensure cache hits only for semantically equivalent queries.

        Args:
            question: User's question
            filters: Optional filters for document retrieval
            conversation_context: Optional recent conversation messages

        Returns:
            Cache key string
        """
        # Build key components
        key_data = {
            "question": question.strip().lower(),
            "filters": self._normalize_filters(filters),
        }

        # Include last 2 conversation messages for context awareness
        if conversation_context:
            recent_context = conversation_context[-2:] if len(conversation_context) > 2 else conversation_context
            key_data["context"] = [
                {"role": msg["role"], "content": msg["content"][:100]}
                for msg in recent_context
            ]

        # Create hash
        key_json = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_json.encode()).hexdigest()[:16]

        return self.KEY_FORMAT.format(prefix=self.prefix, hash=key_hash)

    def _normalize_filters(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize filters for consistent cache keys."""
        if not filters:
            return {}

        normalized = {}
        for key, value in sorted(filters.items()):
            if isinstance(value, list):
                normalized[key] = sorted(value)
            elif isinstance(value, (str, int, float, bool)):
                normalized[key] = value
            else:
                normalized[key] = str(value)

        return normalized

    # =========================================================================
    # Cache Operations
    # =========================================================================

    def get(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a query.

        Args:
            question: User's question
            filters: Optional filters used in the query
            conversation_context: Optional conversation messages

        Returns:
            Cached response dict, or None if not found/expired/error
        """
        if not self.enabled or not self._redis:
            self._record_miss()
            return None

        try:
            key = self.generate_key(question, filters, conversation_context)
            cached = self._redis.get(key)

            if cached:
                # Increment hit counter in Redis
                self._redis.hincrby(f"{key}:meta", "hit_count", 1)

                # Update stats
                self._record_hit()

                logger.debug(f"Cache hit: {key}")
                return json.loads(cached)

            self._record_miss()
            logger.debug(f"Cache miss: {key}")
            return None

        except RedisError as e:
            logger.warning(f"Cache get error: {e}")
            self._record_miss()
            return None

    def set(
        self,
        question: str,
        response: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store query result in cache.

        Args:
            question: User's question
            response: Response dict to cache
            filters: Optional filters used in the query
            conversation_context: Optional conversation messages
            ttl: Time-to-live in seconds (default: default_ttl)

        Returns:
            True if cached successfully, False otherwise
        """
        if not self.enabled or not self._redis:
            return False

        try:
            key = self.generate_key(question, filters, conversation_context)
            ttl = ttl if ttl is not None else self.default_ttl

            # Store with expiration
            cached = json.dumps(response)
            if ttl > 0:
                self._redis.setex(key, ttl, cached)
            else:
                self._redis.set(key, cached)

            # Store metadata
            meta = {
                "created_at": datetime.now().isoformat(),
                "hit_count": 0,
                "ttl": ttl,
            }
            self._redis.hset(f"{key}:meta", mapping=meta)
            if ttl > 0:
                self._redis.expire(f"{key}:meta", ttl)

            # Update stats
            self._stats.total_sets += 1

            logger.debug(f"Cached: {key} (TTL: {ttl}s)")
            return True

        except RedisError as e:
            logger.warning(f"Cache set error: {e}")
            return False

    def delete(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> bool:
        """
        Delete cached result for a query.

        Args:
            question: User's question
            filters: Optional filters used in the query
            conversation_context: Optional conversation messages

        Returns:
            True if deleted, False otherwise
        """
        if not self.enabled or not self._redis:
            return False

        try:
            key = self.generate_key(question, filters, conversation_context)
            deleted = self._redis.delete(key, f"{key}:meta")

            if deleted:
                self._stats.total_deletes += 1
                logger.debug(f"Deleted: {key}")
                return True

            return False

        except RedisError as e:
            logger.warning(f"Cache delete error: {e}")
            return False

    def clear(self) -> bool:
        """
        Clear all cached queries with the configured prefix.

        Returns:
            True if cleared successfully, False otherwise
        """
        if not self.enabled or not self._redis:
            return False

        try:
            # Find all keys with prefix
            pattern = f"{self.prefix}:*"
            keys = self._redis.keys(pattern)

            if keys:
                deleted = self._redis.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries")
                return True

            return True  # No keys to delete

        except RedisError as e:
            logger.warning(f"Cache clear error: {e}")
            return False

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        stats = self._stats.to_dict()

        # Add Redis info if available
        if self._redis:
            try:
                info = self._redis.info("stats")
                stats["redis_total_keys"] = self._redis.dbsize()
                stats["redis_memory_used"] = info.get("used_memory_human", "unknown")
            except RedisError:
                pass

        return stats

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = CacheStats()

    def _record_hit(self) -> None:
        """Record a cache hit."""
        self._stats.total_hits += 1
        self._stats.last_hit_at = datetime.now().isoformat()

    def _record_miss(self) -> None:
        """Record a cache miss."""
        self._stats.total_misses += 1
        self._stats.last_miss_at = datetime.now().isoformat()

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_or_set(
        self,
        question: str,
        callback,
        filters: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        ttl: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get cached result or compute and cache it.

        Args:
            question: User's question
            callback: Function to compute result if not cached
            filters: Optional filters for the query
            conversation_context: Optional conversation messages
            ttl: TTL for cached result

        Returns:
            Response dict from cache or callback
        """
        # Try cache first
        cached = self.get(question, filters, conversation_context)
        if cached is not None:
            return cached

        # Compute and cache
        result = callback()
        self.set(question, result, filters, conversation_context, ttl)
        return result

    def is_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.enabled and self._redis is not None

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            try:
                self._redis.close()
                logger.info("Cache connection closed")
            except RedisError as e:
                logger.warning(f"Error closing cache connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_cache(
    redis_url: Optional[str] = None,
    default_ttl: int = QueryCache.DEFAULT_TTL,
    enabled: bool = True,
) -> QueryCache:
    """
    Convenience function to create a QueryCache instance.

    Args:
        redis_url: Redis connection URL
        default_ttl: Default TTL in seconds
        enabled: Whether caching is enabled

    Returns:
        Configured QueryCache instance
    """
    return QueryCache(
        redis_url=redis_url,
        default_ttl=default_ttl,
        enabled=enabled,
    )


# =============================================================================
# Test Cases
# =============================================================================

def run_test_cases():
    """Run test cases for the query cache."""
    import pprint

    pp = pprint.PrettyPrinter(indent=2)

    print("=" * 80)
    print("QUERY CACHE TEST CASES")
    print("=" * 80)

    # Create cache (disabled if no Redis)
    cache = QueryCache(
        redis_host="localhost",
        enabled=True,
    )

    if not cache.is_enabled():
        print("\nRedis not available, using disabled cache for testing")
        print("Cache operations will return None but won't raise errors\n")
    else:
        print("\nCache enabled and connected to Redis\n")

    # Test 1: Key Generation
    print("─" * 80)
    print("Test 1: Cache Key Generation")
    print("─" * 80)

    question = "What is our average CTR?"
    filters = {"doc_type": "policy", "date_range": "last_week"}

    key1 = cache.generate_key(question, filters)
    key2 = cache.generate_key(question, filters)
    key3 = cache.generate_key(question, {"doc_type": "policy"})  # Different filters

    print(f"Question: {question}")
    print(f"Filters: {filters}")
    print(f"\nGenerated keys:")
    print(f"  Key 1: {key1}")
    print(f"  Key 2: {key2}")
    print(f"  Key 3: {key3}")
    print(f"\nKey 1 == Key 2: {key1 == key2}")
    print(f"Key 1 == Key 3: {key1 == key3}")

    # Test 2: Cache Operations
    print("\n─" * 80)
    print("Test 2: Cache Operations")
    print("─" * 80)

    test_response = {
        "answer": "The average CTR is 1.2%.",
        "query_type": "SQL_QUERY",
        "confidence": 0.95,
        "doc_sources": [],
    }

    # Set
    print(f"\nSetting cache for question: {question}")
    success = cache.set(question, test_response, filters, ttl=60)
    print(f"Set success: {success}")

    # Get
    print(f"\nGetting cache for question: {question}")
    cached = cache.get(question, filters)
    print(f"Cached result: {cached}")

    # Delete
    print(f"\nDeleting cache for question: {question}")
    deleted = cache.delete(question, filters)
    print(f"Delete success: {deleted}")

    # Get after delete
    print(f"\nGetting cache after delete:")
    cached = cache.get(question, filters)
    print(f"Cached result: {cached}")

    # Test 3: Statistics
    print("\n─" * 80)
    print("Test 3: Cache Statistics")
    print("─" * 80)

    # Generate some activity
    for _ in range(3):
        cache.get("test question")
    cache.set("test question", {"answer": "test"})
    cache.get("test question")

    stats = cache.get_stats()
    print("\nCache Statistics:")
    pp.pprint(stats)

    # Test 4: Clear Cache
    print("\n─" * 80)
    print("Test 4: Clear Cache")
    print("─" * 80)

    cache.set("q1", {"answer": "1"})
    cache.set("q2", {"answer": "2"})
    cache.set("q3", {"answer": "3"})

    cleared = cache.clear()
    print(f"\nCache cleared: {cleared}")

    # Clean up
    cache.close()

    print("\n" + "=" * 80)
    print("TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_test_cases()
