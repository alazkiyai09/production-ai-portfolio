"""
Embedding cache for StreamProcess-Pipeline.

Provides caching layer to avoid recomputing embeddings.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.models import EmbeddingCache
from src.storage.repositories import EmbeddingCacheRepository


# ============================================================================
# Cache Backend
# ============================================================================

class CacheBackend:
    """Base cache backend."""

    async def get(self, key: str) -> Optional[List[float]]:
        """Get value from cache."""
        raise NotImplementedError

    async def set(self, key: str, value: List[float], ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        raise NotImplementedError

    async def clear(self) -> None:
        """Clear all cache."""
        raise NotImplementedError


# ============================================================================
# Redis Cache Backend
# ============================================================================

class RedisCacheBackend(CacheBackend):
    """Redis-based cache backend."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "embed:",
        default_ttl: int = 86400,  # 24 hours
    ):
        """
        Initialize Redis cache backend.

        Args:
            redis_url: Redis connection URL
            prefix: Key prefix
            default_ttl: Default TTL in seconds
        """
        self.redis_url = redis_url
        self.prefix = prefix
        self.default_ttl = default_ttl
        self._redis = None

    async def _get_redis(self):
        """Get Redis client."""
        if self._redis is None:
            import redis.asyncio as redis
            self._redis = await redis.from_url(self.redis_url)
        return self._redis

    async def get(self, key: str) -> Optional[List[float]]:
        """Get value from Redis."""
        client = await self._get_redis()
        value = await client.get(f"{self.prefix}{key}")

        if value:
            return json.loads(value)
        return None

    async def set(self, key: str, value: List[float], ttl: Optional[int] = None) -> None:
        """Set value in Redis."""
        client = await self._get_redis()
        serialized = json.dumps(value)

        if ttl or self.default_ttl:
            await client.setex(f"{self.prefix}{key}", ttl or self.default_ttl, serialized)
        else:
            await client.set(f"{self.prefix}{key}", serialized)

    async def delete(self, key: str) -> None:
        """Delete value from Redis."""
        client = await self._get_redis()
        await client.delete(f"{self.prefix}{key}")

    async def clear(self) -> None:
        """Clear all cache with prefix."""
        import redis.asyncio as redis

        client = await self._get_redis()
        keys = await client.keys(f"{self.prefix}*")

        if keys:
            await client.delete(*keys)


# ============================================================================
# Database Cache Backend
# ============================================================================

class DatabaseCacheBackend(CacheBackend):
    """PostgreSQL-based cache backend."""

    def __init__(self, session_factory):
        """
        Initialize database cache backend.

        Args:
            session_factory: Async session factory
        """
        self.session_factory = session_factory

    async def get(self, key: str) -> Optional[List[float]]:
        """Get value from database."""
        async with self.session_factory() as session:
            repo = EmbeddingCacheRepository(session)
            record = await repo.get_by_content_hash(key)

            if record:
                # Update usage
                await repo.update_usage(key)
                return record.embedding

            return None

    async def set(self, key: str, value: List[float], ttl: Optional[int] = None) -> None:
        """Set value in database."""
        async with self.session_factory() as session:
            repo = EmbeddingCacheRepository(session)

            # Check if already exists
            existing = await repo.get_by_content_hash(key)

            if not existing:
                # For DB cache, we need the original content
                # This is a limitation - consider using Redis for full caching
                pass

    async def delete(self, key: str) -> None:
        """Delete value from database."""
        async with self.session_factory() as session:
            # Delete from EmbeddingCache table
            from sqlalchemy import delete
            await session.execute(delete(EmbeddingCache).where(EmbeddingCache.content_hash == key))
            await session.commit()

    async def clear(self) -> None:
        """Clear all cache."""
        async with self.session_factory() as session:
            from sqlalchemy import delete
            await session.execute(delete(EmbeddingCache))
            await session.commit()


# ============================================================================
# Hybrid Cache
# ============================================================================

class HybridEmbeddingCache:
    """
    Hybrid cache with L1 (memory) and L2 (Redis/DB) tiers.

    L1: In-memory cache (fastest)
    L2: Redis or database cache (persistent)
    """

    def __init__(
        self,
        l1_max_size: int = 1000,
        l2_backend: Optional[CacheBackend] = None,
    ):
        """
        Initialize hybrid cache.

        Args:
            l1_max_size: Max L1 cache size
            l2_backend: Optional L2 cache backend
        """
        from src.embedding.generator import EmbeddingCache

        self.l1 = EmbeddingCache(max_size=l1_max_size)
        self.l2 = l2_backend

    async def get(self, content: str) -> Optional[List[float]]:
        """
        Get embedding from cache.

        Args:
            content: Text content

        Returns:
            Cached embedding or None
        """
        # Try L1 first
        cached = self.l1.get(content)
        if cached is not None:
            return cached

        # Try L2
        if self.l2:
            key = self._hash(content)
            cached = await self.l2.get(key)

            if cached is not None:
                # Promote to L1
                self.l1.set(content, cached)
                return cached

        return None

    async def set(self, content: str, embedding: List[float]) -> None:
        """
        Cache embedding.

        Args:
            content: Text content
            embedding: Embedding vector
        """
        # Store in L1
        self.l1.set(content, embedding)

        # Store in L2
        if self.l2:
            key = self._hash(content)
            await self.l2.set(key, embedding)

    async def delete(self, content: str) -> None:
        """
        Delete from cache.

        Args:
            content: Text content
        """
        # Delete from L1
        key = self._hash(content)
        if self.l1._cache.get(key):
            del self.l1._cache[key]
            del self.l1._access_count[key]

        # Delete from L2
        if self.l2:
            await self.l2.delete(key)

    async def clear(self) -> None:
        """Clear all caches."""
        self.l1.clear()
        if self.l2:
            await self.l2.clear()

    def _hash(self, content: str) -> str:
        """Generate hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "CacheBackend",
    "RedisCacheBackend",
    "DatabaseCacheBackend",
    "HybridEmbeddingCache",
]
