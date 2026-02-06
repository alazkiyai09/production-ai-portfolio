"""
Query result caching for DataChat-RAG.

Provides Redis-backed caching with configurable TTL and statistics tracking.
"""

from .query_cache import QueryCache, CacheEntry, CacheStats

__all__ = [
    "QueryCache",
    "CacheEntry",
    "CacheStats",
]
