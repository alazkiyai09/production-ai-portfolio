"""
Embedding generation layer for vector creation.
"""

from src.embedding.generator import (
    EmbeddingGenerator,
    EmbeddingConfig,
    ModelRegistry,
    quick_embedding,
    batch_embeddings,
)
from src.embedding.cache import (
    CacheBackend,
    RedisCacheBackend,
    DatabaseCacheBackend,
    HybridEmbeddingCache,
)
from src.embedding.embed_service import (
    EmbeddingService,
    EmbeddingServiceConfig,
    EmbeddingBenchmark,
    AVAILABLE_MODELS,
    create_embedding_service,
    quick_embed,
)

# Future imports
# from src.embedding.models import ModelRegistry

__all__ = [
    # Service
    "EmbeddingService",
    "EmbeddingServiceConfig",
    "EmbeddingBenchmark",
    "AVAILABLE_MODELS",
    "create_embedding_service",
    "quick_embed",
    # Generator
    "EmbeddingGenerator",
    "EmbeddingConfig",
    "ModelRegistry",
    "quick_embedding",
    "batch_embeddings",
    # Cache
    "CacheBackend",
    "RedisCacheBackend",
    "DatabaseCacheBackend",
    "HybridEmbeddingCache",
]
