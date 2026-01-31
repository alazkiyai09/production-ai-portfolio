"""
Data processing layer for transformation and enrichment.
"""

from src.processing.worker import (
    celery_app,
    process_batch,
    generate_embeddings,
    update_vector_store,
    update_metadata_db,
    handle_failed_task,
    main as worker_main,
    flower_main,
)
from src.processing.transformer import (
    DataTransformer,
    TransformerConfig,
    TransformPipeline,
    validate_transformed_event,
)
from src.processing.batcher import (
    Batcher,
    BatchConfig,
    StreamBatcher,
    PrefetchBatcher,
)

__all__ = [
    # Worker
    "celery_app",
    "process_batch",
    "generate_embeddings",
    "update_vector_store",
    "update_metadata_db",
    "handle_failed_task",
    "worker_main",
    "flower_main",
    # Transformer
    "DataTransformer",
    "TransformerConfig",
    "TransformPipeline",
    "validate_transformed_event",
    # Batcher
    "Batcher",
    "BatchConfig",
    "StreamBatcher",
    "PrefetchBatcher",
]
