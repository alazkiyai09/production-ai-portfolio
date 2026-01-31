"""
Database models for StreamProcess-Pipeline.

SQLAlchemy ORM models for event records and processing status.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Integer, String, Text, JSON, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ============================================================================
# Base
# ============================================================================

class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# ============================================================================
# Event Record Model
# ============================================================================

class EventRecord(Base):
    """
    Stored event record.

    Represents a single AdTech event that has been processed.
    """
    __tablename__ = "event_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    campaign_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Embedding-related fields
    embedding_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    embedding_generated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Processing metadata
    batch_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Status
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)

    # Audit fields
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Indexes
    __table_args__ = (
        Index('ix_event_records_timestamp_campaign', 'timestamp', 'campaign_id'),
        Index('ix_event_records_event_type_timestamp', 'event_type', 'timestamp'),
        Index('ix_event_records_batch_id', 'batch_id'),
    )

    def __repr__(self) -> str:
        return f"<EventRecord(event_id='{self.event_id}', event_type='{self.event_type}', campaign_id='{self.campaign_id}')>"


# ============================================================================
# Processing Status Model
# ============================================================================

class ProcessingStatus(Base):
    """
    Batch processing status tracker.

    Tracks the status of batch processing jobs.
    """
    __tablename__ = "processing_status"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Counts
    total_records: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_records: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_records: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Client info
    client_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Processing metrics
    processing_duration_seconds: Mapped[Optional[float]] = mapped_column(Integer, nullable=True)

    # Audit
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<ProcessingStatus(batch_id='{self.batch_id}', status='{self.status}', progress={self.processed_records}/{self.total_records})>"


# ============================================================================
# Embedding Cache Model
# ============================================================================

class EmbeddingCache(Base):
    """
    Embedding cache for frequently seen content.

    Stores pre-computed embeddings to avoid recomputation.
    """
    __tablename__ = "embedding_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)

    # Embedding stored as JSON array
    embedding: Mapped[list] = mapped_column(JSON, nullable=False)

    # Usage tracking
    usage_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<EmbeddingCache(content_hash='{self.content_hash}', model='{self.embedding_model}', used={self.usage_count} times)>"
