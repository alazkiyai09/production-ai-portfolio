"""
Database repositories for data access.

Provides high-level data access methods.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Select, Update, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.models import EventRecord, EmbeddingCache, ProcessingStatus


# ============================================================================
# Event Repository
# ============================================================================

class EventRepository:
    """Repository for EventRecord operations."""

    def __init__(self, session: AsyncSession):
        """
        Initialize repository.

        Args:
            session: Async database session
        """
        self.session = session

    async def create(
        self,
        event_id: str,
        event_type: str,
        timestamp: datetime,
        campaign_id: str,
        user_id: str,
        content: str,
        metadata: dict,
        batch_id: Optional[str] = None,
    ) -> EventRecord:
        """
        Create a new event record.

        Args:
            event_id: Unique event identifier
            event_type: Type of event
            timestamp: Event timestamp
            campaign_id: Campaign identifier
            user_id: Hashed user identifier
            content: Event content
            metadata: Additional metadata
            batch_id: Optional batch identifier

        Returns:
            Created EventRecord
        """
        record = EventRecord(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            campaign_id=campaign_id,
            user_id=user_id,
            content=content,
            metadata=metadata,
            batch_id=batch_id,
        )

        self.session.add(record)
        await self.session.flush()

        return record

    async def get_by_event_id(self, event_id: str) -> Optional[EventRecord]:
        """
        Get event by event_id.

        Args:
            event_id: Event identifier

        Returns:
            EventRecord or None
        """
        stmt = select(EventRecord).where(EventRecord.event_id == event_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_batch_id(self, batch_id: str) -> List[EventRecord]:
        """
        Get all events for a batch.

        Args:
            batch_id: Batch identifier

        Returns:
            List of EventRecords
        """
        stmt = select(EventRecord).where(EventRecord.batch_id == batch_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_campaign_id(
        self,
        campaign_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[EventRecord]:
        """
        Get events for a campaign.

        Args:
            campaign_id: Campaign identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Max records to return

        Returns:
            List of EventRecords
        """
        stmt = select(EventRecord).where(EventRecord.campaign_id == campaign_id)

        if start_date:
            stmt = stmt.where(EventRecord.timestamp >= start_date)
        if end_date:
            stmt = stmt.where(EventRecord.timestamp <= end_date)

        stmt = stmt.limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_embedding(
        self,
        event_id: str,
        embedding_id: str,
    ) -> Optional[EventRecord]:
        """
        Update embedding information for an event.

        Args:
            event_id: Event identifier
            embedding_id: Vector store embedding ID

        Returns:
            Updated EventRecord or None
        """
        stmt = (
            update(EventRecord)
            .where(EventRecord.event_id == event_id)
            .values(
                embedding_id=embedding_id,
                embedding_generated_at=datetime.utcnow(),
            )
            .returning(EventRecord)
        )

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def mark_processed(self, event_id: str) -> Optional[EventRecord]:
        """
        Mark event as processed.

        Args:
            event_id: Event identifier

        Returns:
            Updated EventRecord or None
        """
        stmt = (
            update(EventRecord)
            .where(EventRecord.event_id == event_id)
            .values(
                is_processed=True,
                processed_at=datetime.utcnow(),
            )
            .returning(EventRecord)
        )

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def exists(self, event_id: str) -> bool:
        """
        Check if event exists.

        Args:
            event_id: Event identifier

        Returns:
            True if exists
        """
        stmt = select(EventRecord.event_id).where(EventRecord.event_id == event_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def count_by_campaign(self, campaign_id: str) -> int:
        """
        Count events for a campaign.

        Args:
            campaign_id: Campaign identifier

        Returns:
            Count of events
        """
        stmt = select(EventRecord).where(EventRecord.campaign_id == campaign_id)
        result = await self.session.execute(stmt)
        return len(list(result.scalars().all()))


# ============================================================================
# Status Repository
# ============================================================================

class StatusRepository:
    """Repository for ProcessingStatus operations."""

    def __init__(self, session: AsyncSession):
        """
        Initialize repository.

        Args:
            session: Async database session
        """
        self.session = session

    async def create(
        self,
        batch_id: str,
        status: str,
        total_records: int,
        client_id: Optional[str] = None,
    ) -> ProcessingStatus:
        """
        Create a new status record.

        Args:
            batch_id: Batch identifier
            status: Initial status
            total_records: Total records in batch
            client_id: Optional client identifier

        Returns:
            Created ProcessingStatus
        """
        record = ProcessingStatus(
            batch_id=batch_id,
            status=status,
            total_records=total_records,
            processed_records=0,
            failed_records=0,
            client_id=client_id,
        )

        self.session.add(record)
        await self.session.flush()

        return record

    async def get_by_batch_id(self, batch_id: str) -> Optional[ProcessingStatus]:
        """
        Get status by batch_id.

        Args:
            batch_id: Batch identifier

        Returns:
            ProcessingStatus or None
        """
        stmt = select(ProcessingStatus).where(ProcessingStatus.batch_id == batch_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(
        self,
        batch_id: str,
        status: str,
        processed_records: Optional[int] = None,
        failed_records: Optional[int] = None,
        error_message: Optional[str] = None,
        error_details: Optional[dict] = None,
    ) -> Optional[ProcessingStatus]:
        """
        Update batch status.

        Args:
            batch_id: Batch identifier
            status: New status
            processed_records: Optional processed count
            failed_records: Optional failed count
            error_message: Optional error message
            error_details: Optional error details dict

        Returns:
            Updated ProcessingStatus or None
        """
        values: dict = {"status": status}

        if status == "processing" and processed_records is None:
            values["started_at"] = datetime.utcnow()

        if status in ["completed", "failed", "partial"]:
            values["completed_at"] = datetime.utcnow()

        if processed_records is not None:
            values["processed_records"] = processed_records

        if failed_records is not None:
            values["failed_records"] = failed_records

        if error_message is not None:
            values["error_message"] = error_message

        if error_details is not None:
            values["error_details"] = error_details

        stmt = (
            update(ProcessingStatus)
            .where(ProcessingStatus.batch_id == batch_id)
            .values(**values)
            .returning(ProcessingStatus)
        )

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def increment_processed(
        self,
        batch_id: str,
        count: int = 1,
    ) -> Optional[ProcessingStatus]:
        """
        Increment processed count.

        Args:
            batch_id: Batch identifier
            count: Number to increment by

        Returns:
            Updated ProcessingStatus or None
        """
        # Get current status
        status = await self.get_by_batch_id(batch_id)
        if not status:
            return None

        return await self.update_status(
            batch_id,
            status.status,
            processed_records=status.processed_records + count,
        )

    async def increment_failed(
        self,
        batch_id: str,
        count: int = 1,
    ) -> Optional[ProcessingStatus]:
        """
        Increment failed count.

        Args:
            batch_id: Batch identifier
            count: Number to increment by

        Returns:
            Updated ProcessingStatus or None
        """
        # Get current status
        status = await self.get_by_batch_id(batch_id)
        if not status:
            return None

        return await self.update_status(
            batch_id,
            status.status,
            failed_records=status.failed_records + count,
        )


# ============================================================================
# Embedding Cache Repository
# ============================================================================

class EmbeddingCacheRepository:
    """Repository for EmbeddingCache operations."""

    def __init__(self, session: AsyncSession):
        """
        Initialize repository.

        Args:
            session: Async database session
        """
        self.session = session

    async def get_by_content_hash(self, content_hash: str) -> Optional[EmbeddingCache]:
        """
        Get cached embedding by content hash.

        Args:
            content_hash: SHA-256 hash of content

        Returns:
            EmbeddingCache or None
        """
        stmt = select(EmbeddingCache).where(EmbeddingCache.content_hash == content_hash)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(
        self,
        content_hash: str,
        content: str,
        embedding_model: str,
        embedding: list,
    ) -> EmbeddingCache:
        """
        Cache an embedding.

        Args:
            content_hash: SHA-256 hash of content
            content: Original content
            embedding_model: Model name used
            embedding: Embedding vector

        Returns:
            Created EmbeddingCache
        """
        cache = EmbeddingCache(
            content_hash=content_hash,
            content=content,
            embedding_model=embedding_model,
            embedding=embedding,
            usage_count=1,
            last_used_at=datetime.utcnow(),
        )

        self.session.add(cache)
        await self.session.flush()

        return cache

    async def update_usage(self, content_hash: str) -> Optional[EmbeddingCache]:
        """
        Update usage count and timestamp.

        Args:
            content_hash: Content hash

        Returns:
            Updated EmbeddingCache or None
        """
        cache = await self.get_by_content_hash(content_hash)
        if not cache:
            return None

        cache.usage_count += 1
        cache.last_used_at = datetime.utcnow()

        await self.session.flush()
        return cache
