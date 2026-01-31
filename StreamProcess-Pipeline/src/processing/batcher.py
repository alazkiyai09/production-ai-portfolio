"""
Batching utilities for StreamProcess-Pipeline.

Provides efficient batching of records for processing.
"""

import asyncio
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Configuration
# ============================================================================

class BatchConfig(BaseModel):
    """Configuration for batcher."""
    batch_size: int = Field(default=100, description="Maximum batch size")
    batch_timeout: float = Field(default=5.0, description="Maximum wait time in seconds")
    max_queue_size: int = Field(default=10000, description="Maximum queue size")
    flush_on_shutdown: bool = Field(default=True, description="Flush pending batches on shutdown")


# ============================================================================
# Batcher
# ============================================================================

class Batcher:
    """
    Accumulates records into batches for efficient processing.

    Features:
    - Time-based batching (flush after timeout)
    - Size-based batching (flush when full)
    - Async batch processing
    - Graceful shutdown
    """

    def __init__(
        self,
        handler: Callable[[List[Dict[str, Any]]], Any],
        config: Optional[BatchConfig] = None,
    ):
        """
        Initialize batcher.

        Args:
            handler: Async callback function to process batches
            config: Optional batch configuration
        """
        self.config = config or BatchConfig()
        self.handler = handler

        # Queue for pending records
        self._queue: Deque[Dict[str, Any]] = deque()
        self._lock = asyncio.Lock()

        # Task control
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_flush = time.time()

        # Metrics
        self._total_records = 0
        self._total_batches = 0
        self._flush_count = 0

    async def add(self, record: Dict[str, Any]) -> None:
        """
        Add a record to the batch queue.

        Args:
            record: Record to add
        """
        async with self._lock:
            if len(self._queue) >= self.config.max_queue_size:
                # Queue full, force flush
                await self._flush()

            self._queue.append(record)
            self._total_records += 1

            # Check if batch is full
            if len(self._queue) >= self.config.batch_size:
                await self._flush()

    async def add_batch(self, records: List[Dict[str, Any]]) -> None:
        """
        Add multiple records to the batch queue.

        Args:
            records: List of records to add
        """
        for record in records:
            await self.add(record)

    async def _flush(self) -> None:
        """
        Flush current batch to handler.

        Returns when handler completes.
        """
        if not self._queue:
            return

        # Extract current batch
        batch = list(self._queue)
        self._queue.clear()

        # Update metrics
        self._total_batches += 1
        self._flush_count += 1
        self._last_flush = time.time()

        # Process batch
        try:
            await self.handler(batch)
        except Exception as e:
            print(f"[Batcher] ERROR: Handler failed for batch of {len(batch)} records: {e}")
            raise

    async def start(self) -> None:
        """Start batcher background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        """Stop batcher and optionally flush remaining records."""
        self._running = False

        if self._task:
            # Wait for task to complete
            try:
                await asyncio.wait_for(self._task, timeout=30.0)
            except asyncio.TimeoutError:
                print("[Batcher] WARNING: Task did not stop in time, cancelling")
                self._task.cancel()
            except Exception as e:
                print(f"[Batcher] ERROR: Task failed: {e}")

        # Flush remaining records
        if self.config.flush_on_shutdown and self._queue:
            await self._flush()

    async def _batch_loop(self) -> None:
        """Background task for time-based flushing."""
        while self._running:
            try:
                await asyncio.sleep(1.0)

                # Check if timeout exceeded
                if time.time() - self._last_flush >= self.config.batch_timeout:
                    async with self._lock:
                        if self._queue:
                            await self._flush()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Batcher] ERROR in batch loop: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get batcher metrics.

        Returns:
            Dictionary with metrics
        """
        return {
            "total_records": self._total_records,
            "total_batches": self._total_batches,
            "queue_size": len(self._queue),
            "flush_count": self._flush_count,
            "avg_batch_size": self._total_records / self._total_batches if self._total_batches > 0 else 0,
        }


# ============================================================================
# Stream Batcher
# ============================================================================

class StreamBatcher:
    """
    Batch streaming data.

    Useful for processing continuous streams of data.
    """

    def __init__(
        self,
        batch_size: int = 100,
        timeout: float = 5.0,
    ):
        """
        Initialize stream batcher.

        Args:
            batch_size: Maximum batch size
            timeout: Maximum wait time before yielding partial batch
        """
        self.batch_size = batch_size
        self.timeout = timeout

    async def batch_stream(
        self,
        stream: Callable[[], AsyncGenerator[Dict[str, Any], None]],
    ) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """
        Batch items from async stream.

        Args:
            stream: Async generator yielding items

        Yields:
            Batches of items
        """
        batch = []
        last_yield = time.time()

        async for item in stream():
            batch.append(item)

            # Check if batch is full
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
                last_yield = time.time()
                continue

            # Check timeout
            if time.time() - last_yield >= self.timeout:
                if batch:
                    yield batch
                    batch = []
                last_yield = time.time()

        # Yield remaining items
        if batch:
            yield batch


# ============================================================================
# Prefetch Batcher
# ============================================================================

class PrefetchBatcher:
    """
    Batcher with prefetching for high-throughput scenarios.

    Maintains a pool of pre-fetched batches to minimize latency.
    """

    def __init__(
        self,
        source: Callable[[], AsyncGenerator[Dict[str, Any], None]],
        batch_size: int = 100,
        prefetch_count: int = 3,
    ):
        """
        Initialize prefetch batcher.

        Args:
            source: Async generator for data source
            batch_size: Batch size
            prefetch_count: Number of batches to prefetch
        """
        self.source = source
        self.batch_size = batch_size
        self.prefetch_count = prefetch_count

        self._queue: Deque[List[Dict[str, Any]]] = deque()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start prefetching."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._prefetch_loop())

    async def stop(self) -> None:
        """Stop prefetching."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def get_batch(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get a pre-fetched batch.

        Returns:
            Batch or None if no batches available
        """
        if self._queue:
            return self._queue.popleft()
        return None

    async def _prefetch_loop(self) -> None:
        """Prefetch batches in background."""
        stream = self.source()
        batch = []

        try:
            async for item in stream:
                batch.append(item)

                if len(batch) >= self.batch_size:
                    # Wait if queue is full
                    while len(self._queue) >= self.prefetch_count:
                        await asyncio.sleep(0.1)

                    self._queue.append(batch)
                    batch = []

        except asyncio.CancelledError:
            pass

        # Add final batch
        if batch:
            self._queue.append(batch)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "BatchConfig",
    "Batcher",
    "StreamBatcher",
    "PrefetchBatcher",
]
