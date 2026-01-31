"""
Database connection and session management.

Provides async database connections and session management.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool


# ============================================================================
# Database Manager
# ============================================================================

class DatabaseManager:
    """
    Database connection manager.

    Manages async engine and session factory.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
    ):
        """
        Initialize database manager.

        Args:
            database_url: Database connection URL
            pool_size: Connection pool size
            max_overflow: Max overflow connections
            pool_timeout: Connection timeout
            pool_recycle: Connection recycle time
            echo: Echo SQL statements
        """
        if database_url is None:
            database_url = os.getenv(
                "POSTGRES_URL",
                os.getenv(
                    "TEST_DATABASE_URL",
                    "postgresql+asyncpg://streamprocess_user:streamprocess_pass@localhost:5432/streamprocess"
                )
            )

        self.database_url = database_url
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

        # Engine configuration
        self.engine_kwargs = {
            "echo": echo,
            "pool_size": pool_size,
            "max_overflow": max_overflow,
            "pool_timeout": pool_timeout,
            "pool_recycle": pool_recycle,
            "pool_pre_ping": True,
        }

    async def connect(self) -> None:
        """Establish database connection."""
        if self._engine is None:
            self._engine = create_async_engine(
                self.database_url,
                **self.engine_kwargs,
            )

            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    @property
    def engine(self) -> AsyncEngine:
        """Get database engine."""
        if self._engine is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get session factory."""
        if self._session_factory is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._session_factory

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session context manager.

        Yields:
            AsyncSession
        """
        if self._session_factory is None:
            await self.connect()

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def create_tables(self) -> None:
        """Create all tables."""
        from src.storage.models import Base

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """Drop all tables (use with caution)."""
        from src.storage.models import Base

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


# ============================================================================
# Global Database Manager
# ============================================================================

_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get global database manager instance.

    Returns:
        DatabaseManager
    """
    global _db_manager

    if _db_manager is None:
        _db_manager = DatabaseManager()

    return _db_manager


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI to get database session.

    Yields:
        AsyncSession
    """
    db_manager = get_db_manager()

    if db_manager._engine is None:
        await db_manager.connect()

    async with db_manager.session() as session:
        yield session


# ============================================================================
# Test Database Helper
# ============================================================================

async def create_test_db() -> DatabaseManager:
    """
    Create a test database manager.

    Returns:
        DatabaseManager configured for testing
    """
    test_url = os.getenv(
        "TEST_DATABASE_URL",
        "sqlite+aiosqlite:///:memory:"
    )

    db = DatabaseManager(database_url=test_url)
    await db.connect()
    await db.create_tables()

    return db


async def drop_test_db(db: DatabaseManager) -> None:
    """
    Drop test database tables.

    Args:
        db: DatabaseManager instance
    """
    await db.drop_tables()
    await db.disconnect()
