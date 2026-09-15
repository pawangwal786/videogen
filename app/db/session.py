"""Database engine and async session management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config.settings import Settings, get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine_kwargs(database_url: str, settings: Settings) -> dict[str, Any]:
    """Construct engine keyword arguments depending on the database dialect."""
    kwargs: dict[str, Any] = {"echo": False}
    if database_url.startswith("sqlite"):
        # SQLite async engine configuration
        kwargs["poolclass"] = NullPool
    else:
        # PostgreSQL with asyncpg connection pooling
        kwargs["pool_size"] = settings.database_pool_size
        kwargs["max_overflow"] = settings.database_max_overflow
        kwargs["pool_timeout"] = settings.database_pool_timeout
        kwargs["pool_recycle"] = settings.database_pool_recycle
        kwargs["pool_pre_ping"] = True
    return kwargs


def init_db(
    database_url: str | None = None,
    settings: Settings | None = None,
) -> async_sessionmaker[AsyncSession]:
    """Initialize or re-initialize the global database engine and session factory."""
    global _engine, _session_factory

    effective_settings = settings or get_settings()
    effective_url = database_url or effective_settings.get_database_url()

    # Dispose previous engine if already initialized
    if _engine is not None:
        # Note: sync dispose initiates clean teardown
        _engine.sync_engine.dispose()

    kwargs = get_engine_kwargs(effective_url, effective_settings)
    _engine = create_async_engine(effective_url, **kwargs)
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    return _session_factory


def get_session_factory(
    settings: Settings | None = None,
) -> async_sessionmaker[AsyncSession]:
    """Return the active session factory, initializing if needed."""
    global _session_factory
    if _session_factory is None:
        return init_db(settings=settings)
    return _session_factory


async def close_db() -> None:
    """Gracefully dispose of the global database engine."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
    _session_factory = None


def reset_db_state() -> None:
    """Reset the module-level engine and session factory references (primarily for testing)."""
    global _engine, _session_factory
    _engine = None
    _session_factory = None


@asynccontextmanager
async def get_db_session(
    factory: async_sessionmaker[AsyncSession] | None = None,
) -> AsyncIterator[AsyncSession]:
    """Provide a transactional AsyncSession scope.
    
    Commits on normal exit, rolls back on exception, and guarantees closure.
    """
    session_maker = factory or get_session_factory()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
