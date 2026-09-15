"""Integration test configuration and PostgreSQL contract enforcement."""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import get_settings
from app.db.base import Base


def require_postgres() -> str:
    """Validate PostgreSQL contract for integration tests.

    Obtains VIDEOGEN_TEST_DATABASE_URL with fallback to VIDEOGEN_DATABASE_URL.
    Fails clearly if PostgreSQL URL is not configured or uses SQLite.
    """
    settings = get_settings()
    url = settings.get_database_url(for_test=True)
    if not url or not ("postgresql" in url or "postgres" in url):
        pytest.fail(
            f"PostgreSQL 16 is mandatory for integration tests; received non-postgres URL: '{url}'. "
            "No SQLite fallback is allowed for integration tests."
        )
    return url


@pytest.fixture(scope="session")
def postgres_url() -> str:
    """Return the verified PostgreSQL connection URL."""
    return require_postgres()


from sqlalchemy.pool import NullPool


@pytest.fixture
async def pg_engine(postgres_url: str):
    """Function-scoped async engine using NullPool for clean event-loop isolation."""
    engine = create_async_engine(
        postgres_url,
        poolclass=NullPool,
    )
    # Verify reachability
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.fail(
            f"PostgreSQL connection to {postgres_url} failed: {exc}. "
            "Integration tests require a reachable PostgreSQL 16 instance."
        )

    # Ensure schema exists
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(pg_engine: AsyncEngine):
    """Provide a clean, isolated database session for each integration test."""
    session_factory = async_sessionmaker(
        bind=pg_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    # Clean tables before test
    async with pg_engine.begin() as conn:
        await conn.execute(text("TRUNCATE TABLE artifacts, job_attempts, jobs, workflows RESTART IDENTITY CASCADE"))

    async with session_factory() as session:
        yield session
        await session.rollback()
