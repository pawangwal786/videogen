"""FastAPI application factory and lifecycle configuration."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.api.dependencies import InMemoryRateLimiter
from app.api.errors import register_error_handlers
from app.api.middleware import CorrelationIdMiddleware, RequestSizeLimitMiddleware
from app.api.routers import artifacts_router, health_router, jobs_router, workflows_router
from app.config.settings import Settings, get_settings
from app.db.session import get_engine_kwargs
from app.orchestration.orchestrator import WorkflowOrchestrator


def create_app(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Owns the application-level lifecycle of database connectivity and service boundaries.
    If session_factory is provided externally, the application does not own or dispose it.
    If session_factory is omitted, the application creates and owns its database engine,
    disposing it cleanly during lifespan shutdown.
    """
    effective_settings = settings or get_settings()

    owned_engine: AsyncEngine | None = None
    if session_factory is None:
        owns_db = True
        db_url = effective_settings.get_database_url()
        kwargs = get_engine_kwargs(db_url, effective_settings)
        owned_engine = create_async_engine(db_url, **kwargs)
        effective_session_factory = async_sessionmaker(
            bind=owned_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    else:
        owns_db = False
        effective_session_factory = session_factory

    orchestrator = WorkflowOrchestrator(session_factory=effective_session_factory)
    rate_limiter = InMemoryRateLimiter(
        requests_per_minute=effective_settings.api_rate_limit_per_minute
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        # Initialize and attach state on startup
        app.state.settings = effective_settings
        app.state.session_factory = effective_session_factory
        app.state.orchestrator = orchestrator
        app.state.rate_limiter = rate_limiter
        app.state.owns_db = owns_db
        app.state.db_engine = owned_engine
        yield
        # Clean shutdown hook: dispose owned database engine
        if app.state.owns_db and app.state.db_engine is not None:
            await app.state.db_engine.dispose()

    app = FastAPI(
        title=effective_settings.api_title,
        version=effective_settings.api_version,
        description=(
            "VideoGen Control Plane API: Durable workflow creation, monitoring, "
            "fenced cancellation, and artifact management."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Attach state immediately as well so dependencies work in test setups before lifespan
    app.state.settings = effective_settings
    app.state.session_factory = effective_session_factory
    app.state.orchestrator = orchestrator
    app.state.rate_limiter = rate_limiter
    app.state.owns_db = owns_db
    app.state.db_engine = owned_engine

    # Middleware: outermost to innermost order (last added runs first in Starlette)
    app.add_middleware(RequestSizeLimitMiddleware, settings=effective_settings)
    app.add_middleware(CorrelationIdMiddleware)

    # Register error handlers
    register_error_handlers(app)

    # Register routers
    app.include_router(health_router)
    app.include_router(workflows_router)
    app.include_router(jobs_router)
    app.include_router(artifacts_router)

    return app
