"""FastAPI application factory and lifecycle configuration."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.dependencies import InMemoryRateLimiter
from app.api.errors import register_error_handlers
from app.api.middleware import CorrelationIdMiddleware, RequestSizeLimitMiddleware
from app.api.routers import artifacts_router, health_router, jobs_router, workflows_router
from app.config.settings import Settings, get_settings
from app.db.session import get_session_factory
from app.orchestration.orchestrator import WorkflowOrchestrator


def create_app(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Owns the application-level lifecycle of database connectivity and service boundaries.
    """
    effective_settings = settings or get_settings()
    effective_session_factory = session_factory or get_session_factory()
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
        yield
        # Clean shutdown hooks can be placed here if needed

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

    # Middleware: outermost to innermost order
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware, settings=effective_settings)

    # Register error handlers
    register_error_handlers(app)

    # Register routers
    app.include_router(health_router)
    app.include_router(workflows_router)
    app.include_router(jobs_router)
    app.include_router(artifacts_router)

    return app
