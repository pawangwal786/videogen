"""API routers export."""

from app.api.routers.artifacts import router as artifacts_router
from app.api.routers.health import router as health_router
from app.api.routers.jobs import router as jobs_router
from app.api.routers.workflows import router as workflows_router

__all__ = [
    "artifacts_router",
    "health_router",
    "jobs_router",
    "workflows_router",
]
