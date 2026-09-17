"""Health and readiness probe router."""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.dependencies import get_session_factory_dep, get_settings_dep
from app.api.schemas.common import HealthResponse, ReadyResponse
from app.config.settings import Settings
from app.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns HTTP 200 if the HTTP process is running. Never checks downstream databases.",
)
async def health_check(
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> HealthResponse:
    return HealthResponse(status="ok", version=settings.api_version)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={
        200: {"description": "Service is ready and connected to database."},
        503: {"description": "Service is not ready; database connection failed."},
    },
    summary="Readiness probe",
    description="Actively checks PostgreSQL database connectivity by executing SELECT 1.",
)
async def readiness_check(
    session_factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory_dep)],
) -> JSONResponse:
    try:
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ReadyResponse(status="ready", database="connected").model_dump(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("readiness_check_failed", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ReadyResponse(
                status="unavailable",
                database="disconnected",
                error="Database connectivity check failed.",
            ).model_dump(),
        )
