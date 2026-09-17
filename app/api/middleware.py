"""API middleware for request correlation, payload size limiting, and security."""

import uuid
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.schemas.common import ErrorBody, ErrorResponse
from app.config.settings import Settings, get_settings
from app.logging import bind_correlation, clear_correlation, get_logger

logger = get_logger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware extracting or generating X-Correlation-ID and propagating it to logs and response."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        corr_id = request.headers.get("X-Correlation-ID") or request.headers.get("X-Request-ID")
        if not corr_id or len(corr_id.strip()) == 0 or len(corr_id) > 128:
            corr_id = str(uuid.uuid4())
        else:
            corr_id = corr_id.strip()

        # Attach to request state for handler access
        request.state.correlation_id = corr_id

        # Bind to structlog contextvars for automatic inclusion in all application logs
        bind_correlation(correlation_id=corr_id)

        try:
            response: Response = await call_next(request)
            response.headers["X-Correlation-ID"] = corr_id
            return response
        finally:
            clear_correlation()


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing maximum request body size before buffering."""

    def __init__(self, app: Any, settings: Settings | None = None) -> None:
        super().__init__(app)
        self._settings = settings or get_settings()

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                length = int(content_length)
                if length > self._settings.api_max_request_body_bytes:
                    err = ErrorResponse(
                        error=ErrorBody(
                            code="PAYLOAD_TOO_LARGE",
                            message=(
                                f"Request payload size {length} bytes exceeds limit of "
                                f"{self._settings.api_max_request_body_bytes} bytes."
                            ),
                        )
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content=err.model_dump(),
                    )
            except ValueError:
                pass

        return await call_next(request)
