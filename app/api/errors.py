"""Standardized error handlers and exception mappings for FastAPI application."""

from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.schemas.common import ErrorBody, ErrorDetail, ErrorResponse
from app.logging import get_logger
from app.orchestration.errors import IdempotencyConflictError, WorkflowNotFoundError

logger = get_logger(__name__)


def _status_code_to_error_code(status_code: int) -> str:
    """Map common HTTP status codes to standardized error code strings."""
    mapping = {
        status.HTTP_400_BAD_REQUEST: "BAD_REQUEST",
        status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
        status.HTTP_403_FORBIDDEN: "FORBIDDEN",
        status.HTTP_404_NOT_FOUND: "NOT_FOUND",
        status.HTTP_409_CONFLICT: "CONFLICT",
        status.HTTP_413_CONTENT_TOO_LARGE: "PAYLOAD_TOO_LARGE",
        status.HTTP_422_UNPROCESSABLE_CONTENT: "VALIDATION_ERROR",
        status.HTTP_429_TOO_MANY_REQUESTS: "RATE_LIMIT_EXCEEDED",
        status.HTTP_500_INTERNAL_SERVER_ERROR: "INTERNAL_SERVER_ERROR",
        status.HTTP_503_SERVICE_UNAVAILABLE: "SERVICE_UNAVAILABLE",
    }
    return mapping.get(status_code, "HTTP_ERROR")


def build_error_response(
    status_code: int,
    code: str,
    message: str,
    details: list[ErrorDetail] | dict[str, Any] | None = None,
    correlation_id: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Construct a standardized JSONResponse with ErrorResponse envelope."""
    response_headers = dict(headers or {})
    if correlation_id and "X-Correlation-ID" not in response_headers:
        response_headers["X-Correlation-ID"] = correlation_id

    error = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            details=details,
            correlation_id=correlation_id,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=error.model_dump(),
        headers=response_headers,
    )


def _extract_correlation_id(request: Request) -> str | None:
    """Extract correlation ID from request state or fallback header."""
    return getattr(request.state, "correlation_id", None) or request.headers.get("X-Correlation-ID")


async def workflow_not_found_handler(request: Request, exc: WorkflowNotFoundError) -> JSONResponse:
    """Handle missing workflow lookups with 404."""
    corr_id = _extract_correlation_id(request)
    return build_error_response(
        status_code=status.HTTP_404_NOT_FOUND,
        code="WORKFLOW_NOT_FOUND",
        message=str(exc),
        correlation_id=corr_id,
    )


async def idempotency_conflict_handler(
    request: Request, exc: IdempotencyConflictError
) -> JSONResponse:
    """Handle idempotency key collisions with 409 Conflict."""
    corr_id = _extract_correlation_id(request)
    return build_error_response(
        status_code=status.HTTP_409_CONFLICT,
        code="IDEMPOTENCY_CONFLICT",
        message=str(exc),
        correlation_id=corr_id,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request schema validation errors with structured 422."""
    corr_id = _extract_correlation_id(request)
    details: list[ErrorDetail] = []
    for err in exc.errors():
        loc = list(err.get("loc", []))
        # Strip internal 'body' prefix if present for cleaner API documentation
        cleaned_loc: list[str | int] = [elem for elem in loc if elem not in ("body",)]
        details.append(
            ErrorDetail(
                location=cleaned_loc or loc,
                message=err.get("msg", "Invalid input"),
                type=err.get("type", "value_error"),
            )
        )

    return build_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        code="VALIDATION_ERROR",
        message="Request validation failed.",
        details=details,
        correlation_id=corr_id,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle explicit HTTP exceptions with standardized envelope."""
    corr_id = _extract_correlation_id(request)
    code = _status_code_to_error_code(exc.status_code)
    return build_error_response(
        status_code=exc.status_code,
        code=code,
        message=str(exc.detail),
        correlation_id=corr_id,
        headers=exc.headers,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected internal exceptions, logging securely without leaking secrets."""
    corr_id = _extract_correlation_id(request)
    logger.error(
        "unhandled_server_error",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=exc,
    )
    return build_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected internal server error occurred.",
        correlation_id=corr_id,
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers on the FastAPI application."""
    app.add_exception_handler(WorkflowNotFoundError, workflow_not_found_handler)  # type: ignore[arg-type]
    app.add_exception_handler(IdempotencyConflictError, idempotency_conflict_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)  # type: ignore[arg-type]
