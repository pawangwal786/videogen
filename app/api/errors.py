"""Standardized error handlers and exception mappings for FastAPI application."""

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


async def workflow_not_found_handler(request: Request, exc: WorkflowNotFoundError) -> JSONResponse:
    """Handle missing workflow lookups with 404."""
    error = ErrorResponse(
        error=ErrorBody(
            code="WORKFLOW_NOT_FOUND",
            message=str(exc),
        )
    )
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=error.model_dump())


async def idempotency_conflict_handler(
    request: Request, exc: IdempotencyConflictError
) -> JSONResponse:
    """Handle idempotency key collisions with 409 Conflict."""
    error = ErrorResponse(
        error=ErrorBody(
            code="IDEMPOTENCY_CONFLICT",
            message=str(exc),
        )
    )
    return JSONResponse(status_code=status.HTTP_409_CONFLICT, content=error.model_dump())


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request schema validation errors with structured 422."""
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

    error = ErrorResponse(
        error=ErrorBody(
            code="VALIDATION_ERROR",
            message="Request validation failed.",
            details=details,
        )
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=error.model_dump(),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle explicit HTTP exceptions with standardized envelope."""
    headers = exc.headers or {}
    code = _status_code_to_error_code(exc.status_code)
    error = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=str(exc.detail),
        )
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error.model_dump(),
        headers=headers,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected internal exceptions, logging securely without leaking secrets."""
    logger.error(
        "unhandled_server_error",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=exc,
    )
    error = ErrorResponse(
        error=ErrorBody(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected internal server error occurred.",
        )
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error.model_dump(),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers on the FastAPI application."""
    app.add_exception_handler(WorkflowNotFoundError, workflow_not_found_handler)  # type: ignore[arg-type]
    app.add_exception_handler(IdempotencyConflictError, idempotency_conflict_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)  # type: ignore[arg-type]
