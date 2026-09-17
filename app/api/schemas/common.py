"""Common API schemas for standardized errors and health checks."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """Field-level error detail."""

    model_config = ConfigDict(extra="forbid")

    location: list[str | int] = Field(default_factory=list)
    message: str
    type: str = "value_error"


class ErrorBody(BaseModel):
    """Standardized error payload."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: list[ErrorDetail] | dict[str, Any] | None = None
    correlation_id: str | None = None


class ErrorResponse(BaseModel):
    """Top-level error response envelope."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorBody


class HealthResponse(BaseModel):
    """Liveness probe response."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    version: str = "0.1.0"


class ReadyResponse(BaseModel):
    """Readiness probe response."""

    model_config = ConfigDict(extra="forbid")

    status: str
    database: str
    error: str | None = None
