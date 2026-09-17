"""Dependency injection providers for FastAPI endpoints."""

import asyncio
import secrets
import time

from fastapi import Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config.settings import Settings, get_settings
from app.orchestration.orchestrator import WorkflowOrchestrator


def get_settings_dep(request: Request) -> Settings:
    """Retrieve application settings from app state or global cache."""
    return getattr(request.app.state, "settings", None) or get_settings()


def get_session_factory_dep(request: Request) -> async_sessionmaker[AsyncSession]:
    """Retrieve database session factory from application state."""
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise RuntimeError("Database session factory is not configured on application state.")
    return factory


def get_orchestrator(request: Request) -> WorkflowOrchestrator:
    """Retrieve WorkflowOrchestrator instance from application state."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise RuntimeError("WorkflowOrchestrator is not configured on application state.")
    return orchestrator


def verify_auth(request: Request) -> bool:
    """Verify request authentication against configured API auth token.

    Uses constant-time comparison via secrets.compare_digest and never logs the supplied token.
    If api_auth_token is not set in configuration, authentication is permissive (dev/test default).
    """
    settings = get_settings_dep(request)
    configured_token = settings.api_auth_token
    if configured_token is None or not configured_token.get_secret_value().strip():
        return True

    expected_bytes = configured_token.get_secret_value().strip().encode("utf-8")

    # Extract client token from Authorization or X-API-Key headers
    auth_header = request.headers.get("Authorization")
    api_key_header = request.headers.get("X-API-Key")

    client_token: str | None = None
    if auth_header and auth_header.startswith("Bearer "):
        client_token = auth_header[7:].strip()
    elif api_key_header:
        client_token = api_key_header.strip()

    if not client_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(client_token.encode("utf-8"), expected_bytes):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


def get_idempotency_key(
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> str | None:
    """Validate and normalize Idempotency-Key header."""
    if idempotency_key is None:
        return None

    cleaned = idempotency_key.strip()
    if not cleaned or len(cleaned) > 255:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Idempotency-Key must be a non-empty string with at most 255 characters.",
        )
    return cleaned


class InMemoryRateLimiter:
    """Endpoint-scoped in-memory sliding-window rate limiter.

    Architectural note:
    This in-memory implementation is appropriate for single-process / local deployments.
    For horizontal scaling across multiple API replicas, a distributed store
    (such as Redis with a sliding window script) must be substituted.
    """

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, client_identifier: str) -> None:
        """Check and record request timestamp, raising HTTP 429 if rate limit is exceeded."""
        now = time.monotonic()
        window_start = now - 60.0
        async with self._lock:
            history = self._requests.setdefault(client_identifier, [])
            # Evict timestamps outside current 60s sliding window
            self._requests[client_identifier] = [t for t in history if t > window_start]
            if len(self._requests[client_identifier]) >= self.requests_per_minute:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Rate limit of {self.requests_per_minute} requests per minute exceeded. "
                        "Please try again later."
                    ),
                )
            self._requests[client_identifier].append(now)


async def check_workflow_creation_rate_limit(request: Request) -> None:
    """Endpoint dependency enforcing rate limiting on workflow creation."""
    limiter: InMemoryRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is not None:
        client_ip = request.client.host if request.client else "unknown"
        await limiter.check(client_ip)
