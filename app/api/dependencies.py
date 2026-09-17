"""Dependency injection providers for FastAPI endpoints."""

import asyncio
import hashlib
import math
import secrets
import time
from dataclasses import dataclass
from typing import Literal

from fastapi import Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config.settings import Settings, get_settings
from app.orchestration.orchestrator import WorkflowOrchestrator


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """Identity representation for authenticated callers containing zero raw secrets.

    Carries only the credential type and a one-way SHA-256 hash suitable for rate limiting
    and audit logging.
    """

    credential_type: Literal["bearer", "api_key", "anonymous"]
    credential_hash: str | None


def extract_client_credential(
    request: Request,
) -> tuple[Literal["bearer", "api_key"] | None, str | None]:
    """Extract raw client credential and type from HTTP request headers.

    Keeps the token as a temporary local string without persisting it in reusable state.
    """
    auth_header = request.headers.get("Authorization")
    api_key_header = request.headers.get("X-API-Key")

    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token:
            return "bearer", token

    if api_key_header:
        token = api_key_header.strip()
        if token:
            return "api_key", token

    return None, None


def get_authenticated_principal(request: Request) -> AuthenticatedPrincipal:
    """Derive an AuthenticatedPrincipal for the current request without storing raw secrets.

    Caches the derived principal on request.state.principal for downstream reuse.
    """
    cached = getattr(request.state, "principal", None)
    if cached is not None and isinstance(cached, AuthenticatedPrincipal):
        return cached

    cred_type, raw_token = extract_client_credential(request)
    if cred_type is not None and raw_token is not None:
        h = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        principal = AuthenticatedPrincipal(credential_type=cred_type, credential_hash=h)
    else:
        principal = AuthenticatedPrincipal(credential_type="anonymous", credential_hash=None)

    request.state.principal = principal
    return principal


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

    Uses constant-time comparison via secrets.compare_digest on a local token string.
    Never stores raw tokens in application state or principal objects.
    If api_auth_token is not set in configuration, authentication is permissive (dev/test default).
    """
    settings = get_settings_dep(request)
    configured_token = settings.api_auth_token
    if configured_token is None or not configured_token.get_secret_value().strip():
        get_authenticated_principal(request)
        return True

    expected_bytes = configured_token.get_secret_value().strip().encode("utf-8")
    _, raw_token = extract_client_credential(request)

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(raw_token.encode("utf-8"), expected_bytes):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Populate and cache safe principal (hash only)
    get_authenticated_principal(request)
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
        """Check and record request timestamp, raising HTTP 429 with dynamic Retry-After if exceeded."""
        now = time.monotonic()
        window_start = now - 60.0
        async with self._lock:
            history = self._requests.setdefault(client_identifier, [])
            # Evict timestamps outside current 60s sliding window
            self._requests[client_identifier] = [t for t in history if t > window_start]
            if len(self._requests[client_identifier]) >= self.requests_per_minute:
                oldest_ts = self._requests[client_identifier][0]
                remaining_seconds = max(1, math.ceil((oldest_ts + 60.0) - now))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Rate limit of {self.requests_per_minute} requests per minute exceeded. "
                        "Please try again later."
                    ),
                    headers={"Retry-After": str(remaining_seconds)},
                )
            self._requests[client_identifier].append(now)


async def check_workflow_creation_rate_limit(request: Request) -> None:
    """Endpoint dependency enforcing rate limiting on workflow creation.

    Identifies caller preferentially by authenticated principal credential hash
    before falling back to client IP. Plaintext tokens are never stored or logged.
    """
    limiter: InMemoryRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is not None:
        principal = get_authenticated_principal(request)
        if principal.credential_hash:
            client_id = f"token:{principal.credential_hash[:16]}"
        else:
            client_ip = request.client.host if request.client else "unknown"
            client_id = f"ip:{client_ip}"

        await limiter.check(client_id)
