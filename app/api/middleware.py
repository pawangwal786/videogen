"""API middleware for request correlation, payload size limiting, and security."""

import uuid

from fastapi import status
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import build_error_response
from app.config.settings import Settings, get_settings
from app.logging import bind_correlation, clear_correlation, get_logger

logger = get_logger(__name__)


class CorrelationIdMiddleware:
    """Pure ASGI middleware extracting or generating X-Correlation-ID and propagating it to logs and response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        corr_id: str | None = None
        for k, v in scope.get("headers", []):
            if k.lower() in (b"x-correlation-id", b"x-request-id"):
                val = v.decode("latin1", errors="replace").strip()
                if val and len(val) <= 128:
                    corr_id = val
                break

        if not corr_id:
            corr_id = str(uuid.uuid4())

        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["correlation_id"] = corr_id

        bind_correlation(correlation_id=corr_id)

        async def send_with_correlation(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                has_header = any(k.lower() == b"x-correlation-id" for k, _ in headers)
                if not has_header:
                    headers.append((b"x-correlation-id", corr_id.encode("latin1")))
                    message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_correlation)
        finally:
            clear_correlation()


class RequestSizeLimitMiddleware:
    """ASGI-level body limiter with an explicit receive-state machine.

    Enforces application-level payload limits:
    1. Early rejection when Content-Length exceeds configured max bytes (or is negative/malformed).
    2. Bounded streaming enforcement for chunked/streaming requests where Content-Length is omitted.
       Once the limit is crossed, transitions to REJECTED, drains remaining stream, emits a
       standardized HTTP 413 exactly once, and guarantees the downstream application is never invoked.
    """

    def __init__(self, app: ASGIApp, settings: Settings | None = None) -> None:
        self.app = app
        self._settings = settings or get_settings()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_bytes = self._settings.api_max_request_body_bytes
        raw_headers = scope.get("headers", [])

        # Extract correlation ID from scope state or headers
        corr_id: str | None = None
        if "state" in scope and "correlation_id" in scope["state"]:
            corr_id = scope["state"]["correlation_id"]
        else:
            for k, v in raw_headers:
                if k.lower() in (b"x-correlation-id", b"x-request-id"):
                    corr_id = v.decode("latin1", errors="replace").strip()
                    break

        if not corr_id:
            corr_id = str(uuid.uuid4())

        # Extract Content-Length if present
        content_length_raw: bytes | None = None
        for k, v in raw_headers:
            if k.lower() == b"content-length":
                content_length_raw = v
                break

        # Layer 1: Content-Length validation
        if content_length_raw is not None:
            try:
                cl = int(content_length_raw.decode("ascii"))
                if cl < 0:
                    while True:
                        msg = await receive()
                        if not msg.get("more_body", False) or msg["type"] == "http.disconnect":
                            break
                    response = build_error_response(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        code="BAD_REQUEST",
                        message="Content-Length header must not be negative.",
                        correlation_id=corr_id,
                    )
                    await response(scope, receive, send)
                    return

                if cl > max_bytes:
                    while True:
                        msg = await receive()
                        if not msg.get("more_body", False) or msg["type"] == "http.disconnect":
                            break
                    response = build_error_response(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        code="PAYLOAD_TOO_LARGE",
                        message=f"Request payload size {cl} bytes exceeds limit of {max_bytes} bytes.",
                        correlation_id=corr_id,
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                while True:
                    msg = await receive()
                    if not msg.get("more_body", False) or msg["type"] == "http.disconnect":
                        break
                response = build_error_response(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    code="BAD_REQUEST",
                    message="Invalid Content-Length header.",
                    correlation_id=corr_id,
                )
                await response(scope, receive, send)
                return

        # Layer 2: Streaming / Chunked receive state machine (NORMAL -> REJECTED)
        received_bytes = 0
        chunks: list[bytes] = []
        state = "NORMAL"

        while True:
            message = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                received_bytes += len(chunk)
                if received_bytes > max_bytes:
                    state = "REJECTED"
                    # Drain and discard remaining chunks from client stream
                    while message.get("more_body", False):
                        message = await receive()
                    break
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                return

        if state == "REJECTED":
            # Terminal state: emit 413 exactly once, do NOT invoke downstream app
            response = build_error_response(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                code="PAYLOAD_TOO_LARGE",
                message=f"Request body exceeded maximum allowed limit of {max_bytes} bytes.",
                correlation_id=corr_id,
            )
            await response(scope, receive, send)
            return

        # Stream completed within allowed limit: state remains NORMAL
        # Replay buffered chunks to downstream application
        body_bytes = b"".join(chunks)
        sent_body = False

        async def replay_receive() -> Message:
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {
                    "type": "http.request",
                    "body": body_bytes,
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay_receive, send)
