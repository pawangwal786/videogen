"""PostgreSQL-backed integration tests for Phase 8 FastAPI Control Plane."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.api.app import create_app
from app.config.settings import Settings
from app.db.base import utc_now
from app.db.models.job import JobAttemptModel, JobModel
from app.orchestration.errors import LeaseConflictError
from app.orchestration.state_machine import AttemptStatus, JobStage, JobStatus, WorkflowStatus
from app.repositories.artifact import ArtifactRepository
from app.repositories.job import JobRepository


@pytest.fixture(autouse=True)
async def clean_database(pg_engine: AsyncEngine):
    """Ensure a clean database state before and after each API test."""
    async with pg_engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE TABLE artifacts, job_attempts, jobs, workflows RESTART IDENTITY CASCADE")
        )
    yield
    async with pg_engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE TABLE artifacts, job_attempts, jobs, workflows RESTART IDENTITY CASCADE")
        )


@pytest.fixture
def session_factory(pg_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Provide a sessionmaker bound to the live PostgreSQL test database."""
    return async_sessionmaker(bind=pg_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.mark.asyncio
async def test_health_probe_liveness_always_ok(
    session_factory: async_sessionmaker[AsyncSession],
):
    """GET /health is a lightweight liveness probe that returns HTTP 200 without DB calls."""
    app = create_app(session_factory=session_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


@pytest.mark.asyncio
async def test_readiness_probe_database_connectivity(
    session_factory: async_sessionmaker[AsyncSession],
):
    """GET /ready actively verifies database connectivity and returns 200 when connected."""
    app = create_app(session_factory=session_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["database"] == "connected"


@pytest.mark.asyncio
async def test_readiness_probe_fails_when_database_disconnected():
    """GET /ready returns 503 Service Unavailable when the database is unreachable, while /health still succeeds."""
    # Create an engine pointing to an unreachable port with a 0.5s timeout
    broken_engine = create_async_engine(
        "postgresql+asyncpg://postgres:wrong@127.0.0.1:59999/videogen_bad",
        connect_args={"timeout": 0.5},
    )
    broken_factory = async_sessionmaker(bind=broken_engine, class_=AsyncSession)
    app = create_app(session_factory=broken_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Liveness must still succeed
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200

        # Readiness must fail with 503
        ready_resp = await client.get("/ready")
        assert ready_resp.status_code == 503
        data = ready_resp.json()
        assert data["status"] == "unavailable"
        assert data["database"] == "disconnected"

    await broken_engine.dispose()


@pytest.mark.asyncio
async def test_create_workflow_and_get_by_id(
    session_factory: async_sessionmaker[AsyncSession],
):
    """POST /workflows persists a workflow and initial RESEARCH job; GET /workflows/{id} retrieves it."""
    app = create_app(session_factory=session_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Create workflow
        resp = await client.post(
            "/workflows",
            json={"topic": "Black hole physics explained"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["topic"] == "Black hole physics explained"
        assert data["status"] == WorkflowStatus.PENDING.value
        assert data["current_stage"] == JobStage.RESEARCH.value
        wf_id = data["id"]

        # Verify initial RESEARCH job was durably created in PostgreSQL
        async with session_factory() as session:
            job_repo = JobRepository(session)
            jobs = await job_repo.list_jobs_for_workflow(wf_id)
            assert len(jobs) == 1
            assert jobs[0].logical_key == "research"
            assert jobs[0].status == JobStatus.PENDING.value

        # Retrieve workflow via GET
        get_resp = await client.get(f"/workflows/{wf_id}")
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert get_data["id"] == wf_id
        assert get_data["topic"] == "Black hole physics explained"


@pytest.mark.asyncio
async def test_idempotency_key_replay_and_semantic_conflict(
    session_factory: async_sessionmaker[AsyncSession],
):
    """Verify Idempotency-Key semantics: same key + same topic replays existing workflow; same key + different topic returns 409."""
    app = create_app(session_factory=session_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        idempotency_key = f"idem-{uuid.uuid4()}"

        # 1. Initial creation (201 Created)
        resp1 = await client.post(
            "/workflows",
            json={"topic": "Generative Video Architecture"},
            headers={"Idempotency-Key": idempotency_key},
        )
        assert resp1.status_code == 201
        wf1 = resp1.json()
        assert wf1["idempotency_key"] == idempotency_key

        # 2. Replay with identical topic (200 OK, same workflow ID)
        resp2 = await client.post(
            "/workflows",
            json={"topic": "Generative Video Architecture"},
            headers={"Idempotency-Key": idempotency_key},
        )
        assert resp2.status_code == 200
        wf2 = resp2.json()
        assert wf2["id"] == wf1["id"]

        # 3. Conflict: same key with different semantic payload (409 Conflict)
        resp3 = await client.post(
            "/workflows",
            json={"topic": "Different Semantic Topic"},
            headers={"Idempotency-Key": idempotency_key},
        )
        assert resp3.status_code == 409
        err = resp3.json()
        assert err["error"]["code"] == "IDEMPOTENCY_CONFLICT"


@pytest.mark.asyncio
async def test_idempotency_key_validation_bounds(
    session_factory: async_sessionmaker[AsyncSession],
):
    """Idempotency-Key header must not be empty/whitespace or exceed 255 characters."""
    app = create_app(session_factory=session_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Whitespace key
        resp_empty = await client.post(
            "/workflows",
            json={"topic": "Topic A"},
            headers={"Idempotency-Key": "   "},
        )
        assert resp_empty.status_code == 422
        assert resp_empty.json()["error"]["code"] == "VALIDATION_ERROR"

        # Key too long (> 255 chars)
        resp_long = await client.post(
            "/workflows",
            json={"topic": "Topic A"},
            headers={"Idempotency-Key": "k" * 256},
        )
        assert resp_long.status_code == 422
        assert resp_long.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_cancel_workflow_fences_active_worker_race(
    session_factory: async_sessionmaker[AsyncSession],
):
    """POST /workflows/{id}/cancel executes the fenced orchestrator path and invalidates active worker leases."""
    app = create_app(session_factory=session_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Create workflow
        resp = await client.post("/workflows", json={"topic": "Workflow to Cancel"})
        assert resp.status_code == 201
        wf_id = resp.json()["id"]

        # 2. Worker claims initial job for this workflow
        worker_id = "test-worker-cancel-api"
        token = str(uuid.uuid4())
        async with session_factory() as session:
            job_repo = JobRepository(session)
            jobs = await job_repo.list_jobs_for_workflow(wf_id)
            assert len(jobs) == 1
            target_job = jobs[0]
            job_id = target_job.id

            target_job.status = JobStatus.CLAIMED.value
            now = utc_now()
            attempt = JobAttemptModel(
                job_id=job_id,
                attempt_number=1,
                worker_id=worker_id,
                lease_token=token,
                status=AttemptStatus.CLAIMED.value,
                started_at=now,
                heartbeat_at=now,
                submission_token=f"{wf_id}:research:1",
            )
            session.add(attempt)
            await session.flush()
            target_job.current_attempt_id = attempt.id
            attempt_id = attempt.id
            await session.commit()

        # 3. Cancel workflow via API
        cancel_resp = await client.post(f"/workflows/{wf_id}/cancel")
        assert cancel_resp.status_code == 200
        cancel_data = cancel_resp.json()
        assert cancel_data["status"] == WorkflowStatus.CANCELLED.value

        # 4. Verify DB state: both job and attempt are CANCELLED
        async with session_factory() as session:
            j = await session.get(JobModel, job_id)
            assert j is not None
            assert j.status == JobStatus.CANCELLED.value
            att = await session.get(JobAttemptModel, attempt_id)
            assert att is not None
            assert att.status == AttemptStatus.CANCELLED.value

        # 5. Worker subsequent mutations are rejected with LeaseConflictError
        async with session_factory() as session:
            job_repo = JobRepository(session)
            with pytest.raises(LeaseConflictError):
                await job_repo.heartbeat_attempt(
                    job_id=job_id, worker_id=worker_id, lease_token=token
                )

            with pytest.raises(LeaseConflictError):
                await job_repo.complete_job(
                    job_id=job_id,
                    worker_id=worker_id,
                    lease_token=token,
                    output_payload={"done": True},
                )

            with pytest.raises(LeaseConflictError):
                await job_repo.fail_job(
                    job_id=job_id,
                    worker_id=worker_id,
                    lease_token=token,
                    error_code="FAIL",
                    error_message="Worker error",
                )


@pytest.mark.asyncio
async def test_cancel_terminal_workflow_is_idempotent(
    session_factory: async_sessionmaker[AsyncSession],
):
    """Cancelling an already cancelled workflow returns 200 OK with unchanged status."""
    app = create_app(session_factory=session_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Create and cancel
        resp = await client.post("/workflows", json={"topic": "Idempotent Cancel"})
        wf_id = resp.json()["id"]

        cancel1 = await client.post(f"/workflows/{wf_id}/cancel")
        assert cancel1.status_code == 200
        assert cancel1.json()["status"] == WorkflowStatus.CANCELLED.value

        # Cancel again
        cancel2 = await client.post(f"/workflows/{wf_id}/cancel")
        assert cancel2.status_code == 200
        assert cancel2.json()["status"] == WorkflowStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_unknown_workflow_returns_404_standard_error(
    session_factory: async_sessionmaker[AsyncSession],
):
    """Lookups or cancellations for unknown workflow IDs return 404 with structured error envelope."""
    app = create_app(session_factory=session_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        missing_id = str(uuid.uuid4())

        get_resp = await client.get(f"/workflows/{missing_id}")
        assert get_resp.status_code == 404
        err = get_resp.json()
        assert err["error"]["code"] == "WORKFLOW_NOT_FOUND"

        cancel_resp = await client.post(f"/workflows/{missing_id}/cancel")
        assert cancel_resp.status_code == 404
        assert cancel_resp.json()["error"]["code"] == "WORKFLOW_NOT_FOUND"

        jobs_resp = await client.get(f"/workflows/{missing_id}/jobs")
        assert jobs_resp.status_code == 404
        assert jobs_resp.json()["error"]["code"] == "WORKFLOW_NOT_FOUND"

        artifacts_resp = await client.get(f"/workflows/{missing_id}/artifacts")
        assert artifacts_resp.status_code == 404
        assert artifacts_resp.json()["error"]["code"] == "WORKFLOW_NOT_FOUND"


@pytest.mark.asyncio
async def test_list_jobs_and_artifacts_for_workflow(
    session_factory: async_sessionmaker[AsyncSession],
):
    """GET /workflows/{id}/jobs and GET /workflows/{id}/artifacts return cleanly mapped DTOs without storage leak."""
    app = create_app(session_factory=session_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Create workflow
        resp = await client.post("/workflows", json={"topic": "Subresources Topic"})
        wf_id = resp.json()["id"]

        # List jobs: must contain the initial RESEARCH job
        jobs_resp = await client.get(f"/workflows/{wf_id}/jobs")
        assert jobs_resp.status_code == 200
        jobs_data = jobs_resp.json()
        assert jobs_data["total"] == 1
        job = jobs_data["jobs"][0]
        assert job["logical_key"] == "research"
        assert job["job_type"] == "research"
        assert job["stage"] == JobStage.RESEARCH.value

        # Insert an artifact directly via repository
        async with session_factory() as session:
            art_repo = ArtifactRepository(session)
            await art_repo.create_artifact(
                workflow_id=wf_id,
                artifact_type="script",
                storage_path="artifacts/test_script.json",
                mime_type="application/json",
                file_size_bytes=1024,
                checksum="sha256:abcd",
                gdrive_file_id="sensitive-gdrive-id-12345",  # Internal storage detail
            )
            await session.commit()

        # Query artifacts endpoint
        art_resp = await client.get(f"/workflows/{wf_id}/artifacts")
        assert art_resp.status_code == 200
        art_data = art_resp.json()
        assert art_data["total"] == 1
        artifact = art_data["artifacts"][0]
        assert artifact["artifact_type"] == "script"
        assert artifact["storage_path"] == "artifacts/test_script.json"
        assert artifact["file_size_bytes"] == 1024
        # Crucial security assertion: gdrive_file_id is NOT in the public schema
        assert "gdrive_file_id" not in artifact


@pytest.mark.asyncio
async def test_authentication_boundary(
    session_factory: async_sessionmaker[AsyncSession],
):
    """When api_auth_token is configured, requests without valid credentials return 401."""
    token = "secret-production-token-999"
    secure_settings = Settings(api_auth_token=SecretStr(token))
    app = create_app(session_factory=session_factory, settings=secure_settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Health and ready remain unauthenticated
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/ready")).status_code == 200

        # 2. Workflow endpoints require auth: missing header -> 401
        resp_no_auth = await client.post("/workflows", json={"topic": "Secure Topic"})
        assert resp_no_auth.status_code == 401
        assert resp_no_auth.json()["error"]["code"] == "UNAUTHORIZED"

        # 3. Invalid token -> 401
        resp_bad_auth = await client.post(
            "/workflows",
            json={"topic": "Secure Topic"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp_bad_auth.status_code == 401
        assert resp_bad_auth.json()["error"]["code"] == "UNAUTHORIZED"

        # 4. Valid Bearer token -> 201
        resp_bearer = await client.post(
            "/workflows",
            json={"topic": "Secure Topic"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp_bearer.status_code == 201

        # 5. Valid X-API-Key header -> 201
        resp_api_key = await client.post(
            "/workflows",
            json={"topic": "Secure Topic 2"},
            headers={"X-API-Key": token},
        )
        assert resp_api_key.status_code == 201


@pytest.mark.asyncio
async def test_correlation_id_propagation(
    session_factory: async_sessionmaker[AsyncSession],
):
    """X-Correlation-ID is echoed back in response headers, or generated as UUID if omitted."""
    app = create_app(session_factory=session_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Supplied correlation ID
        custom_id = "trace-client-request-abc-123"
        resp1 = await client.get("/health", headers={"X-Correlation-ID": custom_id})
        assert resp1.headers.get("X-Correlation-ID") == custom_id

        # Generated correlation ID when absent
        resp2 = await client.get("/health")
        gen_id = resp2.headers.get("X-Correlation-ID")
        assert gen_id is not None
        # Verify valid UUID format
        assert uuid.UUID(gen_id).version == 4


@pytest.mark.asyncio
async def test_request_payload_and_body_size_limits(
    session_factory: async_sessionmaker[AsyncSession],
):
    """Payload validation rejects empty/oversized topics with 422; body exceeding byte limit returns 413."""
    app = create_app(session_factory=session_factory)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Empty topic -> 422
        resp_empty = await client.post("/workflows", json={"topic": ""})
        assert resp_empty.status_code == 422
        assert resp_empty.json()["error"]["code"] == "VALIDATION_ERROR"

        # Whitespace topic -> 422
        resp_ws = await client.post("/workflows", json={"topic": "   "})
        assert resp_ws.status_code == 422
        assert resp_ws.json()["error"]["code"] == "VALIDATION_ERROR"

        # Topic > 500 characters -> 422
        resp_too_long = await client.post("/workflows", json={"topic": "x" * 501})
        assert resp_too_long.status_code == 422
        assert resp_too_long.json()["error"]["code"] == "VALIDATION_ERROR"

        # Extra forbidden field in request body -> 422
        resp_extra = await client.post("/workflows", json={"topic": "Valid", "unknown_field": True})
        assert resp_extra.status_code == 422
        assert resp_extra.json()["error"]["code"] == "VALIDATION_ERROR"

        # Oversized body exceeding 1MB limit -> 413
        oversized_headers = {"Content-Length": str(2_000_000)}
        resp_oversized = await client.post(
            "/workflows",
            content=b"x" * 100,  # Content-length header simulation
            headers=oversized_headers,
        )
        assert resp_oversized.status_code == 413
        assert resp_oversized.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


@pytest.mark.asyncio
async def test_workflow_creation_rate_limiting(
    session_factory: async_sessionmaker[AsyncSession],
):
    """POST /workflows rate limiter enforces configured limit and returns 429 when exceeded."""
    limited_settings = Settings(api_rate_limit_per_minute=3)
    app = create_app(session_factory=session_factory, settings=limited_settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # First 3 requests succeed
        for i in range(3):
            r = await client.post("/workflows", json={"topic": f"Topic {i}"})
            assert r.status_code == 201

        # 4th request exceeds rate limit -> 429
        r_exceeded = await client.post("/workflows", json={"topic": "Topic Exceeded"})
        assert r_exceeded.status_code == 429
        err = r_exceeded.json()
        assert err["error"]["code"] == "RATE_LIMIT_EXCEEDED"
