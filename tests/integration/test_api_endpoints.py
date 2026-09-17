"""PostgreSQL-backed integration tests for Phase 8 FastAPI Control Plane."""

import asyncio
import uuid
from datetime import UTC

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
        assert "Retry-After" in r_exceeded.headers
        assert int(r_exceeded.headers["Retry-After"]) >= 1


@pytest.mark.asyncio
async def test_app_shutdown_disposes_owned_database_engine():
    """Verify that create_app() with internally owned database engine disposes its engine upon lifespan shutdown."""
    app = create_app()
    assert app.state.owns_db is True
    engine = app.state.db_engine
    assert engine is not None

    async with app.router.lifespan_context(app):
        # Verify engine connects and queries successfully during lifespan
        async with engine.connect() as conn:
            res = await conn.execute(text("SELECT 1"))
            assert res.scalar() == 1
        # While alive, connection pool has checked in connection
        assert engine.sync_engine.pool.checkedin() == 1

    # After lifespan shutdown, the owned engine has been disposed and pool connections closed
    assert engine.sync_engine.pool.checkedin() == 0


@pytest.mark.asyncio
async def test_app_shutdown_does_not_dispose_external_session_factory(
    session_factory: async_sessionmaker[AsyncSession],
):
    """Verify that create_app() with externally provided session factory does not dispose the external engine."""
    app = create_app(session_factory=session_factory)
    assert app.state.owns_db is False

    async with app.router.lifespan_context(app):
        pass

    # After lifespan shutdown, the external session_factory must remain fully functional and execute queries
    async with session_factory() as session:
        res = await session.execute(text("SELECT 1"))
        assert res.scalar() == 1


@pytest.mark.asyncio
async def test_concurrent_create_workflow_same_idempotency_key_same_payload(
    session_factory: async_sessionmaker[AsyncSession],
):
    """20 concurrent POST /workflows with identical topic and idempotency key serialize safely.

    Contract:
    - Exactly one creator receives 201 Created.
    - Remaining 19 requests receive 200 OK (idempotent replays).
    - Database contains exactly 1 workflow and exactly 1 initial RESEARCH job.
    - All response JSONs reference the identical workflow ID and topic.
    """
    app = create_app(session_factory=session_factory)
    transport = ASGITransport(app=app)
    shared_key = f"concurrent-same-{uuid.uuid4()}"
    topic = "Concurrent Quantum Simulation"

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        tasks = [
            client.post(
                "/workflows",
                json={"topic": topic},
                headers={"Idempotency-Key": shared_key},
            )
            for _ in range(20)
        ]
        responses = await asyncio.gather(*tasks)

    status_codes = [r.status_code for r in responses]
    assert status_codes.count(201) == 1, f"Expected exactly 1 201 Created, got {status_codes}"
    assert status_codes.count(200) == 19, f"Expected 19 200 OK replays, got {status_codes}"

    first_id = responses[0].json()["id"]
    for r in responses:
        data = r.json()
        assert data["id"] == first_id
        assert data["topic"] == topic
        assert data["current_stage"] == JobStage.RESEARCH.value

    # Deep database verification
    async with session_factory() as session:
        # 1. Exactly one workflow row with this idempotency key
        res_wf = await session.execute(
            text(
                "SELECT id, topic, idempotency_key, status, current_stage FROM workflows WHERE idempotency_key = :k"
            ),
            {"k": shared_key},
        )
        wf_rows = res_wf.fetchall()
        assert len(wf_rows) == 1
        assert wf_rows[0].id == first_id
        assert wf_rows[0].topic == topic
        assert wf_rows[0].idempotency_key == shared_key
        assert wf_rows[0].status == WorkflowStatus.PENDING.value
        assert wf_rows[0].current_stage == JobStage.RESEARCH.value

        # 2. Exactly one initial job created for this workflow (no duplicate initial jobs)
        res_jobs = await session.execute(
            text("SELECT id, stage, status FROM jobs WHERE workflow_id = :wfid"),
            {"wfid": first_id},
        )
        job_rows = res_jobs.fetchall()
        assert len(job_rows) == 1
        assert job_rows[0].stage == JobStage.RESEARCH.value
        assert job_rows[0].status == JobStatus.PENDING.value

        # 3. Overall database state invariant: no duplicate records, no orphan jobs
        res_all_wf = await session.execute(text("SELECT count(*) FROM workflows"))
        assert res_all_wf.scalar() == 1

        res_all_jobs = await session.execute(text("SELECT count(*) FROM jobs"))
        assert res_all_jobs.scalar() == 1

        res_dup_keys = await session.execute(
            text(
                "SELECT idempotency_key FROM workflows GROUP BY idempotency_key HAVING count(*) > 1"
            )
        )
        assert len(res_dup_keys.fetchall()) == 0

        res_orphan_jobs = await session.execute(
            text("SELECT id FROM jobs WHERE workflow_id NOT IN (SELECT id FROM workflows)")
        )
        assert len(res_orphan_jobs.fetchall()) == 0


@pytest.mark.asyncio
async def test_concurrent_create_workflow_same_idempotency_key_conflicting_payload(
    session_factory: async_sessionmaker[AsyncSession],
):
    """Concurrent POST /workflows with identical key but conflicting topics results in 1 winner and 409 Conflicts."""
    app = create_app(session_factory=session_factory)
    transport = ASGITransport(app=app)
    shared_key = f"concurrent-conflict-{uuid.uuid4()}"

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        tasks = [
            client.post(
                "/workflows",
                json={"topic": f"Conflicting Topic {i}"},
                headers={"Idempotency-Key": shared_key},
            )
            for i in range(10)
        ]
        responses = await asyncio.gather(*tasks)

    status_codes = [r.status_code for r in responses]
    assert status_codes.count(201) == 1, f"Expected exactly 1 winner with 201, got {status_codes}"
    assert status_codes.count(409) == 9, f"Expected 9 409 Conflicts, got {status_codes}"

    for r in responses:
        if r.status_code == 409:
            err = r.json()["error"]
            assert err["code"] == "IDEMPOTENCY_CONFLICT"
            assert "correlation_id" in err
            assert err["correlation_id"] is not None

    # Verify DB has only 1 workflow
    async with session_factory() as session:
        res_wf = await session.execute(
            text("SELECT COUNT(*) FROM workflows WHERE idempotency_key = :k"),
            {"k": shared_key},
        )
        assert res_wf.scalar() == 1


@pytest.mark.asyncio
async def test_correlation_id_in_error_response_bodies(
    session_factory: async_sessionmaker[AsyncSession],
):
    """Verify that all error responses (401, 404, 409, 413, 422, 429) include correlation_id in response JSON."""
    settings = Settings(
        api_auth_token=SecretStr("corr-secret"),
        api_rate_limit_per_minute=1,
        api_max_request_body_bytes=1024,
    )
    app = create_app(session_factory=session_factory, settings=settings)
    transport = ASGITransport(app=app)
    custom_corr_id = f"test-corr-{uuid.uuid4()}"
    auth_headers = {"Authorization": "Bearer corr-secret", "X-Correlation-ID": custom_corr_id}

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. 401 Unauthorized
        r_401 = await client.get(
            "/workflows/nonexistent", headers={"X-Correlation-ID": custom_corr_id}
        )
        assert r_401.status_code == 401
        err_401 = r_401.json()["error"]
        assert err_401["code"] == "UNAUTHORIZED"
        assert err_401["correlation_id"] == custom_corr_id

        # 2. 404 Not Found
        r_404 = await client.get(f"/workflows/{uuid.uuid4()}", headers=auth_headers)
        assert r_404.status_code == 404
        err_404 = r_404.json()["error"]
        assert err_404["code"] == "WORKFLOW_NOT_FOUND"
        assert err_404["correlation_id"] == custom_corr_id

        # 3. 422 Validation Error
        r_422 = await client.post("/workflows", json={"topic": ""}, headers=auth_headers)
        assert r_422.status_code == 422
        err_422 = r_422.json()["error"]
        assert err_422["code"] == "VALIDATION_ERROR"
        assert err_422["correlation_id"] == custom_corr_id

        # 4. 413 Payload Too Large
        r_413 = await client.post(
            "/workflows",
            content=b"x" * 2048,
            headers={**auth_headers, "Content-Length": "2048"},
        )
        assert r_413.status_code == 413
        err_413 = r_413.json()["error"]
        assert err_413["code"] == "PAYLOAD_TOO_LARGE"
        assert err_413["correlation_id"] == custom_corr_id

        # 5. 429 Rate Limit Exceeded
        # First request uses the 1-per-minute quota
        await client.post("/workflows", json={"topic": "First Topic"}, headers=auth_headers)
        # Second request triggers 429
        r_429 = await client.post(
            "/workflows", json={"topic": "Second Topic"}, headers=auth_headers
        )
        assert r_429.status_code == 429
        err_429 = r_429.json()["error"]
        assert err_429["code"] == "RATE_LIMIT_EXCEEDED"
        assert err_429["correlation_id"] == custom_corr_id


@pytest.mark.asyncio
async def test_streaming_body_size_limiter(
    session_factory: async_sessionmaker[AsyncSession],
):
    """Verify ASGI-level streaming request body limiter behavior."""
    small_settings = Settings(api_max_request_body_bytes=1024)
    app = create_app(session_factory=session_factory, settings=small_settings)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Negative Content-Length -> 400 Bad Request
        r_neg = await client.post("/workflows", content=b"{}", headers={"Content-Length": "-5"})
        assert r_neg.status_code == 400
        assert r_neg.json()["error"]["code"] == "BAD_REQUEST"

        # Malformed Content-Length -> 400 Bad Request
        r_bad = await client.post(
            "/workflows", content=b"{}", headers={"Content-Length": "invalid"}
        )
        assert r_bad.status_code == 400
        assert r_bad.json()["error"]["code"] == "BAD_REQUEST"

        # Chunked transfer exceeding limit without Content-Length
        async def chunk_generator():
            for _ in range(10):
                yield b"x" * 200  # 2000 bytes total > 1024 byte limit

        r_chunked = await client.post(
            "/workflows",
            content=chunk_generator(),
            headers={"Content-Type": "application/json"},
        )
        assert r_chunked.status_code == 413
        assert r_chunked.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


@pytest.mark.asyncio
async def test_body_size_limiter_never_invokes_downstream_on_rejection():
    """Verify ASGI body limiter invariant: once REJECTED, downstream app is never invoked and 413 is emitted once."""
    from app.api.middleware import RequestSizeLimitMiddleware
    from app.config.settings import Settings

    downstream_invoked = False

    async def dummy_app(scope, receive, send):
        nonlocal downstream_invoked
        downstream_invoked = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    settings = Settings(api_max_request_body_bytes=1024)
    limiter = RequestSizeLimitMiddleware(dummy_app, settings=settings)

    # Chunked stream exceeding 1024 bytes
    chunks = [b"a" * 600, b"b" * 600]  # total 1200 bytes > 1024 bytes limit
    chunk_index = 0

    async def mock_receive():
        nonlocal chunk_index
        if chunk_index < len(chunks):
            c = chunks[chunk_index]
            chunk_index += 1
            return {
                "type": "http.request",
                "body": c,
                "more_body": chunk_index < len(chunks),
            }
        return {"type": "http.request", "body": b"", "more_body": False}

    sent_messages = []

    async def mock_send(message):
        sent_messages.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/workflows",
        "headers": [(b"content-type", b"application/json")],
    }

    await limiter(scope, mock_receive, mock_send)

    # Invariant checks:
    # 1. Downstream app was NEVER called
    assert not downstream_invoked, "Downstream app was invoked despite body exceeding limit!"
    # 2. HTTP 413 response was started exactly once
    start_messages = [m for m in sent_messages if m["type"] == "http.response.start"]
    assert len(start_messages) == 1
    assert start_messages[0]["status"] == 413


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", f"/workflows/{uuid.uuid4()}"),
        ("POST", f"/workflows/{uuid.uuid4()}/cancel"),
        ("GET", f"/workflows/{uuid.uuid4()}/jobs"),
        ("GET", f"/workflows/{uuid.uuid4()}/artifacts"),
    ],
)
async def test_protected_routes_require_authentication(
    session_factory: async_sessionmaker[AsyncSession],
    method: str,
    path: str,
):
    """Verify that every protected route family returns 401 Unauthorized when credentials are absent."""
    auth_settings = Settings(api_auth_token=SecretStr("enforced-secret-token"))
    app = create_app(session_factory=session_factory, settings=auth_settings)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.request(method, path)
        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == "Bearer"
        assert resp.json()["error"]["code"] == "UNAUTHORIZED"


def test_artifact_response_sanitization():
    """Verify ArtifactResponse projects via PublicArtifactMetadata whitelist schema, dropping all internal metadata."""
    from datetime import datetime

    from app.api.schemas.artifact import ArtifactResponse, PublicArtifactMetadata
    from app.orchestration.state_machine import ArtifactLifecycleStatus

    raw_internal_metadata = {
        # Whitelisted public fields
        "duration_seconds": 15.0,
        "width": 1920,
        "height": 1080,
        "frame_rate": 30.0,
        "codec": "libx264",
        "audio_codec": "aac",
        "sample_rate": 44100,
        "bitrate_kbps": 2500,
        "format": "mp4",
        # Arbitrary internal fields, secrets, nested objects, and filesystem paths
        "resolution": "1080p",
        "gdrive_folder_id": "sensitive_folder_id",
        "access_token": "secret_oauth_token",
        "client_secret": "sensitive_client_secret",
        "internal_server_path": "/var/run/videogen/temp.raw",
        "provider_job_details": {
            "internal_worker_id": "worker-xyz",
            "oauth_token": "leak-attempt",
            "nested_payload": {"debug": True},
        },
        "custom_unmodeled_scalar": 42,
    }

    resp = ArtifactResponse(
        id=str(uuid.uuid4()),
        workflow_id=str(uuid.uuid4()),
        artifact_type="video",
        storage_path="C:\\Users\\pawan\\AppData\\Local\\Temp\\final_output.mp4",
        status=ArtifactLifecycleStatus.AVAILABLE,
        created_at=datetime.now(UTC),
        artifact_metadata=raw_internal_metadata,  # type: ignore[arg-type]
    )

    # 1. Storage path must be stripped of absolute drive paths
    assert "\\" not in resp.storage_path
    assert "Users" not in resp.storage_path
    assert resp.storage_path == "final_output.mp4"

    # 2. Metadata is strictly projected onto PublicArtifactMetadata whitelist
    assert resp.artifact_metadata is not None
    assert isinstance(resp.artifact_metadata, PublicArtifactMetadata)
    assert resp.artifact_metadata.duration_seconds == 15.0
    assert resp.artifact_metadata.width == 1920
    assert resp.artifact_metadata.height == 1080
    assert resp.artifact_metadata.frame_rate == 30.0
    assert resp.artifact_metadata.codec == "libx264"
    assert resp.artifact_metadata.audio_codec == "aac"
    assert resp.artifact_metadata.sample_rate == 44100
    assert resp.artifact_metadata.bitrate_kbps == 2500
    assert resp.artifact_metadata.format == "mp4"

    # 3. Model dump contains ZERO unmodeled internal/provider fields
    dumped = resp.artifact_metadata.model_dump()
    assert "gdrive_folder_id" not in dumped
    assert "access_token" not in dumped
    assert "client_secret" not in dumped
    assert "resolution" not in dumped
    assert "internal_server_path" not in dumped
    assert "provider_job_details" not in dumped
    assert "custom_unmodeled_scalar" not in dumped


@pytest.mark.parametrize(
    "raw_input,expected_reused",
    [
        ("123e4567-e89b-12d3-a456-426614174000", True),  # valid UUID
        ("my-service.client_job:1234", True),  # valid custom ID matching [a-zA-Z0-9._:-]{1,64}
        ("a" * 64, True),  # exact max length 64
        ("a" * 65, False),  # > 64 chars
        ("", False),  # empty
        ("   ", False),  # whitespace only
        ("id with space", False),  # space
        ("id\twith_tab", False),  # tab
        ("id\rwith_cr", False),  # CR
        ("id\nwith_lf", False),  # LF
        ("id_with_ünicode", False),  # non-ascii
        ("<script>", False),  # < and >
        ('"quoted"', False),  # double quotes
        ("'single'", False),  # single quotes
        ("path\\escape", False),  # backslash
    ],
)
def test_correlation_id_normalization_matrix(raw_input: str, expected_reused: bool):
    """Verify strict character set enforcement: malformed or unsafe correlation IDs are replaced by UUIDs."""
    from app.api.middleware import normalize_correlation_id

    result = normalize_correlation_id(raw_input)
    assert result is not None
    assert 1 <= len(result) <= 64
    if expected_reused:
        assert result == raw_input
    else:
        assert result != raw_input
        uuid.UUID(result)


def test_correlation_id_normalization_none_and_bytes():
    """Verify normalize_correlation_id handles None and raw bytes cleanly."""
    from app.api.middleware import normalize_correlation_id

    res_none = normalize_correlation_id(None)
    uuid.UUID(res_none)

    res_bytes_ok = normalize_correlation_id(b"valid-bytes-corr-123")
    assert res_bytes_ok == "valid-bytes-corr-123"

    res_bytes_bad = normalize_correlation_id(b"<bad-bytes>")
    uuid.UUID(res_bytes_bad)
