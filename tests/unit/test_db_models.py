"""Unit tests for database models and session management."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import NullPool

from app.config.settings import Settings
from app.db.base import utc_now
from app.db.models.artifact import ArtifactModel
from app.db.models.job import JobAttemptModel, JobModel
from app.db.models.workflow import WorkflowModel
from app.db.session import (
    close_db,
    get_db_session,
    get_engine_kwargs,
    get_session_factory,
    init_db,
    reset_db_state,
)


def test_utc_now():
    now = utc_now()
    assert now.tzinfo == UTC
    assert isinstance(now, datetime)


def test_get_engine_kwargs():
    settings = Settings(_env_file=None)

    # SQLite
    sqlite_kwargs = get_engine_kwargs("sqlite+aiosqlite:///:memory:", settings)
    assert sqlite_kwargs["poolclass"] is NullPool
    assert sqlite_kwargs["echo"] is False

    # PostgreSQL
    pg_kwargs = get_engine_kwargs("postgresql+asyncpg://localhost/videogen", settings)
    assert pg_kwargs["pool_size"] == settings.database_pool_size
    assert pg_kwargs["max_overflow"] == settings.database_max_overflow
    assert pg_kwargs["pool_timeout"] == settings.database_pool_timeout
    assert pg_kwargs["pool_recycle"] == settings.database_pool_recycle
    assert pg_kwargs["pool_pre_ping"] is True


def test_workflow_model_instantiation():
    wf_id = str(uuid.uuid4())
    wf = WorkflowModel(
        id=wf_id,
        topic="Quantum Computing",
        status="PENDING",
        current_stage="RESEARCH",
        idempotency_key="key-123",
    )
    assert wf.id == wf_id
    assert wf.topic == "Quantum Computing"
    assert wf.status == "PENDING"
    assert wf.current_stage == "RESEARCH"
    assert wf.idempotency_key == "key-123"
    assert wf.error_code is None
    assert wf.error_message is None
    assert wf.completed_at is None


def test_job_model_instantiation():
    job_id = str(uuid.uuid4())
    wf_id = str(uuid.uuid4())
    job = JobModel(
        id=job_id,
        workflow_id=wf_id,
        logical_key="video:shot:1",
        job_type="video_generation",
        stage="VIDEO_GENERATION",
        status="PENDING",
        max_attempts=3,
        input_payload={"prompt": "test prompt"},
        version=1,
    )
    assert job.id == job_id
    assert job.workflow_id == wf_id
    assert job.logical_key == "video:shot:1"
    assert job.job_type == "video_generation"
    assert job.stage == "VIDEO_GENERATION"
    assert job.status == "PENDING"
    assert job.max_attempts == 3
    assert job.version == 1
    assert job.input_payload == {"prompt": "test prompt"}
    assert job.output_payload is None
    assert job.current_attempt_id is None


def test_job_attempt_model_instantiation():
    attempt_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    lease_token = str(uuid.uuid4())
    attempt = JobAttemptModel(
        id=attempt_id,
        job_id=job_id,
        attempt_number=1,
        worker_id="worker-01",
        lease_token=lease_token,
        status="CLAIMED",
        provider="veo",
        provider_operation_id="operations/veo-12345",
        submission_token="wf-1:video:shot:1:1",
        request_payload={"model": "veo-2.0"},
        response_metadata={"progress": 50},
    )
    assert attempt.id == attempt_id
    assert attempt.job_id == job_id
    assert attempt.attempt_number == 1
    assert attempt.worker_id == "worker-01"
    assert attempt.lease_token == lease_token
    assert attempt.status == "CLAIMED"
    assert attempt.provider == "veo"
    assert attempt.provider_operation_id == "operations/veo-12345"
    assert attempt.submission_token == "wf-1:video:shot:1:1"
    assert attempt.request_payload == {"model": "veo-2.0"}
    assert attempt.response_metadata == {"progress": 50}


def test_artifact_model_instantiation():
    art_id = str(uuid.uuid4())
    wf_id = str(uuid.uuid4())
    art = ArtifactModel(
        id=art_id,
        workflow_id=wf_id,
        artifact_type="FINAL_VIDEO",
        storage_path="/tmp/final.mp4",
        gdrive_file_id="gdrive-xyz",
        file_size_bytes=1048576,
        checksum="sha256-hash",
        mime_type="video/mp4",
        lifecycle_status="AVAILABLE",
        artifact_metadata={"duration": 15.0},
    )
    assert art.id == art_id
    assert art.workflow_id == wf_id
    assert art.job_id is None
    assert art.artifact_type == "FINAL_VIDEO"
    assert art.storage_path == "/tmp/final.mp4"
    assert art.gdrive_file_id == "gdrive-xyz"
    assert art.file_size_bytes == 1048576
    assert art.checksum == "sha256-hash"
    assert art.mime_type == "video/mp4"
    assert art.lifecycle_status == "AVAILABLE"
    assert art.artifact_metadata == {"duration": 15.0}


@pytest.mark.asyncio
async def test_session_management_lifecycle():
    reset_db_state()

    # Initialize with in-memory sqlite for fast isolated lifecycle test
    factory = init_db(database_url="sqlite+aiosqlite:///:memory:")
    assert factory is not None
    assert get_session_factory() is factory

    # Context manager success path
    async with get_db_session(factory) as session:
        assert isinstance(session, AsyncSession)

    # Context manager error rollback path
    with pytest.raises(ValueError, match="Rollback trigger"):
        async with get_db_session(factory) as session:
            raise ValueError("Rollback trigger")

    # Re-initialization disposes previous
    new_factory = init_db(database_url="sqlite+aiosqlite:///:memory:")
    assert new_factory is not None

    await close_db()
    reset_db_state()
