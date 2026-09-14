from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agents.storyboard.models import StoryboardShot
from app.artifacts.models import ArtifactRef
from app.models.errors import ModelResponseError
from app.models.video import VideoModel, VideoOperation
from app.video.service import VideoGenerationService


@pytest.fixture
def sample_shot():
    return StoryboardShot(
        shot_number=1,
        scene_number=1,
        shot_framing="wide_shot",
        camera_movement="pan_right",
        visual_description="Wide panoramic shot of futuristic city with flying vehicles.",
        video_prompt="4k cinematic footage of futuristic metropolis at dusk, steady pan right.",
        negative_prompt="blurry, watermark",
        estimated_duration_seconds=6.0,  # <= 7.5 -> resolves to 5s
    )


@pytest.fixture
def mock_video_model():
    model = AsyncMock(spec=VideoModel)
    model.provider = "veo"
    model.model_name = "veo-2.0-generate-001"
    model.submit_generation = AsyncMock()
    model.get_operation_status = AsyncMock()
    return model


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.upload = AsyncMock()
    return storage


def test_resolve_provider_duration():
    assert VideoGenerationService.resolve_provider_duration(4.0) == 5
    assert VideoGenerationService.resolve_provider_duration(7.5) == 5
    assert VideoGenerationService.resolve_provider_duration(7.6) == 10
    assert VideoGenerationService.resolve_provider_duration(15.0) == 10


@pytest.mark.asyncio
async def test_generate_shot_success(mock_video_model, mock_storage, sample_shot, tmp_path):
    wf_id = uuid4()
    mock_video_model.submit_generation.return_value = VideoOperation(
        operation_id="ops/test-op-1",
        provider="veo",
        status="submitted",
        done=False,
    )
    mock_video_model.get_operation_status.return_value = VideoOperation(
        operation_id="ops/test-op-1",
        provider="veo",
        status="completed",
        done=True,
        video_bytes=b"fake_mp4_content",
    )

    art_id = uuid4()
    mock_storage.upload.return_value = ArtifactRef(
        id=art_id,
        workflow_id=wf_id,
        artifact_type="video_shot",
        filename="shot_1.mp4",
        mime_type="video/mp4",
        drive_file_id="drive_file_12345",
        size_bytes=len(b"fake_mp4_content"),
    )

    service = VideoGenerationService(
        video_model=mock_video_model,
        storage=mock_storage,
        staging_dir=tmp_path / "staging",
        poll_interval_seconds=0.01,
    )

    record = await service.generate_shot(sample_shot, wf_id)

    assert record.status == "completed"
    assert record.workflow_id == wf_id
    assert record.shot_number == 1
    assert record.duration_seconds == 5  # Resolved from 6.0s
    assert record.output_artifact_id == art_id
    assert record.output_storage_path == "drive_file_12345"

    # Verify staging file was created on disk
    staged_file = tmp_path / "staging" / str(wf_id) / "shot_1.mp4"
    assert staged_file.exists()
    assert staged_file.read_bytes() == b"fake_mp4_content"

    mock_video_model.submit_generation.assert_awaited_once()
    mock_storage.upload.assert_awaited_once()


@pytest.mark.asyncio
async def test_stable_logical_idempotency_reuse_completed(
    mock_video_model, mock_storage, sample_shot, tmp_path
):
    wf_id = uuid4()
    mock_video_model.submit_generation.return_value = VideoOperation(
        operation_id="ops/test-op-1",
        provider="veo",
        status="submitted",
        done=False,
    )
    mock_video_model.get_operation_status.return_value = VideoOperation(
        operation_id="ops/test-op-1",
        provider="veo",
        status="completed",
        done=True,
        video_bytes=b"fake_mp4_content",
    )
    mock_storage.upload.return_value = ArtifactRef(
        id=uuid4(),
        workflow_id=wf_id,
        artifact_type="video_shot",
        filename="shot_1.mp4",
        mime_type="video/mp4",
        drive_file_id="drive_file_12345",
        size_bytes=len(b"fake_mp4_content"),
    )

    service = VideoGenerationService(
        video_model=mock_video_model,
        storage=mock_storage,
        staging_dir=tmp_path / "staging",
        poll_interval_seconds=0.01,
    )

    # First execution -> generates
    record1 = await service.generate_shot(sample_shot, wf_id)
    assert record1.status == "completed"
    assert mock_video_model.submit_generation.await_count == 1

    # Second execution for same logical shot -> returns existing completed record immediately
    record2 = await service.generate_shot(sample_shot, wf_id)
    assert record2.status == "completed"
    assert record2.job_id == record1.job_id
    # submit_generation MUST NOT be called a second time
    assert mock_video_model.submit_generation.await_count == 1


@pytest.mark.asyncio
async def test_stable_logical_idempotency_resume_active(
    mock_video_model, mock_storage, sample_shot, tmp_path
):
    wf_id = uuid4()
    # Mock submit
    mock_video_model.submit_generation.return_value = VideoOperation(
        operation_id="ops/test-op-active",
        provider="veo",
        status="submitted",
        done=False,
    )
    # First poll returns still processing
    mock_video_model.get_operation_status.return_value = VideoOperation(
        operation_id="ops/test-op-active",
        provider="veo",
        status="completed",
        done=True,
        video_bytes=b"resumed_bytes",
    )

    service = VideoGenerationService(
        video_model=mock_video_model,
        storage=None,  # local only
        staging_dir=tmp_path / "staging",
        poll_interval_seconds=0.01,
    )

    # Simulate an existing active job record in the registry
    logical_key = f"{wf_id}:shot:1"
    existing_record = service._create_new_job_record(
        sample_shot, wf_id, logical_key, "9:16", attempt=1
    )
    existing_record.operation_id = "ops/test-op-active"
    existing_record.transition_to("submitted")

    # Calling generate_shot should NOT submit a new operation; it resumes polling the active operation!
    completed_record = await service.generate_shot(sample_shot, wf_id)

    assert completed_record.status == "completed"
    assert completed_record.operation_id == "ops/test-op-active"
    mock_video_model.submit_generation.assert_not_awaited()
    mock_video_model.get_operation_status.assert_awaited()


@pytest.mark.asyncio
async def test_generate_shot_provider_failure_raises(mock_video_model, sample_shot, tmp_path):
    wf_id = uuid4()
    mock_video_model.submit_generation.return_value = VideoOperation(
        operation_id="ops/test-failed",
        provider="veo",
        status="submitted",
        done=False,
    )
    mock_video_model.get_operation_status.return_value = VideoOperation(
        operation_id="ops/test-failed",
        provider="veo",
        status="failed",
        done=True,
        error_message="Safety policy violation: unacceptable content.",
    )

    service = VideoGenerationService(
        video_model=mock_video_model,
        staging_dir=tmp_path / "staging",
        poll_interval_seconds=0.01,
    )

    with pytest.raises(ModelResponseError) as exc_info:
        await service.generate_shot(sample_shot, wf_id)

    assert "Safety policy violation" in str(exc_info.value)

    # Verify job record was transitioned to failed
    job = service.get_job(f"{wf_id}:shot:1")
    assert job is not None
    assert job.status == "failed"
    assert job.error_message == "Safety policy violation: unacceptable content."
