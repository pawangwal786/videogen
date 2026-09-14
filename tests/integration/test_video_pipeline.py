from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agents.storyboard.models import StoryboardResult, StoryboardShot
from app.artifacts.models import ArtifactRef
from app.models.video import VideoModel, VideoOperation
from app.video.service import VideoGenerationService


@pytest.fixture
def sample_storyboard():
    workflow_id = uuid4()
    shots = [
        StoryboardShot(
            shot_number=1,
            scene_number=1,
            shot_framing="wide_shot",
            camera_movement="pan_right",
            visual_description="Wide panoramic establishing shot of solar array.",
            video_prompt="Cinematic wide shot of solar farm gleaming at sunrise.",
            negative_prompt="blurry, watermark",
            estimated_duration_seconds=5.0,
        ),
        StoryboardShot(
            shot_number=2,
            scene_number=1,
            shot_framing="close_up",
            camera_movement="zoom_in",
            visual_description="Close up of photovoltaic crystalline cell.",
            video_prompt="Macro slow zoom into crystalline solar cell capturing light reflections.",
            negative_prompt="low quality, artifacts",
            estimated_duration_seconds=8.0,
        ),
        StoryboardShot(
            shot_number=3,
            scene_number=2,
            shot_framing="medium_shot",
            camera_movement="tracking",
            visual_description="Technician inspecting control panel in desert substation.",
            video_prompt="Tracking medium shot of female engineer checking futuristic control display.",
            negative_prompt="grain, watermark",
            estimated_duration_seconds=12.0,
        ),
    ]
    return StoryboardResult(
        workflow_id=workflow_id,
        title="Solar Revolution",
        aspect_ratio="9:16",
        visual_style="cinematic documentary, crisp 4k, anamorphic lens flare",
        target_duration_seconds=25,
        estimated_duration_seconds=25.0,
        total_shots=3,
        shots=shots,
    )


@pytest.mark.asyncio
async def test_video_pipeline_end_to_end_generation(sample_storyboard, tmp_path):
    """Verify end-to-end storyboard shot generation, provider duration resolution, local staging, and artifact persistence."""
    mock_model = AsyncMock(spec=VideoModel)
    mock_model.provider = "veo"
    mock_model.model_name = "veo-2.0-generate-001"

    # Mock submit_generation returning unique operations
    async def fake_submit(request):
        return VideoOperation(
            operation_id=f"ops/shot-{request.shot_number}",
            provider="veo",
            status="submitted",
            done=False,
        )

    # Mock get_operation_status returning completed with fake bytes
    async def fake_status(operation_id):
        shot_num = operation_id.split("-")[-1]
        return VideoOperation(
            operation_id=operation_id,
            provider="veo",
            status="completed",
            done=True,
            video_bytes=f"fake_mp4_shot_{shot_num}".encode(),
        )

    mock_model.submit_generation.side_effect = fake_submit
    mock_model.get_operation_status.side_effect = fake_status

    # Mock artifact storage
    mock_storage = AsyncMock()

    async def fake_upload(
        local_path, destination, workflow_id, artifact_type, mime_type, overwrite=True
    ):
        return ArtifactRef(
            id=uuid4(),
            workflow_id=workflow_id,
            artifact_type=artifact_type,
            filename=local_path.name,
            mime_type=mime_type,
            drive_file_id=f"drive_file_{local_path.name}",
            size_bytes=local_path.stat().st_size,
        )

    mock_storage.upload.side_effect = fake_upload

    staging_dir = tmp_path / "staging"
    service = VideoGenerationService(
        video_model=mock_model,
        storage=mock_storage,
        staging_dir=staging_dir,
        poll_interval_seconds=0.01,
    )

    # Execute generation across all storyboard shots
    records = await service.generate_storyboard(
        sample_storyboard,
        aspect_ratio="9:16",
        max_concurrency=2,
    )

    # 1. Output count and order
    assert len(records) == 3
    assert [r.shot_number for r in records] == [1, 2, 3]

    # 2. Status and provider metadata
    for r in records:
        assert r.status == "completed"
        assert r.provider == "veo"
        assert r.model_name == "veo-2.0-generate-001"
        assert r.aspect_ratio == "9:16"
        assert r.workflow_id == sample_storyboard.workflow_id

    # 3. Provider duration resolution
    # Shot 1: 5.0s -> 5s
    # Shot 2: 8.0s -> 10s (exceeds 7.5s threshold)
    # Shot 3: 12.0s -> 10s
    assert records[0].duration_seconds == 5
    assert records[1].duration_seconds == 10
    assert records[2].duration_seconds == 10

    # 4. Local staging verification
    wf_staging = staging_dir / str(sample_storyboard.workflow_id)
    assert wf_staging.exists()
    for shot_num in [1, 2, 3]:
        staged_file = wf_staging / f"shot_{shot_num}.mp4"
        assert staged_file.exists()
        assert staged_file.read_bytes() == f"fake_mp4_shot_{shot_num}".encode()

    # 5. Cloud storage artifact persistence
    assert mock_storage.upload.await_count == 3
    for r in records:
        assert r.output_artifact_id is not None
        assert r.output_storage_path == f"drive_file_shot_{r.shot_number}.mp4"
        artifact = service.get_artifact(r.logical_key)
        assert artifact is not None
        assert artifact.id == r.output_artifact_id


@pytest.mark.asyncio
async def test_video_pipeline_idempotency_prevents_duplicate_billing(sample_storyboard, tmp_path):
    """Verify that re-running the video pipeline reuses completed generation records and avoids duplicate provider calls."""
    mock_model = AsyncMock(spec=VideoModel)
    mock_model.provider = "veo"
    mock_model.model_name = "veo-2.0-generate-001"

    async def fake_submit(request):
        return VideoOperation(
            operation_id=f"ops/shot-{request.shot_number}",
            provider="veo",
            status="submitted",
            done=False,
        )

    async def fake_status(operation_id):
        return VideoOperation(
            operation_id=operation_id,
            provider="veo",
            status="completed",
            done=True,
            video_bytes=b"video_data",
        )

    mock_model.submit_generation.side_effect = fake_submit
    mock_model.get_operation_status.side_effect = fake_status

    service = VideoGenerationService(
        video_model=mock_model,
        staging_dir=tmp_path / "staging",
        poll_interval_seconds=0.01,
    )

    # First run: 3 shots submitted
    first_run = await service.generate_storyboard(sample_storyboard)
    assert len(first_run) == 3
    assert mock_model.submit_generation.await_count == 3

    # Second run for same storyboard: must NOT call submit_generation again
    second_run = await service.generate_storyboard(sample_storyboard)
    assert len(second_run) == 3
    assert mock_model.submit_generation.await_count == 3  # Still 3! No extra billing

    for r1, r2 in zip(first_run, second_run, strict=True):
        assert r1.job_id == r2.job_id
        assert r1.status == "completed"
        assert r2.status == "completed"
