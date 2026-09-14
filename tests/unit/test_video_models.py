from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models.video import (
    VideoGenerationRequest,
    VideoJobRecord,
    VideoOperation,
)


def test_video_generation_request_valid():
    wf_id = uuid4()
    req = VideoGenerationRequest(
        workflow_id=wf_id,
        shot_number=1,
        prompt="Cinematic tracking shot through server rack room with neon blue cooling lines.",
        negative_prompt="blurry, watermark",
        duration_seconds=5,
        aspect_ratio="9:16",
        attempt=1,
    )

    assert req.workflow_id == wf_id
    assert req.shot_number == 1
    assert req.duration_seconds == 5
    assert req.aspect_ratio == "9:16"
    assert req.attempt == 1

    # Verify stable logical key vs attempt key
    assert req.logical_key == f"{wf_id}:shot:1"
    assert req.attempt_key == f"{wf_id}:shot:1:attempt:1"


def test_video_generation_request_validation_errors():
    wf_id = uuid4()

    # Shot number must be >= 1
    with pytest.raises(ValidationError):
        VideoGenerationRequest(
            workflow_id=wf_id,
            shot_number=0,
            prompt="Valid prompt text for video generation.",
            duration_seconds=5,
        )

    # Prompt too short (< 10 chars)
    with pytest.raises(ValidationError):
        VideoGenerationRequest(
            workflow_id=wf_id,
            shot_number=1,
            prompt="Short",
            duration_seconds=5,
        )

    # Duration seconds out of bounds (< 1)
    with pytest.raises(ValidationError):
        VideoGenerationRequest(
            workflow_id=wf_id,
            shot_number=1,
            prompt="Valid prompt text for video generation.",
            duration_seconds=0,
        )


def test_video_operation_model():
    op = VideoOperation(
        operation_id="veo-op-123",
        provider="veo",
        status="completed",
        done=True,
        video_bytes=b"fake_mp4_bytes",
        metadata={"resolution": "720p"},
    )
    assert op.operation_id == "veo-op-123"
    assert op.provider == "veo"
    assert op.done is True
    assert op.video_bytes == b"fake_mp4_bytes"


def test_video_job_record_lifecycle_transitions():
    wf_id = uuid4()
    record = VideoJobRecord(
        workflow_id=wf_id,
        shot_number=1,
        logical_key=f"{wf_id}:shot:1",
        provider="veo",
        model_name="veo-2.0-generate-001",
        prompt="Valid prompt text for video generation.",
        duration_seconds=5,
        aspect_ratio="9:16",
    )

    assert record.status == "queued"
    assert record.started_at is None
    assert record.completed_at is None

    # Transition to submitted -> sets started_at
    record.transition_to("submitted")
    assert record.status == "submitted"
    assert record.started_at is not None
    assert record.completed_at is None

    # Transition to processing
    record.transition_to("processing")
    assert record.status == "processing"

    # Transition to completed -> sets completed_at
    record.transition_to("completed")
    assert record.status == "completed"
    assert record.completed_at is not None

    # Cannot transition out of terminal state
    with pytest.raises(ValueError) as exc_info:
        record.transition_to("processing")
    assert "Invalid lifecycle transition" in str(exc_info.value)


def test_video_job_record_invalid_transition():
    wf_id = uuid4()
    record = VideoJobRecord(
        workflow_id=wf_id,
        shot_number=1,
        logical_key=f"{wf_id}:shot:1",
        provider="veo",
        model_name="veo-2.0-generate-001",
        prompt="Valid prompt text for video generation.",
        duration_seconds=5,
        aspect_ratio="9:16",
    )

    # Cannot jump straight from queued to completed
    with pytest.raises(ValueError) as exc_info:
        record.transition_to("completed")
    assert "Invalid lifecycle transition" in str(exc_info.value)
