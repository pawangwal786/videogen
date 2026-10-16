"""Unit tests verifying Phase 9.0 contracts, DTOs, immutability, and fingerprints."""

import typing
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

from app.config.settings import Settings
from app.stages import (
    AssemblyResult,
    AudioClipArtifact,
    AudioStageResult,
    MediaSynthesisResult,
    PipelineStage,
    PublishResult,
    QCMetrics,
    QCResult,
    StageContext,
    StageFingerprintCalculator,
    StageHandler,
    StorageResult,
    StorageUploadRequest,
    TextGenerationRequest,
    TextGenerationResult,
    TTSRequest,
    TTSResult,
    VideoClipArtifact,
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoReconciliationResult,
    VideoSubmission,
)


@pytest.fixture
def mock_settings() -> Settings:
    """Provide a valid Settings instance for testing."""
    return Settings(
        VIDEOGEN_ENV="test",
        VIDEOGEN_API_AUTH_TOKEN="test-auth-token-12345",
        VIDEOGEN_DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/videogen_test",
        VIDEOGEN_TEST_DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/videogen_test",
    )


def test_stage_context_immutability(mock_settings: Settings) -> None:
    """Verify that StageContext is strictly frozen and cannot be mutated."""
    wf_id = uuid4()
    ctx = StageContext(
        workflow_id=wf_id,
        job_id="job-123",
        attempt_id="att-456",
        worker_id="worker-789",
        lease_token="lease-abc",
        attempt_number=1,
        settings=mock_settings,
    )

    assert ctx.workflow_id == wf_id
    assert ctx.job_id == "job-123"
    assert ctx.attempt_number == 1

    # Attempting mutation must raise ValidationError
    with pytest.raises(ValidationError):
        ctx.job_id = "mutated-job"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        ctx.attempt_number = 2  # type: ignore[misc]


def test_zero_any_in_contract_fields() -> None:
    """Verify that zero 'Any' types exist in any declared contract model."""
    contract_classes = [
        StageContext,
        VideoClipArtifact,
        MediaSynthesisResult,
        AudioClipArtifact,
        AudioStageResult,
        AssemblyResult,
        QCMetrics,
        QCResult,
        PublishResult,
        TextGenerationRequest,
        TextGenerationResult,
        VideoGenerationRequest,
        VideoSubmission,
        VideoGenerationResult,
        VideoReconciliationResult,
        TTSRequest,
        TTSResult,
        StorageUploadRequest,
        StorageResult,
    ]

    for cls in contract_classes:
        for field_name, field_info in cls.model_fields.items():
            ann = field_info.annotation
            assert ann is not typing.Any, f"{cls.__name__}.{field_name} uses raw typing.Any!"
            # Check for Union containing Any
            type_args = typing.get_args(ann)
            assert typing.Any not in type_args, (
                f"{cls.__name__}.{field_name} includes typing.Any in type args {type_args}!"
            )


def test_media_synthesis_dto_validation() -> None:
    """Verify MediaSynthesisResult and VideoClipArtifact invariants."""
    wf_id = uuid4()
    clip_id = uuid4()
    checksum = "a" * 64

    clip = VideoClipArtifact(
        shot_number=1,
        artifact_id=clip_id,
        storage_path="data/artifacts/shot_01.mp4",
        duration_seconds=5.0,
        checksum_sha256=checksum,
        provider_operation_id="op-12345",
    )

    result = MediaSynthesisResult(
        workflow_id=wf_id,
        total_clips=1,
        clips=[clip],
        total_duration_seconds=5.0,
    )

    assert result.workflow_id == wf_id
    assert result.total_clips == 1
    assert result.clips[0].provider_operation_id == "op-12345"

    # Check checksum length validation
    with pytest.raises(ValidationError):
        VideoClipArtifact(
            shot_number=1,
            artifact_id=clip_id,
            storage_path="shot.mp4",
            duration_seconds=5.0,
            checksum_sha256="too_short",
        )


def test_audio_stage_dto_validation() -> None:
    """Verify AudioStageResult and AudioClipArtifact invariants."""
    wf_id = uuid4()
    narration_id = uuid4()
    checksum = "b" * 64

    narration = AudioClipArtifact(
        clip_type="narration",
        artifact_id=narration_id,
        storage_path="data/narration.mp3",
        duration_seconds=12.5,
        format="mp3",
        checksum_sha256=checksum,
        sample_rate=44100,
    )

    audio_res = AudioStageResult(
        workflow_id=wf_id,
        narration=narration,
        total_duration_seconds=12.5,
    )

    assert audio_res.narration.sample_rate == 44100
    assert audio_res.background_music is None


def test_assembly_dto_validation() -> None:
    """Verify AssemblyResult invariants."""
    wf_id = uuid4()
    master_id = uuid4()
    checksum = "c" * 64

    res = AssemblyResult(
        workflow_id=wf_id,
        master_artifact_id=master_id,
        storage_path="data/master.mp4",
        duration_seconds=60.0,
        width=1920,
        height=1080,
        frame_rate=30.0,
        video_codec="h264",
        audio_codec="aac",
        checksum_sha256=checksum,
        file_size_bytes=10485760,
    )

    assert res.width == 1920
    assert res.height == 1080
    assert res.video_codec == "h264"


def test_qc_stage_dto_validation() -> None:
    """Verify QCResult and QCMetrics invariants."""
    wf_id = uuid4()

    metrics = QCMetrics(
        actual_duration_seconds=60.2,
        target_duration_seconds=60.0,
        duration_drift_ratio=0.0033,
        loudness_lufs=-14.5,
        has_audio_stream=True,
        has_video_stream=True,
        resolution_width=1920,
        resolution_height=1080,
        frame_rate=30.0,
        black_frame_count=0,
        frozen_frame_count=0,
    )

    qc_pass = QCResult(
        workflow_id=wf_id,
        verdict="PASS",
        metrics=metrics,
    )
    assert qc_pass.verdict == "PASS"
    assert len(qc_pass.failure_reasons) == 0

    qc_fail = QCResult(
        workflow_id=wf_id,
        verdict="FAIL",
        metrics=metrics,
        failure_reasons=["Audio loudness too high"],
    )
    assert qc_fail.verdict == "FAIL"
    assert "Audio loudness too high" in qc_fail.failure_reasons

    # Invalid verdict
    with pytest.raises(ValidationError):
        QCResult(
            workflow_id=wf_id,
            verdict="UNKNOWN",  # type: ignore[arg-type]
            metrics=metrics,
        )


def test_publish_dto_validation() -> None:
    """Verify PublishResult invariants."""
    wf_id = uuid4()
    checksum = "d" * 64

    pub = PublishResult(
        workflow_id=wf_id,
        provider="google_drive",
        remote_file_id="drive-file-998877",
        public_url="https://drive.google.com/file/d/drive-file-998877/view",
        checksum_sha256=checksum,
        file_size_bytes=10485760,
    )

    assert pub.provider == "google_drive"
    assert pub.remote_file_id == "drive-file-998877"
    assert pub.public_url is not None


def test_provider_capability_contracts() -> None:
    """Verify provider capability request/response contracts."""

    # 1. Text provider contract with typed response_model
    class DummyTargetSchema(BaseModel):
        summary: str

    req = TextGenerationRequest(
        user_prompt="Summarize topic X",
        system_prompt="You are a researcher",
        response_model=DummyTargetSchema,
    )
    assert req.response_model is DummyTargetSchema

    res = TextGenerationResult(
        content='{"summary": "test"}',
        provider="gemini",
        model="gemini-2.5-flash",
        usage_prompt_tokens=100,
        usage_completion_tokens=20,
    )
    assert res.provider == "gemini"

    # 2. Video provider contract
    v_req = VideoGenerationRequest(
        client_request_id="shot-01-req",
        prompt="Cinematic shot of laser optics",
        duration_seconds=5,
        aspect_ratio="16:9",
    )
    assert v_req.client_request_id == "shot-01-req"

    v_sub = VideoSubmission(
        provider_operation_id="veo-op-9876",
        status="PENDING",
    )
    assert v_sub.status == "PENDING"

    v_res = VideoGenerationResult(
        provider_operation_id="veo-op-9876",
        status="COMPLETED",
        download_url="https://example.com/video.mp4",
    )
    assert v_res.status == "COMPLETED"

    v_recon = VideoReconciliationResult(
        status="RESOLVED",
        provider_operation_id="veo-op-9876",
    )
    assert v_recon.status == "RESOLVED"

    # 3. TTS provider contract
    tts_req = TTSRequest(
        client_request_id="narration-req",
        text="Hello world narration",
        voice_id="en-US-Neural2-F",
    )
    assert tts_req.client_request_id == "narration-req"
    tts_res = TTSResult(
        audio_bytes=b"fake-mp3-bytes",
        audio_format="mp3",
        duration_seconds=2.5,
        sample_rate=44100,
    )
    assert tts_res.sample_rate == 44100

    # 4. Storage provider contract
    wf_id = uuid4()
    st_req = StorageUploadRequest(
        client_request_id="upload-req",
        workflow_id=wf_id,
        local_file_path="/tmp/master.mp4",
        destination_folder_id="root-folder",
        file_name="master.mp4",
        mime_type="video/mp4",
    )
    assert st_req.client_request_id == "upload-req"
    st_res = StorageResult(
        provider="google_drive",
        remote_file_id="drive-id-123",
        checksum_sha256="e" * 64,
        file_size_bytes=1000,
        published_at=datetime.now(UTC),
    )
    assert st_res.provider == "google_drive"


@pytest.mark.asyncio
async def test_stage_handler_protocol_implementation(mock_settings: Settings) -> None:
    """Verify that StageHandler ABC can be cleanly implemented and executed."""

    class DummyInput(BaseModel):
        query: str

    class DummyOutput(BaseModel):
        answer: str

    class ConcreteHandler(StageHandler[DummyInput, DummyOutput]):
        @property
        def stage_name(self) -> str:
            return "dummy_stage"

        async def load_input(self, context: StageContext) -> DummyInput:
            return DummyInput(query=f"query-for-{context.workflow_id}")

        async def execute(self, input_data: DummyInput, context: StageContext) -> DummyOutput:
            return DummyOutput(answer=f"result: {input_data.query}")

    handler = ConcreteHandler()
    assert handler.stage_name == "dummy_stage"

    ctx = StageContext(
        workflow_id=uuid4(),
        job_id="j1",
        attempt_id="a1",
        worker_id="w1",
        lease_token="t1",
        attempt_number=1,
        settings=mock_settings,
    )

    inp = await handler.load_input(ctx)
    assert inp.query == f"query-for-{ctx.workflow_id}"

    out = await handler.execute(inp, ctx)
    assert out.answer == f"result: query-for-{ctx.workflow_id}"


def test_stage_fingerprint_calculator_determinism_and_sensitivity() -> None:
    """Verify deterministic output and input sensitivity of StageFingerprintCalculator."""
    calc = StageFingerprintCalculator

    # 1. Research fingerprint
    fp1 = calc.for_research("Quantum Computing", 60, "gemini-2.5-flash")
    fp2 = calc.for_research("Quantum Computing", 60, "gemini-2.5-flash")
    assert fp1 == fp2
    assert len(fp1) == 64

    # Sensitivity check: change duration
    fp3 = calc.for_research("Quantum Computing", 90, "gemini-2.5-flash")
    assert fp1 != fp3

    # 2. Script fingerprint
    checksum_a = "1" * 64
    checksum_b = "2" * 64
    s_fp1 = calc.for_script(checksum_a, 140, 60)
    s_fp2 = calc.for_script(checksum_a, 140, 60)
    assert s_fp1 == s_fp2
    assert s_fp1 != calc.for_script(checksum_b, 140, 60)
    assert s_fp1 != calc.for_script(checksum_a, 150, 60)

    # 3. Storyboard fingerprint
    sb_fp1 = calc.for_storyboard(checksum_a, "cinematic", "16:9")
    sb_fp2 = calc.for_storyboard(checksum_a, "cinematic", "16:9")
    assert sb_fp1 == sb_fp2
    assert sb_fp1 != calc.for_storyboard(checksum_a, "anime", "16:9")

    # 4. Media synthesis fingerprint
    ms_fp1 = calc.for_media_synthesis(checksum_a, "16:9", "1080p")
    ms_fp2 = calc.for_media_synthesis(checksum_a, "16:9", "1080p")
    assert ms_fp1 == ms_fp2
    assert ms_fp1 != calc.for_media_synthesis(checksum_a, "9:16", "1080p")

    # 5. Audio fingerprint
    au_fp1 = calc.for_audio(checksum_a, "voice-01", 140)
    au_fp2 = calc.for_audio(checksum_a, "voice-01", 140)
    assert au_fp1 == au_fp2
    assert au_fp1 != calc.for_audio(checksum_a, "voice-02", 140)

    # 6. Assembly fingerprint
    as_fp1 = calc.for_assembly([checksum_a, checksum_b], checksum_a, "fast_1080p")
    as_fp2 = calc.for_assembly([checksum_a, checksum_b], checksum_a, "fast_1080p")
    assert as_fp1 == as_fp2
    # Order sensitivity: reversing manifest clips changes fingerprint
    as_fp3 = calc.for_assembly([checksum_b, checksum_a], checksum_a, "fast_1080p")
    assert as_fp1 != as_fp3

    # 7. QC fingerprint
    qc_fp1 = calc.for_qc(checksum_a, "v1.0")
    qc_fp2 = calc.for_qc(checksum_a, "v1.0")
    assert qc_fp1 == qc_fp2
    assert qc_fp1 != calc.for_qc(checksum_a, "v2.0")

    # 8. Publish fingerprint
    pub_fp1 = calc.for_publish(checksum_a, "folder-root-1")
    pub_fp2 = calc.for_publish(checksum_a, "folder-root-1")
    assert pub_fp1 == pub_fp2
    assert pub_fp1 != calc.for_publish(checksum_a, "folder-root-2")


def test_pipeline_stage_enum() -> None:
    """Verify PipelineStage enum contains all 8 required canonical stages."""
    expected = [
        "RESEARCH",
        "SCRIPT",
        "STORYBOARD",
        "MEDIA_SYNTHESIS",
        "AUDIO",
        "ASSEMBLY",
        "QC",
        "PUBLISH",
    ]
    actual = [stage.value for stage in PipelineStage]
    assert actual == expected
