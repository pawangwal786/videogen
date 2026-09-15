import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.artifacts.models import ArtifactRef
from app.mpt.errors import (
    MediaConfigurationError,
    MediaProbeError,
    MediaValidationError,
)
from app.mpt.models import (
    AudioTrack,
    MediaAssemblyRequest,
    MediaClip,
    MediaInfo,
    MediaProfile,
)
from app.mpt.service import MediaAssemblyService


@pytest.fixture
def mock_processor():
    proc = AsyncMock()
    proc.probe.return_value = MediaInfo(
        duration_seconds=15.0,
        width=1080,
        height=1920,
        frame_rate=30.0,
        video_codec="h264",
        has_video=True,
    )

    async def fake_norm(source, output, profile, target_duration=None, tolerance_seconds=0.5):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"normalized")
        return MediaInfo(
            duration_seconds=target_duration or 5.0,
            width=profile.width,
            height=profile.height,
            frame_rate=30.0,
            video_codec="h264",
            has_video=True,
        )

    async def fake_concat(clips, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"concatenated")
        return MediaInfo(
            duration_seconds=15.0,
            width=1080,
            height=1920,
            frame_rate=30.0,
            video_codec="h264",
            has_video=True,
        )

    async def fake_add_audio(video, audio, output, video_duration):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"final_with_audio")
        return MediaInfo(
            duration_seconds=video_duration,
            width=1080,
            height=1920,
            frame_rate=30.0,
            video_codec="h264",
            audio_codec="aac",
            has_video=True,
            has_audio=True,
        )

    proc.normalize_clip.side_effect = fake_norm
    proc.concatenate.side_effect = fake_concat
    proc.add_audio.side_effect = fake_add_audio
    return proc


@pytest.mark.asyncio
async def test_assemble_video_only_success(mock_processor, tmp_path):
    clip1 = tmp_path / "clip1.mp4"
    clip2 = tmp_path / "clip2.mp4"
    clip1.write_bytes(b"clip1")
    clip2.write_bytes(b"clip2")

    wf_id = uuid4()
    profile = MediaProfile.from_aspect_ratio("9:16")

    req = MediaAssemblyRequest(
        workflow_id=wf_id,
        clips=[
            MediaClip(shot_number=1, source_path=clip1, target_duration_seconds=5.0),
            MediaClip(shot_number=2, source_path=clip2, target_duration_seconds=10.0),
        ],
        profile=profile,
    )

    staging = tmp_path / "staging"
    service = MediaAssemblyService(
        media_processor=mock_processor,
        staging_dir=staging,
        max_concurrency=2,
    )

    result = await service.assemble(req)

    assert result.workflow_id == wf_id
    assert result.duration_seconds == 15.0
    assert result.output_path.exists()
    assert result.artifact_ref is None

    # Normalized called for both clips
    assert mock_processor.normalize_clip.await_count == 2
    # Concat called once
    assert mock_processor.concatenate.await_count == 1
    # Audio not called
    assert mock_processor.add_audio.await_count == 0

    # Intermediate directories should be cleaned up
    work_dir = staging / str(wf_id) / "assembly"
    assert not (work_dir / "normalized").exists()
    assert not (work_dir / "concat").exists()
    assert (work_dir / "final" / "final_video.mp4").exists()


@pytest.mark.asyncio
async def test_assemble_with_audio_track(mock_processor, tmp_path):
    clip1 = tmp_path / "clip1.mp4"
    clip1.write_bytes(b"clip1")
    audio = tmp_path / "bg_music.mp3"
    audio.write_bytes(b"audio")

    wf_id = uuid4()
    profile = MediaProfile.from_aspect_ratio("9:16")

    req = MediaAssemblyRequest(
        workflow_id=wf_id,
        clips=[
            MediaClip(shot_number=1, source_path=clip1, target_duration_seconds=5.0),
        ],
        profile=profile,
        audio_track=AudioTrack(source_path=audio, volume=0.8, loop=True),
    )

    staging = tmp_path / "staging"
    service = MediaAssemblyService(
        media_processor=mock_processor,
        staging_dir=staging,
    )

    result = await service.assemble(req)

    assert result.workflow_id == wf_id
    assert result.duration_seconds == 15.0
    assert mock_processor.add_audio.await_count == 1


@pytest.mark.asyncio
async def test_assemble_with_cloud_storage(mock_processor, tmp_path):
    clip1 = tmp_path / "clip1.mp4"
    clip1.write_bytes(b"clip1")

    wf_id = uuid4()
    profile = MediaProfile.from_aspect_ratio("16:9")

    req = MediaAssemblyRequest(
        workflow_id=wf_id,
        clips=[
            MediaClip(shot_number=1, source_path=clip1, target_duration_seconds=5.0),
        ],
        profile=profile,
    )

    mock_storage = AsyncMock()
    mock_artifact = ArtifactRef(
        workflow_id=wf_id,
        artifact_type="final_video",
        filename="final_video.mp4",
        mime_type="video/mp4",
        drive_file_id="drive_final_123",
        size_bytes=1024,
    )
    mock_storage.upload.return_value = mock_artifact

    staging = tmp_path / "staging"
    service = MediaAssemblyService(
        media_processor=mock_processor,
        storage=mock_storage,
        staging_dir=staging,
    )

    result = await service.assemble(req)

    assert result.artifact_ref == mock_artifact
    assert mock_storage.upload.await_count == 1
    call_args = mock_storage.upload.call_args[1]
    assert call_args["artifact_type"] == "final_video"
    assert call_args["mime_type"] == "video/mp4"


@pytest.mark.asyncio
async def test_resolve_missing_clip_via_storage(mock_processor, tmp_path):
    missing_local = tmp_path / "missing_shot1.mp4"
    wf_id = uuid4()

    art_ref = ArtifactRef(
        workflow_id=wf_id,
        artifact_type="video_shot",
        filename="shot_1.mp4",
        mime_type="video/mp4",
        drive_file_id="drive_shot_1",
    )

    req = MediaAssemblyRequest(
        workflow_id=wf_id,
        clips=[
            MediaClip(
                shot_number=1,
                source_path=missing_local,
                target_duration_seconds=5.0,
                artifact_ref=art_ref,
            ),
        ],
        profile=MediaProfile.from_aspect_ratio("9:16"),
    )

    mock_storage = AsyncMock()

    async def fake_download(ref, dest):
        dest.write_bytes(b"downloaded_bytes")

    mock_storage.download.side_effect = fake_download
    mock_storage.upload.return_value = art_ref

    staging = tmp_path / "staging"
    service = MediaAssemblyService(
        media_processor=mock_processor,
        storage=mock_storage,
        staging_dir=staging,
    )

    result = await service.assemble(req)
    assert result.workflow_id == wf_id
    assert mock_storage.download.await_count == 1


@pytest.mark.asyncio
async def test_resolve_missing_clip_fails_without_storage(mock_processor, tmp_path):
    missing_local = tmp_path / "missing_shot1.mp4"
    wf_id = uuid4()

    req = MediaAssemblyRequest(
        workflow_id=wf_id,
        clips=[
            MediaClip(
                shot_number=1,
                source_path=missing_local,
                target_duration_seconds=5.0,
            ),
        ],
        profile=MediaProfile.from_aspect_ratio("9:16"),
    )

    service = MediaAssemblyService(
        media_processor=mock_processor,
        staging_dir=tmp_path / "staging",
    )

    with pytest.raises(MediaValidationError, match="Clip file does not exist locally"):
        await service.assemble(req)


@pytest.mark.asyncio
async def test_cleanup_on_processor_failure(mock_processor, tmp_path):
    clip1 = tmp_path / "clip1.mp4"
    clip1.write_bytes(b"clip1")

    wf_id = uuid4()
    req = MediaAssemblyRequest(
        workflow_id=wf_id,
        clips=[MediaClip(shot_number=1, source_path=clip1, target_duration_seconds=5.0)],
        profile=MediaProfile.from_aspect_ratio("9:16"),
    )

    mock_processor.normalize_clip.side_effect = RuntimeError("FFmpeg core dumped")

    staging = tmp_path / "staging"
    service = MediaAssemblyService(
        media_processor=mock_processor,
        staging_dir=staging,
    )

    with pytest.raises(RuntimeError, match="FFmpeg core dumped"):
        await service.assemble(req)

    # Intermediates should be cleaned up even on failure
    work_dir = staging / str(wf_id) / "assembly"
    assert not (work_dir / "normalized").exists()
    assert not (work_dir / "concat").exists()


@pytest.mark.asyncio
async def test_concurrency_limiting_with_semaphore(mock_processor, tmp_path):
    clips = []
    for i in range(1, 5):
        c = tmp_path / f"c{i}.mp4"
        c.write_bytes(b"c")
        clips.append(MediaClip(shot_number=i, source_path=c, target_duration_seconds=5.0))

    active_count = 0
    max_active = 0

    async def fake_normalize(source, output, profile, *args, **kwargs):
        nonlocal active_count, max_active
        active_count += 1
        max_active = max(max_active, active_count)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"normalized")
        await asyncio.sleep(0.02)
        active_count -= 1
        return MediaInfo(
            duration_seconds=5.0, width=1080, height=1920, frame_rate=30.0, has_video=True
        )

    mock_processor.normalize_clip.side_effect = fake_normalize

    wf_id = uuid4()
    req = MediaAssemblyRequest(
        workflow_id=wf_id,
        clips=clips,
        profile=MediaProfile.from_aspect_ratio("9:16"),
    )

    service = MediaAssemblyService(
        media_processor=mock_processor,
        staging_dir=tmp_path / "staging",
        max_concurrency=2,
    )

    await service.assemble(req)

    # Max concurrency should never exceed 2
    assert max_active == 2


def test_media_assembly_service_invalid_concurrency(mock_processor, tmp_path):
    with pytest.raises(MediaConfigurationError, match="max_concurrency must be >= 1"):
        MediaAssemblyService(media_processor=mock_processor, max_concurrency=0)

    with pytest.raises(MediaConfigurationError, match="max_concurrency must be >= 1"):
        MediaAssemblyService(media_processor=mock_processor, max_concurrency=-2)


def test_media_assembly_service_invalid_duration_tolerance(mock_processor, tmp_path):
    with pytest.raises(MediaConfigurationError, match="duration_tolerance_seconds must be >= 0"):
        MediaAssemblyService(media_processor=mock_processor, duration_tolerance_seconds=-0.5)


@pytest.mark.asyncio
async def test_resolve_download_missing_target_fails(mock_processor, tmp_path):
    wf_id = uuid4()
    art_ref = ArtifactRef(
        workflow_id=wf_id,
        artifact_type="video_shot",
        filename="shot_1.mp4",
        mime_type="video/mp4",
        drive_file_id="drive_1",
    )
    req = MediaAssemblyRequest(
        workflow_id=wf_id,
        clips=[
            MediaClip(
                shot_number=1,
                source_path=tmp_path / "missing.mp4",
                target_duration_seconds=5.0,
                artifact_ref=art_ref,
            ),
        ],
        profile=MediaProfile.from_aspect_ratio("9:16"),
    )
    mock_storage = AsyncMock()
    # Fake download succeeds without creating the file
    mock_storage.download = AsyncMock()

    service = MediaAssemblyService(
        media_processor=mock_processor,
        storage=mock_storage,
        staging_dir=tmp_path / "staging",
    )

    with pytest.raises(MediaValidationError, match="does not exist"):
        await service.assemble(req)


@pytest.mark.asyncio
async def test_resolve_download_zero_byte_target_fails(mock_processor, tmp_path):
    wf_id = uuid4()
    art_ref = ArtifactRef(
        workflow_id=wf_id,
        artifact_type="video_shot",
        filename="shot_1.mp4",
        mime_type="video/mp4",
        drive_file_id="drive_1",
    )
    req = MediaAssemblyRequest(
        workflow_id=wf_id,
        clips=[
            MediaClip(
                shot_number=1,
                source_path=tmp_path / "missing.mp4",
                target_duration_seconds=5.0,
                artifact_ref=art_ref,
            ),
        ],
        profile=MediaProfile.from_aspect_ratio("9:16"),
    )
    mock_storage = AsyncMock()

    async def fake_download(ref, dest):
        dest.write_bytes(b"")

    mock_storage.download.side_effect = fake_download

    service = MediaAssemblyService(
        media_processor=mock_processor,
        storage=mock_storage,
        staging_dir=tmp_path / "staging",
    )

    with pytest.raises(MediaValidationError, match="empty \\(0 bytes\\)"):
        await service.assemble(req)


@pytest.mark.asyncio
async def test_resolve_download_corrupt_probe_failure(mock_processor, tmp_path):
    wf_id = uuid4()
    art_ref = ArtifactRef(
        workflow_id=wf_id,
        artifact_type="video_shot",
        filename="shot_1.mp4",
        mime_type="video/mp4",
        drive_file_id="drive_1",
    )
    req = MediaAssemblyRequest(
        workflow_id=wf_id,
        clips=[
            MediaClip(
                shot_number=1,
                source_path=tmp_path / "missing.mp4",
                target_duration_seconds=5.0,
                artifact_ref=art_ref,
            ),
        ],
        profile=MediaProfile.from_aspect_ratio("9:16"),
    )
    mock_storage = AsyncMock()

    async def fake_download(ref, dest):
        dest.write_bytes(b"corrupt_data")

    mock_storage.download.side_effect = fake_download
    mock_processor.probe.side_effect = MediaProbeError("ffprobe: Invalid NAL unit")

    service = MediaAssemblyService(
        media_processor=mock_processor,
        storage=mock_storage,
        staging_dir=tmp_path / "staging",
    )

    with pytest.raises(MediaValidationError, match="failed media probe validation"):
        await service.assemble(req)


@pytest.mark.asyncio
async def test_resolve_download_non_video_fails(mock_processor, tmp_path):
    wf_id = uuid4()
    art_ref = ArtifactRef(
        workflow_id=wf_id,
        artifact_type="video_shot",
        filename="shot_1.mp4",
        mime_type="video/mp4",
        drive_file_id="drive_1",
    )
    req = MediaAssemblyRequest(
        workflow_id=wf_id,
        clips=[
            MediaClip(
                shot_number=1,
                source_path=tmp_path / "missing.mp4",
                target_duration_seconds=5.0,
                artifact_ref=art_ref,
            ),
        ],
        profile=MediaProfile.from_aspect_ratio("9:16"),
    )
    mock_storage = AsyncMock()

    async def fake_download(ref, dest):
        dest.write_bytes(b"audio_bytes")

    mock_storage.download.side_effect = fake_download
    mock_processor.probe.return_value = MediaInfo(
        duration_seconds=5.0,
        width=0,
        height=0,
        frame_rate=0.0,
        has_video=False,
        has_audio=True,
    )

    service = MediaAssemblyService(
        media_processor=mock_processor,
        storage=mock_storage,
        staging_dir=tmp_path / "staging",
    )

    with pytest.raises(MediaValidationError, match="contains no video stream"):
        await service.assemble(req)


@pytest.mark.asyncio
async def test_resolve_local_clip_zero_bytes_fails(mock_processor, tmp_path):
    clip1 = tmp_path / "clip1.mp4"
    clip1.write_bytes(b"")  # zero bytes

    wf_id = uuid4()
    req = MediaAssemblyRequest(
        workflow_id=wf_id,
        clips=[MediaClip(shot_number=1, source_path=clip1, target_duration_seconds=5.0)],
        profile=MediaProfile.from_aspect_ratio("9:16"),
    )

    service = MediaAssemblyService(
        media_processor=mock_processor,
        staging_dir=tmp_path / "staging",
    )

    with pytest.raises(MediaValidationError, match="empty \\(0 bytes\\)"):
        await service.assemble(req)
