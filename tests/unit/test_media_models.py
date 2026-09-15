from pathlib import Path
from uuid import uuid4

import pytest

from app.artifacts.models import ArtifactRef
from app.mpt.errors import (
    MediaAudioMuxError,
    MediaConcatenationError,
    MediaConfigurationError,
    MediaEncodingError,
    MediaProbeError,
    MediaProcessingError,
    MediaTimeoutError,
    MediaValidationError,
)
from app.mpt.models import (
    AudioTrack,
    MediaAssemblyRequest,
    MediaAssemblyResult,
    MediaClip,
    MediaInfo,
    MediaProfile,
)


def test_media_info_model():
    info = MediaInfo(
        duration_seconds=10.5,
        width=1080,
        height=1920,
        frame_rate=30.0,
        video_codec="h264",
        audio_codec="aac",
        has_video=True,
        has_audio=True,
        bitrate=2500000,
        size_bytes=3200000,
    )
    assert info.duration_seconds == 10.5
    assert info.width == 1080
    assert info.height == 1920
    assert info.frame_rate == 30.0
    assert info.video_codec == "h264"
    assert info.audio_codec == "aac"
    assert info.has_video is True
    assert info.has_audio is True
    assert info.bitrate == 2500000
    assert info.size_bytes == 3200000


def test_media_profile_aspect_ratios():
    p_916 = MediaProfile.from_aspect_ratio("9:16", fps=60)
    assert p_916.width == 1080
    assert p_916.height == 1920
    assert p_916.frame_rate == 60

    p_169 = MediaProfile.from_aspect_ratio("16:9")
    assert p_169.width == 1920
    assert p_169.height == 1080

    p_11 = MediaProfile.from_aspect_ratio("1:1")
    assert p_11.width == 1080
    assert p_11.height == 1080

    p_45 = MediaProfile.from_aspect_ratio("4:5")
    assert p_45.width == 1080
    assert p_45.height == 1350

    p_23 = MediaProfile.from_aspect_ratio("2:3")
    assert p_23.width == 1080
    assert p_23.height == 1620

    with pytest.raises(ValueError, match="Unsupported aspect ratio"):
        MediaProfile.from_aspect_ratio("invalid_ratio")


def test_media_clip_model():
    path = Path("tmp/staging/shot_1.mp4")
    art = ArtifactRef(
        workflow_id=uuid4(),
        artifact_type="video_shot",
        filename="shot_1.mp4",
        mime_type="video/mp4",
        drive_file_id="drive_123",
    )
    clip = MediaClip(
        shot_number=1,
        source_path=path,
        target_duration_seconds=5.0,
        actual_duration_seconds=5.2,
        artifact_ref=art,
    )
    assert clip.shot_number == 1
    assert clip.source_path == path
    assert clip.target_duration_seconds == 5.0
    assert clip.actual_duration_seconds == 5.2
    assert clip.artifact_ref == art

    # Validation errors
    with pytest.raises(ValueError):
        MediaClip(shot_number=0, source_path=path, target_duration_seconds=5.0)
    with pytest.raises(ValueError):
        MediaClip(shot_number=1, source_path=path, target_duration_seconds=-1.0)


def test_audio_track_model():
    audio_path = Path("tmp/music.mp3")
    track = AudioTrack(
        source_path=audio_path,
        volume=0.8,
        start_offset_seconds=1.5,
        loop=True,
    )
    assert track.source_path == audio_path
    assert track.volume == 0.8
    assert track.start_offset_seconds == 1.5
    assert track.loop is True

    # Invalid volume
    with pytest.raises(ValueError):
        AudioTrack(source_path=audio_path, volume=2.5)
    with pytest.raises(ValueError):
        AudioTrack(source_path=audio_path, volume=-0.1)


def test_media_assembly_request_validation():
    wf_id = uuid4()
    profile = MediaProfile.from_aspect_ratio("9:16")

    clip1 = MediaClip(shot_number=1, source_path=Path("s1.mp4"), target_duration_seconds=5.0)
    clip2 = MediaClip(shot_number=2, source_path=Path("s2.mp4"), target_duration_seconds=10.0)

    # Valid out-of-order clips automatically sorted
    req = MediaAssemblyRequest(
        workflow_id=wf_id,
        clips=[clip2, clip1],
        profile=profile,
    )
    assert [c.shot_number for c in req.clips] == [1, 2]

    # Duplicate shot numbers
    with pytest.raises(ValueError, match="Duplicate shot numbers"):
        MediaAssemblyRequest(
            workflow_id=wf_id,
            clips=[clip1, clip1],
            profile=profile,
        )

    # Non-continuous shot sequence (e.g. 1, 3)
    clip3 = MediaClip(shot_number=3, source_path=Path("s3.mp4"), target_duration_seconds=5.0)
    with pytest.raises(ValueError, match="continuous 1-indexed sequence"):
        MediaAssemblyRequest(
            workflow_id=wf_id,
            clips=[clip1, clip3],
            profile=profile,
        )

    # Sequence not starting at 1 (e.g. 2, 3)
    with pytest.raises(ValueError, match="continuous 1-indexed sequence"):
        MediaAssemblyRequest(
            workflow_id=wf_id,
            clips=[clip2, clip3],
            profile=profile,
        )


def test_media_assembly_result_model():
    wf_id = uuid4()
    out_path = Path("tmp/final.mp4")
    info = MediaInfo(duration_seconds=15.0, width=1080, height=1920, frame_rate=30.0)
    res = MediaAssemblyResult(
        workflow_id=wf_id,
        output_path=out_path,
        duration_seconds=15.0,
        media_info=info,
    )
    assert res.workflow_id == wf_id
    assert res.output_path == out_path
    assert res.duration_seconds == 15.0
    assert res.media_info == info
    assert res.artifact_ref is None


def test_media_errors_hierarchy():
    wf_id = uuid4()
    base_err = MediaProcessingError(
        "Base error",
        workflow_id=wf_id,
        operation="test_op",
        context={"extra": 123},
    )
    assert str(base_err) == "Base error"
    assert base_err.workflow_id == wf_id
    assert base_err.operation == "test_op"
    assert base_err.context == {"extra": 123}

    assert issubclass(MediaConfigurationError, MediaProcessingError)
    assert issubclass(MediaProbeError, MediaProcessingError)
    assert issubclass(MediaValidationError, MediaProcessingError)
    assert issubclass(MediaEncodingError, MediaProcessingError)
    assert issubclass(MediaConcatenationError, MediaProcessingError)
    assert issubclass(MediaAudioMuxError, MediaProcessingError)
    assert issubclass(MediaTimeoutError, MediaProcessingError)
