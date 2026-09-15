import shutil
import subprocess
from uuid import uuid4

import pytest

from app.mpt.ffmpeg import FFmpegMediaProcessor
from app.mpt.models import (
    AudioTrack,
    MediaAssemblyRequest,
    MediaClip,
    MediaProfile,
)
from app.mpt.service import MediaAssemblyService


def has_ffmpeg_binaries() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


pytestmark = pytest.mark.skipif(
    not has_ffmpeg_binaries(), reason="FFmpeg or FFprobe binary not found"
)


@pytest.fixture
def synthetic_media(tmp_path):
    """Generate lightweight synthetic video and audio fixtures using FFmpeg lavfi filters."""
    media_dir = tmp_path / "raw_media"
    media_dir.mkdir(parents=True, exist_ok=True)

    clip1_path = media_dir / "clip1.mp4"
    clip2_path = media_dir / "clip2.mp4"
    audio_path = media_dir / "audio.m4a"

    # Clip 1: 3 seconds, 320x240, 30fps
    cmd1 = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=3:size=320x240:rate=30",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(clip1_path),
    ]
    subprocess.run(cmd1, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Clip 2: 5 seconds, 640x480, 25fps (different dimensions and frame rate to test normalization)
    cmd2 = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=5:size=640x480:rate=25",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(clip2_path),
    ]
    subprocess.run(cmd2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Audio: 4 seconds 440Hz sine wave tone
    cmd_audio = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=4",
        "-c:a",
        "aac",
        str(audio_path),
    ]
    subprocess.run(cmd_audio, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return clip1_path, clip2_path, audio_path


@pytest.mark.asyncio
async def test_real_ffmpeg_probe(synthetic_media):
    clip1, _, audio = synthetic_media
    processor = FFmpegMediaProcessor()

    video_info = await processor.probe(clip1)
    assert video_info.has_video is True
    assert video_info.width == 320
    assert video_info.height == 240
    assert abs(video_info.duration_seconds - 3.0) < 0.2
    assert abs(video_info.frame_rate - 30.0) < 0.5

    audio_info = await processor.probe(audio)
    assert audio_info.has_audio is True
    assert abs(audio_info.duration_seconds - 4.0) < 0.2


@pytest.mark.asyncio
async def test_real_ffmpeg_normalization_trim_and_extend(synthetic_media, tmp_path):
    clip1, clip2, _ = synthetic_media
    processor = FFmpegMediaProcessor()
    profile = MediaProfile.from_aspect_ratio("9:16", fps=30)

    # 1. Trimming: Clip1 is 3.0s, target is 2.0s -> trimmed to 2.0s
    norm1 = tmp_path / "norm1.mp4"
    info1 = await processor.normalize_clip(
        clip1,
        norm1,
        profile,
        target_duration=2.0,
        tolerance_seconds=0.2,
    )
    assert abs(info1.duration_seconds - 2.0) < 0.2
    assert info1.width == 1080
    assert info1.height == 1920
    assert abs(info1.frame_rate - 30.0) < 0.5
    assert not info1.has_audio

    # 2. Extending: Clip2 is 5.0s, target is 7.0s -> freeze-frame extension to 7.0s
    norm2 = tmp_path / "norm2.mp4"
    info2 = await processor.normalize_clip(
        clip2,
        norm2,
        profile,
        target_duration=7.0,
        tolerance_seconds=0.2,
    )
    assert abs(info2.duration_seconds - 7.0) < 0.2
    assert info2.width == 1080
    assert info2.height == 1920
    assert abs(info2.frame_rate - 30.0) < 0.5
    assert not info2.has_audio


@pytest.mark.asyncio
async def test_real_ffmpeg_concatenation_and_audio(synthetic_media, tmp_path):
    clip1, clip2, audio = synthetic_media
    processor = FFmpegMediaProcessor()
    profile = MediaProfile.from_aspect_ratio("9:16", fps=30)

    # Normalize both to identical profile
    norm1 = tmp_path / "norm1.mp4"
    norm2 = tmp_path / "norm2.mp4"
    await processor.normalize_clip(clip1, norm1, profile, target_duration=2.0)
    await processor.normalize_clip(clip2, norm2, profile, target_duration=3.0)

    # Concat
    concat_out = tmp_path / "concat.mp4"
    concat_info = await processor.concatenate([norm1, norm2], concat_out)
    assert concat_out.exists()
    assert abs(concat_info.duration_seconds - 5.0) < 0.3
    assert concat_info.width == 1080
    assert concat_info.height == 1920

    # Add looped audio (audio is 4s, video is 5s)
    final_looped = tmp_path / "final_looped.mp4"
    track_loop = AudioTrack(source_path=audio, volume=0.9, loop=True)
    looped_info = await processor.add_audio(
        concat_out, track_loop, final_looped, video_duration=concat_info.duration_seconds
    )

    assert final_looped.exists()
    assert looped_info.has_video is True
    assert looped_info.has_audio is True
    assert abs(looped_info.duration_seconds - 5.0) < 0.3


@pytest.mark.asyncio
async def test_real_media_assembly_service_end_to_end(synthetic_media, tmp_path):
    clip1, clip2, audio = synthetic_media
    processor = FFmpegMediaProcessor()
    staging = tmp_path / "staging"

    service = MediaAssemblyService(
        media_processor=processor,
        staging_dir=staging,
        max_concurrency=2,
    )

    wf_id = uuid4()
    req = MediaAssemblyRequest(
        workflow_id=wf_id,
        clips=[
            # Input out of order to verify sorting: 2 then 1
            MediaClip(shot_number=1, source_path=clip1, target_duration_seconds=2.0),
            MediaClip(shot_number=2, source_path=clip2, target_duration_seconds=4.0),
        ],
        profile=MediaProfile.from_aspect_ratio("9:16", fps=30),
        audio_track=AudioTrack(source_path=audio, volume=0.7, loop=True),
    )

    result = await service.assemble(req)

    assert result.workflow_id == wf_id
    assert result.output_path.exists()
    assert abs(result.duration_seconds - 6.0) < 0.4
    assert result.media_info.has_video is True
    assert result.media_info.has_audio is True
    assert result.media_info.width == 1080
    assert result.media_info.height == 1920

    # Verify intermediate cleanup
    work_dir = staging / str(wf_id) / "assembly"
    assert not (work_dir / "normalized").exists()
    assert not (work_dir / "concat").exists()
    assert (work_dir / "final" / "final_video.mp4").exists()
