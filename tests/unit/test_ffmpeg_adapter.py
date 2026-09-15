import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
from app.mpt.ffmpeg import FFmpegMediaProcessor
from app.mpt.models import AudioTrack, MediaInfo, MediaProfile


@pytest.fixture
def mock_binaries():
    with patch("shutil.which") as mock_which:
        mock_which.side_effect = lambda binary: f"/usr/bin/{binary}"
        yield mock_which


def test_ffmpeg_processor_init_missing_ffmpeg():
    with patch("shutil.which") as mock_which:
        mock_which.side_effect = lambda binary: None if binary == "ffmpeg" else "/usr/bin/ffprobe"
        with pytest.raises(
            MediaConfigurationError, match="FFmpeg executable 'ffmpeg' was not found"
        ):
            FFmpegMediaProcessor(ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe")


def test_ffmpeg_processor_init_missing_ffprobe():
    with patch("shutil.which") as mock_which:
        mock_which.side_effect = lambda binary: "/usr/bin/ffmpeg" if binary == "ffmpeg" else None
        with pytest.raises(
            MediaConfigurationError, match="FFprobe executable 'ffprobe' was not found"
        ):
            FFmpegMediaProcessor(ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe")


@pytest.mark.asyncio
async def test_probe_success(mock_binaries, tmp_path):
    clip = tmp_path / "test.mp4"
    clip.write_bytes(b"dummy")

    fake_ffprobe_json = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1080,
                "height": 1920,
                "r_frame_rate": "30/1",
                "duration": "10.000",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "duration": "10.000",
            },
        ],
        "format": {
            "duration": "10.000",
            "bit_rate": "2500000",
            "size": "3125000",
        },
    }

    processor = FFmpegMediaProcessor()
    with patch.object(processor, "_run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = (json.dumps(fake_ffprobe_json), "")
        info = await processor.probe(clip)

    assert info.duration_seconds == 10.0
    assert info.width == 1080
    assert info.height == 1920
    assert info.frame_rate == 30.0
    assert info.video_codec == "h264"
    assert info.audio_codec == "aac"
    assert info.has_video is True
    assert info.has_audio is True
    assert info.bitrate == 2500000
    assert info.size_bytes == 3125000


@pytest.mark.asyncio
async def test_probe_missing_file(mock_binaries, tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    processor = FFmpegMediaProcessor()
    with pytest.raises(MediaValidationError, match="Source file for probe does not exist"):
        await processor.probe(missing)


@pytest.mark.asyncio
async def test_probe_malformed_json(mock_binaries, tmp_path):
    clip = tmp_path / "test.mp4"
    clip.write_bytes(b"dummy")

    processor = FFmpegMediaProcessor()
    with patch.object(processor, "_run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = ("not valid json", "")
        with pytest.raises(MediaProbeError, match="Failed to parse ffprobe JSON output"):
            await processor.probe(clip)


@pytest.mark.asyncio
async def test_probe_no_streams(mock_binaries, tmp_path):
    clip = tmp_path / "test.mp4"
    clip.write_bytes(b"dummy")

    processor = FFmpegMediaProcessor()
    with patch.object(processor, "_run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = (json.dumps({"streams": [], "format": {}}), "")
        with pytest.raises(MediaProbeError, match="No audio or video streams found"):
            await processor.probe(clip)


@pytest.mark.asyncio
async def test_probe_invalid_or_negative_duration(mock_binaries, tmp_path):
    clip = tmp_path / "test.mp4"
    clip.write_bytes(b"dummy")

    processor = FFmpegMediaProcessor()
    with patch.object(processor, "_run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = (
            json.dumps({"streams": [{"codec_type": "video", "duration": "-5.0"}]}),
            "",
        )
        with pytest.raises(MediaProbeError, match="Negative duration parsed"):
            await processor.probe(clip)


@pytest.mark.asyncio
async def test_run_command_timeout(mock_binaries):
    processor = FFmpegMediaProcessor(timeout_seconds=0.01)

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
    mock_proc.terminate = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(MediaTimeoutError, match="timed out"):
            await processor._run_command(["dummy", "arg"], operation="normalize")
        assert mock_proc.terminate.called


@pytest.mark.asyncio
async def test_run_command_error_mapping(mock_binaries):
    processor = FFmpegMediaProcessor()

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"Unknown codec libx264\nFatal error"))
    mock_proc.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(MediaEncodingError, match="Normalize failed with exit code 1"):
            await processor._run_command(["dummy"], operation="normalize")

        with pytest.raises(MediaConcatenationError, match="Concat failed with exit code 1"):
            await processor._run_command(["dummy"], operation="concat")

        with pytest.raises(MediaAudioMuxError, match="Add_audio failed with exit code 1"):
            await processor._run_command(["dummy"], operation="add_audio")


@pytest.mark.asyncio
async def test_normalize_clip_duration_reconciliation_trim(mock_binaries, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"dummy")
    output = tmp_path / "normalized.mp4"

    processor = FFmpegMediaProcessor()
    profile = MediaProfile.from_aspect_ratio("9:16")

    # Source duration is 10.0s, target is 6.0s (difference > tolerance -> trim)
    probed_source = MagicMock(duration_seconds=10.0, has_video=True)
    probed_output = MagicMock(duration_seconds=6.0, has_video=True)

    with (
        patch.object(processor, "probe") as mock_probe,
        patch.object(processor, "_run_command", new_callable=AsyncMock) as mock_run,
    ):
        mock_probe.side_effect = [probed_source, probed_output]
        mock_run.return_value = ("", "")

        await processor.normalize_clip(
            source=source,
            output=output,
            profile=profile,
            target_duration=6.0,
            tolerance_seconds=0.5,
        )

        args = mock_run.call_args[0][0]
        assert "-t" in args
        assert "6.000" in args
        assert any("scale=1080:1920" in a for a in args)


@pytest.mark.asyncio
async def test_normalize_clip_duration_reconciliation_extend(mock_binaries, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"dummy")
    output = tmp_path / "extended.mp4"

    processor = FFmpegMediaProcessor()
    profile = MediaProfile.from_aspect_ratio("9:16")

    # Source is 10.0s, target is 12.0s (underflow > 0.5s -> freeze-frame extension)
    probed_source = MagicMock(duration_seconds=10.0, has_video=True)
    probed_output = MagicMock(duration_seconds=12.0, has_video=True)

    with (
        patch.object(processor, "probe") as mock_probe,
        patch.object(processor, "_run_command", new_callable=AsyncMock) as mock_run,
    ):
        mock_probe.side_effect = [probed_source, probed_output]
        mock_run.return_value = ("", "")

        await processor.normalize_clip(
            source=source,
            output=output,
            profile=profile,
            target_duration=12.0,
            tolerance_seconds=0.5,
        )

        args = mock_run.call_args[0][0]
        # Should include tpad=stop_mode=clone:stop_duration=2.000
        vf_arg = args[args.index("-vf") + 1]
        assert "tpad=stop_mode=clone:stop_duration=2.000" in vf_arg
        assert "-t" in args
        assert "12.000" in args


@pytest.mark.asyncio
async def test_concatenate_demuxer(mock_binaries, tmp_path):
    clip1 = tmp_path / "clip1.mp4"
    clip2 = tmp_path / "clip2.mp4"
    clip1.write_bytes(b"c1")
    clip2.write_bytes(b"c2")
    output = tmp_path / "concat.mp4"

    processor = FFmpegMediaProcessor()
    probed_output = MagicMock(duration_seconds=15.0)

    with (
        patch.object(processor, "probe", return_value=probed_output),
        patch.object(processor, "_run_command", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = ("", "")
        res = await processor.concatenate([clip1, clip2], output)

        assert res.duration_seconds == 15.0
        args = mock_run.call_args[0][0]
        assert "-f" in args and "concat" in args
        assert "-c" in args and "copy" in args


@pytest.mark.asyncio
async def test_add_audio_looping_and_padding(mock_binaries, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"vid")
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"aud")
    output = tmp_path / "final.mp4"

    processor = FFmpegMediaProcessor()

    # Case 1: Audio is shorter (5s) than video (10s), loop=True
    audio_track_loop = AudioTrack(source_path=audio, volume=0.5, loop=True)
    audio_info = MagicMock(duration_seconds=5.0, has_audio=True)
    final_info = MagicMock(duration_seconds=10.0, has_audio=True)

    with (
        patch.object(processor, "probe") as mock_probe,
        patch.object(processor, "_run_command", new_callable=AsyncMock) as mock_run,
    ):
        mock_probe.side_effect = [audio_info, final_info]
        mock_run.return_value = ("", "")

        await processor.add_audio(video, audio_track_loop, output, video_duration=10.0)

        args = mock_run.call_args[0][0]
        assert "-stream_loop" in args
        assert "-1" in args
        assert "volume=0.5" in args[args.index("-af") + 1]

    # Case 2: Audio is shorter (5s) than video (10s), loop=False -> apad silence
    audio_track_noloop = AudioTrack(source_path=audio, volume=0.7, loop=False)
    with (
        patch.object(processor, "probe") as mock_probe,
        patch.object(processor, "_run_command", new_callable=AsyncMock) as mock_run,
    ):
        mock_probe.side_effect = [audio_info, final_info]
        mock_run.return_value = ("", "")

        await processor.add_audio(video, audio_track_noloop, output, video_duration=10.0)

        args = mock_run.call_args[0][0]
        assert "-stream_loop" not in args
        af_arg = args[args.index("-af") + 1]
        assert "apad" in af_arg
        assert "volume=0.7" in af_arg


@pytest.mark.asyncio
async def test_run_command_cancellation(mock_binaries):
    processor = FFmpegMediaProcessor()
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.CancelledError)
    mock_proc.terminate = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(asyncio.CancelledError):
            await processor._run_command(["dummy"], operation="normalize")
        assert mock_proc.terminate.called


@pytest.mark.asyncio
async def test_terminate_process_fallback_to_kill():
    mock_proc = MagicMock()
    mock_proc.terminate = MagicMock()
    mock_proc.wait = AsyncMock(side_effect=[TimeoutError, 0])
    mock_proc.kill = MagicMock()
    await FFmpegMediaProcessor._terminate_process(mock_proc)
    assert mock_proc.kill.called


@pytest.mark.asyncio
async def test_terminate_process_lookup_error():
    mock_proc = MagicMock()
    mock_proc.terminate = MagicMock(side_effect=ProcessLookupError)
    await FFmpegMediaProcessor._terminate_process(mock_proc)


@pytest.mark.asyncio
async def test_probe_audio_only_and_r_frame_rate_variations(mock_binaries, tmp_path):
    clip = tmp_path / "audio_only.m4a"
    clip.write_bytes(b"dummy")
    processor = FFmpegMediaProcessor()

    fake_json = {
        "streams": [{"codec_type": "audio", "codec_name": "aac", "duration": "4.5"}],
        "format": {},
    }
    with patch.object(processor, "_run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = (json.dumps(fake_json), "")
        info = await processor.probe(clip)
        assert info.duration_seconds == 4.5
        assert info.has_audio is True
        assert info.has_video is False

    fake_json2 = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "duration": "2.0",
                "r_frame_rate": "25",
            }
        ],
        "format": {},
    }
    with patch.object(processor, "_run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = (json.dumps(fake_json2), "")
        info2 = await processor.probe(clip)
        assert info2.frame_rate == 25.0

    # Invalid fraction denominator 0
    fake_json3 = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "duration": "2.0",
                "r_frame_rate": "30/0",
            }
        ],
        "format": {},
    }
    with patch.object(processor, "_run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = (json.dumps(fake_json3), "")
        info3 = await processor.probe(clip)
        assert info3.frame_rate == 0.0


@pytest.mark.asyncio
async def test_probe_unparseable_duration(mock_binaries, tmp_path):
    clip = tmp_path / "corrupt.mp4"
    clip.write_bytes(b"dummy")
    processor = FFmpegMediaProcessor()
    fake_json = {
        "streams": [{"codec_type": "video", "duration": "not-a-number"}],
        "format": {},
    }
    with patch.object(processor, "_run_command", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = (json.dumps(fake_json), "")
        with pytest.raises(MediaProbeError, match="Invalid duration"):
            await processor.probe(clip)


@pytest.mark.asyncio
async def test_normalize_clip_validation_errors(mock_binaries, tmp_path):
    missing = tmp_path / "missing.mp4"
    processor = FFmpegMediaProcessor()
    profile = MediaProfile.from_aspect_ratio("9:16")

    with pytest.raises(MediaValidationError, match="not found"):
        await processor.normalize_clip(missing, tmp_path / "out.mp4", profile)

    audio_only = tmp_path / "audio.m4a"
    audio_only.write_bytes(b"audio")
    audio_info = MediaInfo(
        duration_seconds=5.0,
        width=0,
        height=0,
        frame_rate=0.0,
        has_video=False,
        has_audio=True,
    )
    with (
        patch.object(processor, "probe", return_value=audio_info),
        pytest.raises(MediaValidationError, match="no video stream"),
    ):
        await processor.normalize_clip(audio_only, tmp_path / "out.mp4", profile)


@pytest.mark.asyncio
async def test_normalize_clip_no_target_duration(mock_binaries, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"dummy")
    processor = FFmpegMediaProcessor()
    profile = MediaProfile.from_aspect_ratio("9:16")
    vid_info = MediaInfo(
        duration_seconds=5.0,
        width=1080,
        height=1920,
        frame_rate=30.0,
        has_video=True,
    )
    with (
        patch.object(processor, "probe", return_value=vid_info),
        patch.object(processor, "_run_command", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = ("", "")
        await processor.normalize_clip(clip, tmp_path / "out.mp4", profile, target_duration=None)
        args = mock_run.call_args[0][0]
        assert "-t" not in args


@pytest.mark.asyncio
async def test_concatenate_validation_errors(mock_binaries, tmp_path):
    processor = FFmpegMediaProcessor()
    with pytest.raises(MediaValidationError, match="Cannot concatenate empty list"):
        await processor.concatenate([], tmp_path / "out.mp4")

    missing = tmp_path / "missing.mp4"
    with pytest.raises(MediaValidationError, match="Clip file not found for concat"):
        await processor.concatenate([missing], tmp_path / "out.mp4")


@pytest.mark.asyncio
async def test_add_audio_validation_errors(mock_binaries, tmp_path):
    processor = FFmpegMediaProcessor()
    missing_vid = tmp_path / "missing_vid.mp4"
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"aud")
    track = AudioTrack(source_path=audio)

    with pytest.raises(MediaValidationError, match="Video file not found"):
        await processor.add_audio(missing_vid, track, tmp_path / "out.mp4", 5.0)

    vid = tmp_path / "vid.mp4"
    vid.write_bytes(b"vid")
    missing_audio_track = AudioTrack(source_path=tmp_path / "missing_aud.m4a")
    with pytest.raises(MediaValidationError, match="Audio file not found"):
        await processor.add_audio(vid, missing_audio_track, tmp_path / "out.mp4", 5.0)

    no_audio_info = MediaInfo(
        duration_seconds=5.0,
        width=1080,
        height=1920,
        frame_rate=30.0,
        has_video=True,
        has_audio=False,
    )
    with (
        patch.object(processor, "probe", return_value=no_audio_info),
        pytest.raises(MediaValidationError, match="no audio stream"),
    ):
        await processor.add_audio(vid, track, tmp_path / "out.mp4", 5.0)


@pytest.mark.asyncio
async def test_run_command_probe_and_unknown_failure(mock_binaries):
    processor = FFmpegMediaProcessor()
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"ffprobe fatal error"))
    mock_proc.returncode = 1
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(MediaProbeError, match="Probe failed with exit code 1"):
            await processor._run_command(["ffprobe"], operation="probe")

        with pytest.raises(MediaProcessingError, match="Custom_op failed with exit code 1"):
            await processor._run_command(["custom"], operation="custom_op")
