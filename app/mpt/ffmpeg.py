from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.logging import get_logger
from app.mpt.adapter import MediaProcessor
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
from app.mpt.models import AudioTrack, MediaInfo, MediaProfile

logger = get_logger(__name__)


class FFmpegMediaProcessor(MediaProcessor):
    """Concrete FFmpeg and FFprobe implementation of MediaProcessor."""

    def __init__(
        self,
        *,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
        timeout_seconds: float = 300.0,
    ) -> None:
        if timeout_seconds <= 0.0:
            raise MediaConfigurationError("timeout_seconds must be > 0", operation="init")

        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary
        self.timeout_seconds = timeout_seconds

        # Fast discovery: Verify binaries exist on PATH or filesystem
        self._ffmpeg_path = shutil.which(self.ffmpeg_binary)
        if not self._ffmpeg_path:
            raise MediaConfigurationError(
                f"FFmpeg executable '{self.ffmpeg_binary}' was not found on PATH.",
                operation="init",
            )

        self._ffprobe_path = shutil.which(self.ffprobe_binary)
        if not self._ffprobe_path:
            raise MediaConfigurationError(
                f"FFprobe executable '{self.ffprobe_binary}' was not found on PATH.",
                operation="init",
            )

    async def _run_command(self, args: list[str], *, operation: str) -> tuple[str, str]:
        """Execute subprocess asynchronously with argument arrays, timeout, and cancellation reaping."""
        logger.debug("mpt.ffmpeg.exec", operation=operation, command=args[0], num_args=len(args))

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            await self._terminate_process(proc)
            raise MediaTimeoutError(
                f"Operation '{operation}' timed out after {self.timeout_seconds}s.",
                operation=operation,
            ) from None
        except asyncio.CancelledError:
            await self._terminate_process(proc)
            raise

        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")

        if proc.returncode != 0:
            err_snippet = "\n".join(stderr.strip().splitlines()[-5:])
            msg = f"{operation.capitalize()} failed with exit code {proc.returncode}: {err_snippet}"
            logger.error(
                "mpt.ffmpeg.failed",
                operation=operation,
                returncode=proc.returncode,
                stderr=err_snippet,
            )

            if operation == "probe":
                raise MediaProbeError(msg, operation=operation, context={"stderr": err_snippet})
            if operation == "normalize":
                raise MediaEncodingError(msg, operation=operation, context={"stderr": err_snippet})
            if operation == "concat":
                raise MediaConcatenationError(
                    msg, operation=operation, context={"stderr": err_snippet}
                )
            if operation == "add_audio":
                raise MediaAudioMuxError(msg, operation=operation, context={"stderr": err_snippet})
            raise MediaProcessingError(msg, operation=operation, context={"stderr": err_snippet})

        return stdout, stderr

    @staticmethod
    async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
        """Safely terminate and reap subprocess across platforms (including Windows)."""
        try:
            proc.terminate()
        except ProcessLookupError:
            return

        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass

    async def probe(self, source: Path) -> MediaInfo:
        """Probe technical metadata for a local media file."""
        if not source.exists() or not source.is_file():
            raise MediaValidationError(
                f"Source file for probe does not exist: {source}", operation="probe"
            )

        args = [
            self.ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,bit_rate:stream=codec_type,codec_name,width,height,r_frame_rate,duration",
            "-of",
            "json",
            str(source),
        ]

        stdout, _ = await self._run_command(args, operation="probe")

        try:
            data: dict[str, Any] = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise MediaProbeError(
                f"Failed to parse ffprobe JSON output: {e}",
                operation="probe",
                context={"raw_output": stdout[:500]},
            ) from e

        streams: list[dict[str, Any]] = data.get("streams", [])
        fmt: dict[str, Any] = data.get("format", {})

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        if not video_stream and not audio_stream:
            raise MediaProbeError(
                f"No audio or video streams found in file: {source}", operation="probe"
            )

        # Parse duration from format or fallback to stream
        raw_duration = fmt.get("duration")
        if raw_duration is None and video_stream:
            raw_duration = video_stream.get("duration")
        if raw_duration is None and audio_stream:
            raw_duration = audio_stream.get("duration")

        try:
            duration_seconds = float(raw_duration) if raw_duration is not None else 0.0
        except (ValueError, TypeError) as e:
            raise MediaProbeError(
                f"Invalid duration parsed from {source}: {raw_duration}",
                operation="probe",
            ) from e

        if duration_seconds < 0:
            raise MediaProbeError(
                f"Negative duration parsed: {duration_seconds}", operation="probe"
            )

        # Video parameters
        width = int(video_stream.get("width", 0)) if video_stream else 0
        height = int(video_stream.get("height", 0)) if video_stream else 0
        video_codec = video_stream.get("codec_name") if video_stream else None
        audio_codec = audio_stream.get("codec_name") if audio_stream else None

        frame_rate = 0.0
        if video_stream and "r_frame_rate" in video_stream:
            r_fr = str(video_stream["r_frame_rate"])
            if "/" in r_fr:
                num, den = r_fr.split("/", 1)
                try:
                    num_f, den_f = float(num), float(den)
                    frame_rate = num_f / den_f if den_f != 0 else 0.0
                except ValueError:
                    frame_rate = 0.0
            else:
                try:
                    frame_rate = float(r_fr)
                except ValueError:
                    frame_rate = 0.0

        bitrate = int(fmt.get("bit_rate")) if fmt.get("bit_rate") else None
        size_bytes = int(fmt.get("size")) if fmt.get("size") else None

        return MediaInfo(
            duration_seconds=duration_seconds,
            width=width,
            height=height,
            frame_rate=frame_rate,
            video_codec=video_codec,
            audio_codec=audio_codec,
            has_video=video_stream is not None,
            has_audio=audio_stream is not None,
            bitrate=bitrate,
            size_bytes=size_bytes,
        )

    async def normalize_clip(
        self,
        source: Path,
        output: Path,
        profile: MediaProfile,
        target_duration: float | None = None,
        tolerance_seconds: float = 0.5,
    ) -> MediaInfo:
        """Normalize a video clip: scale, pad, set fps/pixel format, strip audio, and reconcile duration."""
        if not source.exists() or not source.is_file():
            raise MediaValidationError(
                f"Source file for normalization not found: {source}", operation="normalize"
            )

        source_info = await self.probe(source)
        if not source_info.has_video:
            raise MediaValidationError(
                f"Source file has no video stream: {source}", operation="normalize"
            )

        actual_duration = source_info.duration_seconds
        output.parent.mkdir(parents=True, exist_ok=True)

        # Build scale, pad, fps, format filter
        vf_parts = [
            f"scale={profile.width}:{profile.height}:force_original_aspect_ratio=decrease",
            f"pad={profile.width}:{profile.height}:(ow-iw)/2:(oh-ih)/2",
            f"fps={profile.frame_rate}",
            f"format={profile.pixel_format}",
        ]

        apply_trim = False
        if target_duration is not None and target_duration > 0:
            if actual_duration > target_duration + tolerance_seconds:
                # 1. Actual > Target: trim during encoding
                apply_trim = True
            elif actual_duration < target_duration - tolerance_seconds:
                # 3. Actual < Target: freeze final frame to target duration
                diff = target_duration - actual_duration
                vf_parts.append(f"tpad=stop_mode=clone:stop_duration={diff:.3f}")
                apply_trim = True
            # else: abs(actual - target) <= tolerance -> retain actual

        args = [
            self.ffmpeg_binary,
            "-y",
            "-i",
            str(source),
            "-vf",
            ",".join(vf_parts),
            "-c:v",
            profile.video_codec,
            "-preset",
            "fast",
            "-crf",
            "20",
            "-an",  # Strip audio so intermediate normalized clips are purely video
        ]

        if apply_trim and target_duration is not None:
            args.extend(["-t", f"{target_duration:.3f}"])

        args.append(str(output))

        await self._run_command(args, operation="normalize")
        return await self.probe(output)

    async def concatenate(
        self,
        clips: Sequence[Path],
        output: Path,
    ) -> MediaInfo:
        """Concatenate pre-normalized clips using the FFmpeg concat demuxer."""
        if not clips:
            raise MediaValidationError(
                "Cannot concatenate empty list of clips.", operation="concat"
            )

        for clip in clips:
            if not clip.exists() or not clip.is_file():
                raise MediaValidationError(
                    f"Clip file not found for concat: {clip}", operation="concat"
                )

        output.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = output.parent / f"concat_manifest_{uuid4().hex[:8]}.txt"

        manifest_lines: list[str] = []
        for clip in clips:
            posix_path = clip.resolve().as_posix().replace("'", "'\\''")
            manifest_lines.append(f"file '{posix_path}'")

        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

        args = [
            self.ffmpeg_binary,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest_path),
            "-c",
            "copy",
            str(output),
        ]

        try:
            await self._run_command(args, operation="concat")
        finally:
            if manifest_path.exists():
                manifest_path.unlink(missing_ok=True)

        return await self.probe(output)

    async def add_audio(
        self,
        video: Path,
        audio: AudioTrack,
        output: Path,
        video_duration: float,
    ) -> MediaInfo:
        """Mux an audio track with a video, applying looping or silence padding to match video duration."""
        if not video.exists() or not video.is_file():
            raise MediaValidationError(
                f"Video file not found for audio mux: {video}", operation="add_audio"
            )
        if not audio.source_path.exists() or not audio.source_path.is_file():
            raise MediaValidationError(
                f"Audio file not found for audio mux: {audio.source_path}", operation="add_audio"
            )

        audio_info = await self.probe(audio.source_path)
        if not audio_info.has_audio:
            raise MediaValidationError(
                f"Source audio file has no audio stream: {audio.source_path}", operation="add_audio"
            )

        output.parent.mkdir(parents=True, exist_ok=True)

        args = [self.ffmpeg_binary, "-y"]
        args.extend(["-i", str(video)])

        # Loop if audio is shorter and loop is requested
        if audio.loop and audio_info.duration_seconds < video_duration:
            args.extend(["-stream_loop", "-1", "-i", str(audio.source_path)])
        else:
            args.extend(["-i", str(audio.source_path)])

        # Build audio filter chain
        af_filters: list[str] = [f"volume={audio.volume}"]
        if not audio.loop and audio_info.duration_seconds < video_duration:
            af_filters.append("apad")

        args.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-af",
                ",".join(af_filters),
                "-t",
                f"{video_duration:.3f}",
                str(output),
            ]
        )

        await self._run_command(args, operation="add_audio")
        return await self.probe(output)
