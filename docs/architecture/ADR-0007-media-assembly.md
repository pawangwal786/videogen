# ADR-0007: Media Assembly, Normalization, and Duration Reconciliation

## Status

Accepted

## Context

In Phase 5, VideoGen established the video generation layer, outputting individual shot video clips (`VideoJobRecord`, staged MP4s, and `ArtifactRef`). Because providers such as Google Veo only support fixed discrete durations (5s or 10s), generated clips often differ from the creative timing targets defined in the `StoryboardResult`. Furthermore, individual generated clips can vary in resolution, pixel format, time base, and frame rate.

Phase 6 requires an assembly layer that takes discrete video shot artifacts, normalizes them, reconciles clip durations against storyboard targets, concatenates them in deterministic order, optionally muxes an audio track, stages the final video, and uploads it to cloud storage (`ArtifactStorage`).

## Decisions

### 1. Dedicated MediaProcessor Protocol (Infrastructure Boundary)
- We define a provider- and tool-neutral protocol in `app.mpt.adapter.MediaProcessor`:
  ```python
  class MediaProcessor(Protocol):
      async def probe(self, source: Path) -> MediaInfo: ...
      async def normalize_clip(
          self,
          source: Path,
          output: Path,
          profile: MediaProfile,
          target_duration: float | None = None,
          tolerance_seconds: float = 0.5,
      ) -> MediaInfo: ...
      async def concatenate(self, clips: Sequence[Path], output: Path) -> MediaInfo: ...
      async def add_audio(
          self, video: Path, audio: AudioTrack, output: Path, video_duration: float
      ) -> MediaInfo: ...
  ```
- **Boundary Invariant**: At the `MediaProcessor` boundary, every source media file must exist locally and be readable. `MediaProcessor` accepts only local filesystem paths. `MediaAssemblyService` is responsible for resolving `ArtifactRef` and validating the resulting local media (verifying existence, regular file, positive file size, and probeable video stream) before processor invocation. `MediaProcessor` never interacts with cloud storage or resolves remote URIs.
- Application services (`MediaAssemblyService`, `VideoGenerationService`) do not construct FFmpeg command lines.

### 2. Concrete FFmpeg Media Processor (`FFmpegMediaProcessor`)
- Implemented in `app.mpt.ffmpeg.FFmpegMediaProcessor`.
- **Discovery**: Discovers `ffmpeg` and `ffprobe` binaries via `shutil.which`. If missing, raises `MediaConfigurationError` at initialization/runtime without coupling configuration classes to domain error hierarchies.
- **Subprocess Security**:
  - Executes commands strictly using asynchronous subprocess argument lists (`asyncio.create_subprocess_exec`).
  - Never uses `shell=True` or shell string formatting.
  - Enforces explicit timeouts (`media_assembly_timeout_seconds`).
  - **Cross-Platform / Windows Process Safety**: Subprocess cancellation explicitly reaps and terminates child processes (`proc.terminate()`, wait, `proc.kill()` fallback, `await proc.wait()`), preventing orphaned FFmpeg processes on Windows.

### 3. Canonical Media Profile & Normalization Invariant
- Before concatenation, all clips are normalized to a uniform profile derived from storyboard aspect ratio (`MediaProfile.from_aspect_ratio`):
  - Supported aspect ratios: `9:16` (1080×1920), `16:9` (1920×1080), `1:1` (1080×1080), `4:5` (1080×1350), and `2:3` (1080×1620).
  - Video codec: `libx264`, constant frame rate (30 fps), pixel format: `yuv420p`.
  - Filter: `scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p`.
  - Audio: Stripped during intermediate normalization (`-an`) so all clips are strictly video-only.
- **Concat Demuxer Invariant**: Because all normalized clips share identical codec, resolution, pixel format, and frame rate, concatenation uses the fast FFmpeg concat demuxer (`-f concat -safe 0 -i manifest -c copy`). This avoids generational re-encoding loss during concatenation.

### 4. Three-Way Duration Reconciliation Policy
- In Phase 5, Veo produces discrete 5s or 10s clips. Phase 6 reconciles actual probed clip duration ($T_{\text{actual}}$) against the creative target ($T_{\text{target}}$) using a configurable tolerance ($\Delta = 0.5\text{s}$):
  1. **Trim ($T_{\text{actual}} > T_{\text{target}} + \Delta$)**:
     Clip is trimmed during encoding via `-t {T_target}`.
  2. **Retain ($|T_{\text{actual}} - T_{\text{target}}| \le \Delta$)**:
     Clip duration is retained without manipulation.
  3. **Extend ($T_{\text{actual}} < T_{\text{target}} - \Delta$)**:
     Clip is deterministically extended by cloning/freezing the final frame (`tpad=stop_mode=clone:stop_duration={T_target - T_actual}` followed by `-t {T_target}`). This allows 10s Veo clips to satisfy 12s+ storyboard shots seamlessly without jarring temporal loops.
- **Authoritative Probing**: Post-reconciliation and final assembly durations are authoritatively determined by probing with `ffprobe`, never assumed by arithmetic summation.

### 5. Deterministic Audio Integration Semantics
- When an `AudioTrack` is provided:
  - **Looping (`loop=True`)**: If audio duration < video duration, the track loops continuously (`-stream_loop -1`) and is trimmed to the exact video duration.
  - **Play-Once with Silence (`loop=False`)**: If audio duration < video duration, the track plays once and the remaining video plays in silence (padded via `apad` filter).
  - **Longer Audio**: Audio is trimmed to the exact final video duration.
  - Volume is scaled via `-filter:a "volume={track.volume}"`.
  - Final output contains exactly one audio stream: AAC 44.1kHz stereo (`-c:a aac -b:a 192k -ar 44100 -ac 2`).
- **Audio Timeline Scope**: Audio timeline alignment, seeking, and multi-track placement are deferred to Phase 7, where voiceover/music synchronization requirements will be specified and implemented as part of the audio timeline contract. In Phase 6, `AudioTrack` focuses strictly on full-video background audio integration (volume, looping, and silence padding).

### 6. Media Assembly Service & Workspace Isolation
- `MediaAssemblyService` in `app.mpt.service.py`:
  - **Artifact Resolution**: If a clip references an `ArtifactRef` and local file is absent, downloads the artifact from `ArtifactStorage` into local staging.
  - **Shot Ordering**: Validates sequential continuity ($1, 2, \dots, N$) and sorts strictly by `shot_number`.
  - **Isolated Staging**: Each assembly operation works within `staging/{workflow_id}/assembly/{normalized,concat,final}/`.
  - **Guaranteed Cleanup**: Temporary normalized clips and manifests are cleaned up in `finally:` blocks. On failure, best-effort cleanup occurs while preserving the original domain error.
  - **Concurrency Control**: Bounded via `asyncio.Semaphore(max_concurrency)`.
  - **Storage Persistence**: Uploads final video MP4 to `ArtifactStorage` as `artifact_type="final_video"`, returning authoritative `ArtifactRef`.

### 7. Removal of MPT Coverage Exemption
- With the implementation of `app/mpt/`, the Phase 5 deferred coverage exemption for `app/mpt/adapter.py` is removed.
- All production modules under `app/mpt/` (`adapter.py`, `models.py`, `errors.py`, `ffmpeg.py`, `service.py`) must meet the hard $\ge 90.0\%$ statement coverage gate.
