# Phase 9.4: Real Capability Adapters & End-to-End Pipeline Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement real external capability adapters (Google Veo for video generation, Edge TTS / ElevenLabs for speech synthesis, Google Drive for export storage) conforming to the typed capability protocols in `app/stages/providers.py`, and incrementally integrate the remaining downstream stage handlers (`STORYBOARD`, `MEDIA_SYNTHESIS`, `AUDIO`, `ASSEMBLY`, `QC`, `PUBLISH`) with durable lease-aware recovery and strict cryptographic artifact handoff.

**Architecture & Non-Negotiable Contracts:**
- **Zero Raw Dictionaries:** All stage inputs, outputs, and provider contracts must use strongly typed Pydantic models with zero `Any`.
- **Capability Isolation:** Handlers interact exclusively with `app.stages.providers` protocols (`TextGenerationProvider`, `VideoGenerationProvider`, `TextToSpeechProvider`, `StorageProvider`). Vendor SDK details remain strictly encapsulated in adapter implementations.
- **Durable Polling & Recovery:** Veo long-running operations must persist external operation IDs in durable job state. On worker restart or attempt retry, `reconcile()` or `poll()` resumes the existing operation to prevent duplicate billable submissions.
- **Content-Addressed Artifact Contract:** Inter-stage communication flows exclusively through verified immutable files in `ArtifactStore` recorded in PostgreSQL.

**Tech Stack:** Python 3.12, Google GenAI SDK (`google-genai` for Gemini & Veo), Edge TTS / ElevenLabs, Google Drive API v3, FFmpeg / ffprobe, PostgreSQL 16 (asyncpg), SQLAlchemy 2.0 (async), Pydantic v2.

**Spec Reference:** [phase-9-pipeline-contract.md](file:///e:/ML%20Projects/videogen/docs/architecture/phase-9-pipeline-contract.md)
**ADR References:** [ADR-0002](file:///e:/ML%20Projects/videogen/docs/architecture/ADR-0002-infrastructure-and-provider-boundaries.md), [ADR-0005](file:///e:/ML%20Projects/videogen/docs/architecture/ADR-0005-storyboard-agent-contract-and-schema.md), [ADR-0006](file:///e:/ML%20Projects/videogen/docs/architecture/ADR-0006-video-model-and-generation-lifecycle.md), [ADR-0007](file:///e:/ML%20Projects/videogen/docs/architecture/ADR-0007-media-assembly.md)

---

## File Structure

```text
app/
  config/
    settings.py                     # Add Veo timeouts, TTS provider, and voice settings
  stages/
    providers.py                    # Capability protocols (Text, Video, TTS, Storage)
    adapters/
      __init__.py
      text.py                       # ModelRouter -> TextGenerationProvider adapter
      veo.py                        # VeoVideoModel -> VideoGenerationProvider adapter
      tts.py                        # EdgeTTS / ElevenLabs -> TextToSpeechProvider adapter
      storage.py                    # GoogleDriveStorage -> StorageProvider adapter
    storyboard.py                   # StoryboardStageHandler (Stage 3)
    media_synthesis.py              # MediaSynthesisStageHandler (Stage 4)
    audio.py                        # AudioStageHandler (Stage 5)
    assembly.py                     # AssemblyStageHandler (Stage 6)
    qc.py                           # QCStageHandler (Stage 7)
    publish.py                      # PublishStageHandler (Stage 8)
tests/
  unit/
    test_veo_capability_adapter.py   # Unit tests for Veo VideoGenerationProvider adapter
    test_tts_adapter.py             # Unit tests for TTS adapter
    test_storage_adapter.py         # Unit tests for Google Drive StorageProvider adapter
    test_downstream_stages.py       # Unit tests for Storyboard, Synthesis, Audio, Assembly, QC, Publish
  integration/
    test_end_to_end_pipeline_postgres.py # Full 8-stage pipeline execution on live PostgreSQL 16
```

---

## Tasks

### Task 1: Complete Text Capability Bridge (`TextGenerationProvider`)
- [ ] Connect `ModelRouter` to the `TextGenerationProvider` protocol in `app/stages/adapters/text.py`.
- [ ] Ensure `TextGenerationRequest` -> `ModelRouter.generate()` properly sets prompt, system prompt, temperature, and clean model override.
- [ ] Add unit tests verifying exception normalization, model identity derivation, and provider fallback behavior.

### Task 2: Implement Google Veo `VideoGenerationProvider` Capability Adapter
- [ ] Implement `VeoCapabilityAdapter` conforming to `VideoGenerationProvider` in `app/stages/adapters/veo.py`:
  - `submit(request: VideoGenerationRequest) -> VideoSubmission` (asynchronous submission via `client.aio.models.generate_videos`).
  - `poll(operation_id: str) -> VideoGenerationResult` (poll via `client.aio.operations.get`, check done, errors, safety filtering, and download video bytes).
  - `reconcile(client_request_id: str) -> VideoReconciliationResult` (check in-flight operation state without duplicating submissions).
- [ ] Normalize provider exceptions (`APIError`, `TimeoutError`) into typed `ModelError` hierarchies.
- [ ] Add unit tests in `tests/unit/test_veo_capability_adapter.py` mocking `google.genai.Client`:
  - Test successful async submission and operation ID generation.
  - Test polling in-progress, completed with bytes, failed, and safety-blocked responses.
  - Test reconciliation behavior with and without known operation IDs.

### Task 3: Implement Text-to-Speech Capability Adapter (`TextToSpeechProvider`)
- [ ] Add TTS configuration settings to `app/config/settings.py` (`videogen_tts_provider`, `tts_voice_id`, `elevenlabs_api_key`).
- [ ] Implement `EdgeTTSAdapter` in `app/stages/adapters/tts.py` using `edge-tts`:
  - Synthesize narration audio bytes, measure duration, validate sample rate.
- [ ] Implement optional `ElevenLabsTTSAdapter` for API-key authenticated high-fidelity speech.
- [ ] Add unit tests in `tests/unit/test_tts_adapter.py` testing audio synthesis, duration measurement, and error handling.

### Task 4: Implement Storage Export Adapter (`StorageProvider`)
- [ ] Implement `GoogleDriveStorageAdapter` in `app/stages/adapters/storage.py` conforming to `StorageProvider`:
  - Verify destination folder permissions.
  - Upload finished master video with resumable/chunked upload.
  - Verify remote checksum and file size against local artifact.
  - Return `StorageResult` with `remote_file_id`, `public_url`, and SHA-256 hash.
- [ ] Add unit tests in `tests/unit/test_storage_adapter.py` with mock Google Drive API responses.

### Task 5: Implement Storyboard & Media Synthesis Stage Handlers (Stages 3 & 4)
- [ ] Implement `StoryboardStageHandler` (`PipelineStage.STORYBOARD`):
  - `load_input()` loads and verifies `ScriptResult` artifact.
  - Executes `StoryboardAgent` to generate `StoryboardResult`.
- [ ] Implement `MediaSynthesisStageHandler` (`PipelineStage.MEDIA_SYNTHESIS`):
  - `load_input()` loads and verifies `StoryboardResult` artifact.
  - Iterates storyboard shots, delegates video generation to `VideoGenerationProvider`, saves individual `VideoClipArtifact` MP4 files, and produces `MediaSynthesisResult`.
- [ ] Add unit tests covering fingerprinting, input validation, and artifact generation.

### Task 6: Implement Audio, Assembly, QC, and Publish Stage Handlers (Stages 5–8)
- [ ] Implement `AudioStageHandler` (`PipelineStage.AUDIO`):
  - Synthesizes narration audio from script via `TextToSpeechProvider`.
- [ ] Implement `AssemblyStageHandler` (`PipelineStage.ASSEMBLY`):
  - Stitches video clips and audio tracks via `FFmpegService` into `master.mp4`.
- [ ] Implement `QCStageHandler` (`PipelineStage.QC`):
  - Evaluates media metrics (`QCMetrics`), enforces duration drift, black frames, and audio presence, issuing `workflow.qc_verdict = 'PASS' | 'FAIL'`.
- [ ] Implement `PublishStageHandler` (`PipelineStage.PUBLISH`):
  - Strictly gates execution on `workflow.qc_verdict == 'PASS'`, uploads master video via `StorageProvider`, and writes `PublishResult`.
- [ ] Add unit tests for all stage handlers.

### Task 7: End-to-End Live PostgreSQL Verification & Production Worker Registration
- [ ] Register all 8 stage handlers in `create_production_worker()`.
- [ ] Create `tests/integration/test_end_to_end_pipeline_postgres.py` executing the complete 8-stage pipeline on live PostgreSQL 16.
- [ ] Run full regression suite (`pytest`), verify $\ge 90.0\%$ coverage, run `alembic check`, and format with `ruff`.
