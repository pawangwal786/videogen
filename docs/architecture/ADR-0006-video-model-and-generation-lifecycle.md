# ADR-0006: Video Model Abstraction, Generation Lifecycle, and Idempotency

## Status

Accepted

## Context

Phase 5 introduces the video generation layer in VideoGen. Unlike text generation (which is synchronous or fast-streaming, stateless, and relatively low-cost), video generation is fundamentally different:
1. **Asynchronous execution**: Models (such as Google Veo) run long-running asynchronous compute jobs that must be submitted and polled.
2. **High compute and financial cost**: Duplicate submissions caused by timeouts or blind retries lead to multiple billable generation operations.
3. **Discrete provider capabilities**: Video models typically support fixed discrete durations (e.g. 5s or 10s for Veo) rather than arbitrary floating-point durations.
4. **Binary artifact delivery**: Output is raw video bytes (MP4) or cloud URIs, requiring a staging pipeline before persistent cloud storage.

## Decisions

### 1. Dedicated VideoModel Protocol (Modality Isolation)
- In accordance with our architecture principles, we do not overload `TextModel` or create a generic multi-modal god router.
- We define a dedicated [VideoModel](file:///e:/ML%20Projects/videogen/app/models/video.py) protocol:
  ```python
  class VideoModel(Protocol):
      async def submit_generation(self, request: VideoGenerationRequest) -> VideoOperation: ...
      async def get_operation_status(self, operation_id: str) -> VideoOperation: ...
      async def cancel_generation(self, operation_id: str) -> None: ...
  ```
- **Cancellation Semantics**: `cancel_generation()` requests cancellation when supported by the provider; otherwise the adapter reports that cancellation is unsupported and the job remains subject to provider completion. The domain protocol does not pretend cancellation succeeded if the underlying provider lacks cancellation support.
- Upstream agents (such as `StoryboardAgent`) have zero coupling to `VideoModel` or provider adapters.

### 2. Provider-Specific Duration Normalization Policy
- `StoryboardShot.estimated_duration_seconds` represents the creative and narrative timing target (floating-point seconds).
- `VideoGenerationRequest` represents the concrete execution request after provider capability resolution.
- **Policy**: Veo generation uses provider-supported discrete clip durations (e.g. 5 or 10 seconds). Normalization from storyboard target duration to provider duration is an explicit generation-layer decision and must not mutate the storyboard contract.
- Phase 6 (Editing and Media Assembly via MPT/FFmpeg) owns trimming, concatenation, or speed-matching the generated clips to the exact storyboard duration.

### 3. Stable Logical-Shot Idempotency
- To protect against duplicate billable generations, we enforce a strict separation between a logical shot generation request and an execution retry attempt:
  ```text
  Logical Identity = f"{workflow_id}:shot:{shot_number}"
  Attempt Number   = Execution metadata (attempt=1, 2, ...)
  ```
- Before submitting any request to a video provider, the service executes the following state check:
  ```text
  Is there already an operation record for this logical shot?
      │
      ├── Active (submitted / processing) ──> Resume polling existing operation_id
      ├── Completed                      ──> Reuse existing generated artifact
      ├── Definitively failed/cancelled  ──> Initiate new attempt according to policy
      └── None                           ──> Submit new provider operation
  ```
- This guarantees that network timeouts or service restarts within a session never trigger duplicate concurrent generations for the same shot.
- **Process-Local Scope & Phase 7 Production Requirement**: The current Phase 5 idempotency registry is process-local (`dict[str, VideoJobRecord]`). This protects against duplicate generation within a single `VideoGenerationService` instance. In Phase 7 (Persistent Orchestration), job state and idempotency keys must be backed by a durable store (PostgreSQL as the authoritative source of truth with atomic claims) to ensure safety across worker restarts, multiple containers, and distributed environments.

### 4. Configurable Provider Adapter (VeoVideoModel)
- [VeoVideoModel](file:///e:/ML%20Projects/videogen/app/models/veo.py) implements `VideoModel` against Google's official `google-genai` SDK (`client.aio.models.generate_videos` and `client.aio.operations.get`).
- The model name is not hardcoded; it is configurable via `Settings.VEO_MODEL_NAME` with a default of `"veo-2.0-generate-001"`.
- Provider exceptions (`genai.errors.APIError`) are normalized into the established [ModelError](file:///e:/ML%20Projects/videogen/app/models/errors.py) hierarchy.
- Content moderation or safety blocks (`rai_media_filtered_reasons`) are captured and translated into structured `ModelResponseError` failures.

### 5. Artifact and Storage Boundary
- Video generation preserves the strict infrastructure boundary:
  ```text
  Veo Provider
       │ (raw bytes / URI)
       ▼
  Local Staging (tmp/staging/{workflow_id}/shot_{shot_number}.mp4)
       │
       ▼
  ArtifactStorage (GoogleDriveStorage: workflows/{workflow_id}/shots/shot_{shot_number}.mp4)
       │
       ▼
  ArtifactRef (immutable tracking record with artifact_id, size, storage_path)
  ```
- Neither Veo nor the domain agents directly manage Google Drive paths or tokens.

### 6. Generation Lifecycle State Machine
Every generation job transitions deterministically through well-defined lifecycle states:
```text
queued ──> submitted ──> processing ──> completed
   │           │             │
   └───────────┴─────────────┴───────> failed / cancelled
```

## Consequences

- **Cost Safety**: Stable logical idempotency prevents accidental double-billing when recovering from transient network interruptions within an execution session.
- **Phase 6 Duration Reconciliation Boundary**: Phase 6 (Media Assembly / Editing), not Phase 5, owns the reconciliation between storyboard target durations and actual Veo clip durations via trimming, concatenation, or speed adjustments.
- **Phase 7 Persistent Orchestration Requirement**: Multi-worker, multi-container deployments will move the in-memory idempotency registry into durable PostgreSQL tables with atomic claims.
- **Provider Interchangeability**: Replacing or augmenting Veo with another video diffusion engine requires only an adapter implementing `VideoModel`, leaving storyboard contracts and downstream assembly intact.
- **Traceability**: Every video shot artifact is uniquely linked back to its storyboard shot, script scene, and narrative beat.
