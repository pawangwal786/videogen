# Phase 9: End-to-End VideoGen Pipeline Architecture & Contract Specification

## 1. Executive Summary

This document establishes the definitive architectural contract and execution model for **Phase 9: End-to-End VideoGen Pipeline**.

Phase 9 transforms the hardened control plane and durable job substrate (established in Phases 7.5, 8.1, and 8.2) into an autonomous, multi-stage agentic video generation pipeline. The architecture strictly adheres to three foundational rules:
1. **Single Orchestration Substrate**: All stages execute as durable PostgreSQL jobs claimed by `JobWorker`. No secondary orchestration engines, event buses, or in-memory DAGs are permitted.
2. **Artifact-Driven State Boundaries**: Stages do not communicate via in-memory Python objects or untyped dictionaries. Immutable, content-addressed artifacts recorded in PostgreSQL serve as the sole inter-stage contract.
3. **Capability-Based Provider Adapters**: External services (LLM, video generation, speech synthesis, cloud storage) are abstracted behind typed capability interfaces (`TextGenerationProvider`, `VideoGenerationProvider`, etc.), completely shielding workflow orchestration from vendor-specific APIs.

---

## 2. Explicit Prohibitions

To prevent architectural degradation, the following patterns are **strictly prohibited**:
- **NO Celery, Temporal, Airflow, or external workflow orchestrators**.
- **NO in-memory workflow state or pipeline DAG engines**.
- **NO event buses (Kafka, RabbitMQ, Redis Pub/Sub)**.
- **NO direct stage-to-stage in-memory Python object passing**.
- **NO `dict[str, Any]` at stage or provider capability boundaries**.
- **NO vendor-specific provider logic inside `WorkflowOrchestrator` or `JobWorker`**.
- **NO artifact overwriting or in-place file mutation**.
- **NO `PUBLISH` job creation without durable `workflow.qc_verdict == 'PASS'`**.
- **NO unbounded media synthesis or FFmpeg concurrency**.
- **NO media file uploads accepted through the HTTP control-plane API**.

---

## 3. End-to-End Pipeline Topology

The video generation lifecycle advances monotonically through eight discrete, durable stages:

```text
                              WORKFLOW (Pending)
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ 1. RESEARCH                                      │
             │    Capability: TextGenerationProvider            │
             │    Artifact: research.json                       │
             └────────────────────────┬─────────────────────────┘
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ 2. SCRIPT                                        │
             │    Capability: TextGenerationProvider            │
             │    Artifact: script.json                         │
             └────────────────────────┬─────────────────────────┘
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ 3. STORYBOARD                                    │
             │    Capability: TextGenerationProvider            │
             │    Artifact: storyboard.json                     │
             └────────────────────────┬─────────────────────────┘
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ 4. MEDIA SYNTHESIS                               │
             │    Capability: VideoGenerationProvider           │
             │    Artifacts: scene-001.mp4 ... scene-NNN.mp4    │
             └────────────────────────┬─────────────────────────┘
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ 5. AUDIO                                         │
             │    Capability: TextToSpeechProvider              │
             │    Artifacts: narration.mp3, bgm.mp3             │
             └────────────────────────┬─────────────────────────┘
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ 6. ASSEMBLY                                      │
             │    Capability: MediaAssemblyEngine (FFmpeg)      │
             │    Artifact: master.mp4                          │
             └────────────────────────┬─────────────────────────┘
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ 7. QUALITY CONTROL (QC)                          │
             │    Capability: MediaProbeEngine (FFprobe + rules)│
             │    Artifact: qc-report.json                      │
             └────────────────────────┬─────────────────────────┘
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
               [Verdict == PASS]               [Verdict == FAIL]
                      │                               │
                      ▼                               ▼
       ┌──────────────────────────────┐        Workflow FAILED
       │ 8. PUBLISH                   │        (or Remediation)
       │    Capability: StorageProvider│
       │    Artifact: publication.json│
       └──────────────┬───────────────┘
                      │
                      ▼
             WORKFLOW COMPLETED
```

---

## 4. `StageHandler[TInput, TOutput]` Execution Contract

### 4.1 Interface Specification

Every stage is implemented by a dedicated handler adhering to the generic `StageHandler[TInput, TOutput]` protocol:

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID
from pydantic import BaseModel, ConfigDict
from app.config.settings import Settings

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


class StageContext(BaseModel):
    """Immutable execution context provided to the stage handler."""

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID
    job_id: str
    attempt_id: str
    worker_id: str
    lease_token: str
    attempt_number: int
    settings: Settings


class StageHandler(Generic[TInput, TOutput], ABC):
    """Abstract contract for all pipeline stage executors."""

    @abstractmethod
    async def load_input(self, context: StageContext) -> TInput:
        """Resolve, validate, and deserialize input artifacts from ArtifactRepository.

        Must NOT access in-memory state of prior stages.
        """
        ...

    @abstractmethod
    async def execute(self, input_data: TInput, context: StageContext) -> TOutput:
        """Pure stage execution.

        Must NOT commit database transactions or persist output artifacts directly.
        Returns a strongly-typed TOutput model.
        """
        ...
```

### 4.2 Infrastructure Ownership Invariant

To guarantee consistency, atomic commits, and centralized error recovery, **stage handlers do not own persistence**.

The execution lifecycle is owned exclusively by `JobWorker`:
1. **Idempotency Check**: `JobWorker` computes the deterministic stage input fingerprint. If an `AVAILABLE` artifact matching the fingerprint already exists for this workflow, the stage execution is skipped, and the existing artifact is reused.
2. **Input Loading**: `JobWorker` invokes `handler.load_input(context)`.
3. **Execution**: `JobWorker` invokes `handler.execute(input_data, context)`.
4. **Heartbeating**: Background task maintains attempt lease fencing during execution.
5. **Artifact Persistence**: `JobWorker` serializes `TOutput`, writes the immutable file to canonical storage, computes the SHA-256 checksum, and records the artifact record in PostgreSQL.
6. **Job Completion & Advancement**: `JobWorker` commits the transaction, marking the job `COMPLETED`, and invokes `WorkflowOrchestrator.advance_workflow()`.

---

## 5. Artifact Immutability & Storage Contract

### 5.1 Monotonic Lifecycle Invariant

Artifacts in the database transition monotonically through their lifecycle:

```text
PENDING ──► AVAILABLE ──► SUPERSEDED
   │
   └──► FAILED
```

- **`PENDING`**: File reservation created; writing/uploading in progress.
- **`AVAILABLE`**: Content written, verified against SHA-256 checksum, and ready for consumption. **An `AVAILABLE` artifact is permanently immutable and is NEVER modified in place.**
- **`SUPERSEDED`**: If a workflow stage is retried or regenerated with different parameters, older artifacts transition to `SUPERSEDED`. They remain on disk for auditability and recovery, never overwritten.
- **`FAILED`**: Generation or upload failed.

### 5.2 Content-Addressed Storage & Confinement

All local artifacts are staged under a structured hierarchy:

```text
{VIDEOGEN_ARTIFACT_ROOT}/{workflow_id}/{stage}/{artifact_id}.{ext}
```

**Filesystem Safety Rules**:
1. `workflow_id` must be a valid RFC 4122 UUID.
2. `artifact_id` must be a valid RFC 4122 UUID.
3. `stage` must be an exact string value of the `JobStage` enum.
4. `ext` must be allow-listed (`json`, `mp4`, `mp3`, `wav`, `png`, `jpg`).
5. **Canonical Path Confinement**: Before reading or writing, the worker resolves the real path and verifies:
   ```python
   target_path.resolve().is_relative_to(artifact_root.resolve())
   ```
   Any path traversal attempt raises `SecurityViolationError` immediately.

---

## 6. Database-Level QC $\to$ PUBLISH Gating Invariant

Quality Control (QC) is not a passive report—it is an **authoritative gate**:

1. **Durable QC Result**: The `QC` stage evaluates duration drift ($\le 10\%$), audio/video synchronization, loudness (-16 to -14 LUFS), resolution ($1920\times1080$), and black frame detection. It produces:
   ```python
   class QCResult(BaseModel):
       verdict: Literal["PASS", "FAIL"]
       metrics: QCMetrics
       failure_reasons: list[str]
   ```
2. **Database Field Enforcement**:
   - The `workflows` table maintains a durable `qc_verdict` column (`PASS`, `FAIL`, or `NULL`).
   - When the `QC` job completes, `workflow.qc_verdict` is set in the same transaction.
3. **Repository-Level Invariant**:
   - `JobRepository.create_next_job()` and `WorkflowOrchestrator.advance_workflow()` enforce:
     ```python
     if next_stage == JobStage.PUBLISH and workflow.qc_verdict != "PASS":
         raise InvariantViolationError("Cannot create PUBLISH job: QC has not PASSED")
     ```
   - If `qc_verdict == "FAIL"`, the workflow transitions to `FAILED` (or routes to revision). Creating a `PUBLISH` job without QC pass is structurally blocked at the persistence layer.

---

## 7. Ambiguous-Submission Recovery Taxonomy

When executing external provider operations (e.g. video generation), a worker may crash after submitting work but before the `provider_operation_id` is committed to PostgreSQL.

Phase 9 establishes a strict three-tier taxonomy:

| Provider Capability Class | Failure Recovery Strategy |
| :--- | :--- |
| **Class A: Idempotency Key Supported** | Recovery worker resubmits the request using the deterministic stage idempotency key. The provider recognizes the key and returns the existing in-flight operation without duplicating work. |
| **Class B: Client Request ID Lookup Supported** | Recovery worker queries the provider by `client_request_id` (the `attempt_id` or stage idempotency key) to locate and attach to the active operation. |
| **Class C: Neither Supported** | **Ambiguous External Side Effect**. The recovery worker does **NOT** blindly resubmit, preventing costly duplicate generations. The attempt is marked `UNRESOLVED` and flagged for operator review or provider-specific safe resolution. |

The state transitions remain fenced by the Phase 7.5 model:
```text
CLAIMED ──► SUBMISSION_PENDING ──► [Submit to Provider] ──► record_submitted() ──► RUNNING ──► poll_loop()
```

---

## 8. Typed Provider Capability Interfaces (Zero `Any`)

Provider adapters are defined strictly by capability, not vendor names:

### 8.1 `TextGenerationProvider`
```python
class TextGenerationRequest(BaseModel):
    system_prompt: str | None = None
    user_prompt: str
    response_model: type[BaseModel] | None = None
    temperature: float = 0.7
    max_tokens: int = 4096


class TextGenerationResult(BaseModel):
    content: str
    provider: str
    model: str
    usage_prompt_tokens: int
    usage_completion_tokens: int
```

### 8.2 `VideoGenerationProvider`
```python
class VideoGenerationRequest(BaseModel):
    client_request_id: str
    prompt: str
    negative_prompt: str | None = None
    duration_seconds: int = 5
    aspect_ratio: Literal["16:9", "9:16"] = "16:9"
    seed: int | None = None


class VideoSubmission(BaseModel):
    provider_operation_id: str
    status: Literal["PENDING", "PROCESSING"]
    estimated_duration_seconds: int | None = None


class VideoGenerationResult(BaseModel):
    provider_operation_id: str
    video_bytes: bytes | None = None
    download_url: str | None = None
    status: Literal["COMPLETED", "FAILED"]
    error_message: str | None = None
```

### 8.3 `TextToSpeechProvider`
```python
class TTSRequest(BaseModel):
    client_request_id: str
    text: str
    voice_id: str
    stability: float = 0.5
    similarity_boost: float = 0.75


class AudioResult(BaseModel):
    audio_bytes: bytes
    audio_format: Literal["mp3", "wav"]
    duration_seconds: float
    sample_rate: int
```

### 8.4 `StorageProvider`
```python
class StorageUploadRequest(BaseModel):
    client_request_id: str
    local_file_path: str
    destination_folder_id: str
    file_name: str
    mime_type: str


class StorageResult(BaseModel):
    provider: str
    remote_file_id: str
    public_url: str | None = None
    checksum_sha256: str
    file_size_bytes: int
    published_at: datetime
```

---

## 9. Stage Idempotency Fingerprint Policy

**Job Retry $\neq$ Stage Recomputation.**

A transient network failure, worker timeout, or worker crash must not trigger duplicate external provider calls if the desired artifact was already computed.

Before invoking external compute, `JobWorker` evaluates the stage's **Deterministic Idempotency Fingerprint**:

| Stage | Idempotency Formula |
| :--- | :--- |
| `RESEARCH` | `sha256(topic + str(target_duration_seconds) + model_name)` |
| `SCRIPT` | `sha256(research_artifact_checksum + pacing_rules + target_duration)` |
| `STORYBOARD` | `sha256(script_artifact_checksum + visual_style_tags)` |
| `MEDIA_SYNTHESIS` | `sha256(storyboard_scene_checksum + aspect_ratio + resolution)` |
| `AUDIO` | `sha256(script_narration_checksum + voice_id + audio_pacing)` |
| `ASSEMBLY` | `sha256(ordered_video_checksums_manifest + audio_checksum + ffmpeg_profile)` |
| `QC` | `sha256(master_video_checksum + qc_thresholds_version)` |
| `PUBLISH` | `sha256(qc_master_video_checksum + destination_root_folder_id)` |

If an `AVAILABLE` artifact with a matching fingerprint exists for the workflow, the worker re-attaches the existing artifact without re-running provider generation.

---

## 10. Per-Process Resource & Concurrency Isolation

1. **Per-Process Semaphore**: Video encoding (FFmpeg) and audio rendering are CPU/IO bound. Each worker process enforces an internal `asyncio.Semaphore(settings.media_max_concurrency)`.
2. **Total Cluster Concurrency**:
   $$\text{Total Media Concurrency} = N_{\text{worker\_processes}} \times \text{media\_max\_concurrency}$$
   The default `media_max_concurrency = 2` ensures host CPU is not starved.
3. **Workload Separation**: HTTP API processes handle zero media synthesis or rendering. They only validate requests and enqueue durable database rows.

---

## 11. Four-Layer Testing Strategy

Every stage and adapter in Phase 9 must be validated across four explicit layers:

### Layer 1: Unit Tests
- Each `StageHandler` tested in isolation with mocked capability providers.
- Scenarios: valid payload, schema validation failure, empty input, provider timeout, retryable error, non-retryable fatal error.

### Layer 2: Provider Contract Tests
- Validates provider adapters against mock HTTP endpoints / SDK boundaries.
- Verifies operation ID extraction, polling status mapping, timeout handling, client request ID propagation, and Class A/B reconciliation.

### Layer 3: Pipeline Integration Tests
- End-to-end execution of all 8 stages (`RESEARCH` $\to$ `PUBLISH`) against a real PostgreSQL 16 database.
- Uses deterministic fake capability adapters to ensure reproducible, fast validation of database state transitions, artifact creation, and QC gating.

### Layer 4: Crash & Recovery Concurrency Tests
- Mandatory failure injection covering all documented failure windows:
  1. Worker crash before provider submission (job returned to `PENDING`).
  2. Worker crash after submission but before persisting `provider_operation_id` (Class A/B/C taxonomy).
  3. Worker crash after persisting `provider_operation_id` (attempt $N-1$ recovery via reconciler).
  4. Worker crash during polling.
  5. Worker crash after artifact file created but before DB commit (stale file replaced by new attempt).
  6. Worker crash after DB commit but before `advance_workflow()` (recovery detects completed job and triggers advancement).
  7. QC `FAIL` blocking `PUBLISH` job generation.
  8. Concurrent duplicate publication requests blocked by database row locking.

---

## 12. Implementation Sequencing Roadmap

Implementation of Phase 9 will proceed across six strictly sequenced sub-phases:

- **Phase 9.0: Contracts & DTOs First**
  - Implement typed stage DTOs (`ResearchResult`, `ScriptResult`, etc.).
  - Implement typed provider DTOs (`TextGenerationRequest`, `VideoGenerationRequest`, etc.).
  - Implement `StageHandler` and `StageContext` protocols.
  - Zero real provider implementations.

- **Phase 9.1: Artifact Infrastructure & Confinement**
  - Implement enhanced `ArtifactRepository`.
  - Content-addressed SHA-256 calculation and immutable writing.
  - Path confinement and UUID validation.
  - Monotonic lifecycle transitions (`PENDING`, `AVAILABLE`, `SUPERSEDED`, `FAILED`).

- **Phase 9.2: Pipeline Runtime Integration**
  - Wire `StageHandler` dispatch into `JobWorker`.
  - Implement infrastructure-owned artifact persistence and stage idempotency checks.
  - Implement database-level `workflow.qc_verdict` gating in `WorkflowOrchestrator`.

- **Phase 9.3: Deterministic Pipeline with Fake Providers**
  - Implement in-memory fake capability providers.
  - Validate end-to-end chaining of all 8 stages against PostgreSQL 16.

- **Phase 9.4: Real Capability Adapters**
  - Implement concrete adapters: Gemini / OpenRouter (`TextGenerationProvider`), Veo (`VideoGenerationProvider`), EdgeTTS / ElevenLabs (`TextToSpeechProvider`), Google Drive (`StorageProvider`).
  - Enforce Google Drive mandatory credentials in production when publish stage is enabled.

- **Phase 9.5: Crash & Recovery Testing**
  - Execute full crash injection and idempotency test matrix.
  - Verify zero state leakage and complete lease fencing across all failure windows.
