# Phase 9.3: Research & Script Stage Handlers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 9.3: Research & Script Stage Handlers. Build typed, durable Research and Script stage executors, enforce cryptographic cached artifact integrity in `StageRuntime`, transition `WorkflowOrchestrator` to strictly artifact-driven inter-stage contracts, and verify end-to-end execution and recovery on PostgreSQL 16.

**Architecture:**
- `ResearchStageHandler` and `ScriptStageHandler` implement `StageHandler[TInput, TOutput]` with strict Pydantic DTOs.
- `ScriptStageHandler.load_input()` resolves and verifies the upstream `ResearchResult` artifact directly from durable storage using `StageArtifactRepository.get_available(..., verify_integrity=True)`.
- `StageRuntime` enforces the repository's integrity-verification contract before consuming cached artifacts on fingerprint hits, preventing silent cache corruption.
- `JobWorker` dispatches stages with canonical fingerprints derived via `StageFingerprintCalculator` incorporating upstream artifact checksums.
- `WorkflowOrchestrator` creates downstream jobs referencing durable artifact identifiers and checksums rather than embedding legacy in-memory dictionaries.

**Tech Stack:** Python 3.12, PostgreSQL 16 (asyncpg), SQLAlchemy 2.0 (async), Pydantic v2, Pytest.

**Spec:** [phase-9-pipeline-contract.md](file:///e:/ML%20Projects/videogen/docs/architecture/phase-9-pipeline-contract.md)
**Technical Debt Register:** [phase-9-1-technical-debt.md](file:///e:/ML%20Projects/videogen/docs/architecture/phase-9-1-technical-debt.md) (Frozen)

## Global Constraints

- Zero raw `typing.Any` or untyped dictionaries across public interfaces, stage DTOs, and handler boundaries.
- No direct stage-to-stage in-memory Python object passing; all inter-stage state must flow through immutable content-addressed artifacts.
- Cached artifacts must be verified for cryptographic integrity (SHA-256 and size) before consumption.
- Downstream jobs must receive durable artifact identifiers (`_artifact_id`, `_checksum`), never embedded raw stage output dictionaries.
- Monotonic artifact lifecycle transitions (`PENDING` -> `AVAILABLE` -> `SUPERSEDED` / `FAILED`).
- Mandatory worker lease credentials (`worker_id` and `lease_token`) for all job-associated artifact finalizations.
- All multi-row locking transactions must adhere strictly to the global lock hierarchy: $\text{WorkflowModel} \to \text{JobModel} \to \text{JobAttemptModel} \to \text{ArtifactModel}$.
- Do not implement external AI provider network calls in this phase; use test doubles / mock capability providers conforming to `TextModel` / `Agent` contracts (Phase 9.4 scope).

---

## File Structure

```text
app/stages/
  ├── context.py                                  # Add job_payload to StageContext
  ├── runtime.py                                  # Enforce integrity verification on cached artifacts
  ├── research.py                                 # ResearchStageHandler implementation
  ├── script.py                                    # ScriptStageHandler implementation
  └── __init__.py                                 # Export ResearchStageHandler, ScriptStageHandler
app/orchestration/
  ├── worker.py                                   # Canonical stage fingerprinting in StageHandlerAdapter
  └── orchestrator.py                             # Artifact-driven downstream job creation
tests/unit/
  ├── test_stage_runtime.py                       # Verify integrity verification on cache hit
  ├── test_research_stage.py                      # Unit tests for ResearchStageHandler
  └── test_script_stage.py                        # Unit tests for ScriptStageHandler
tests/integration/
  └── test_research_script_pipeline_postgres.py   # PostgreSQL 16 integration tests for Research -> Script
```

---

## Tasks

### Task 1: Enforce Cached Artifact Integrity & Context Enrichment in `StageRuntime` (Gaps 1 & 2)

**Files:**
- Modify: `app/stages/context.py:10-26`
- Modify: `app/stages/runtime.py:20-85`
- Modify: `app/orchestration/worker.py:75-105`
- Test: `tests/unit/test_stage_runtime.py`

**Interfaces:**
- `StageContext.job_payload: Mapping[str, Any]` (immutable mapping of the triggering job's input parameters)
- `create_stage_context(job: JobModel, attempt: JobAttemptModel, settings: Settings) -> StageContext`: populates `job_payload=job.input_payload or {}`
- `StageRuntime.execute_stage()`: on fingerprint hit (or raced conflict), calls `await self._repo.get_available(UUID(cached_artifact.id), verify_integrity=True)` before reading and returning the cached artifact. Raises `SecurityViolationError` if checksum or file size mismatches disk.
- `_StageHandlerAdapter`: resolves canonical fingerprints using `StageFingerprintCalculator` when registering standard stages.

- [x] **Step 1: Write failing unit tests for cached artifact integrity verification**

In `tests/unit/test_stage_runtime.py`:
```python
@pytest.mark.asyncio
async def test_stage_runtime_verifies_cached_artifact_integrity(tmp_path, session_factory):
    """Corrupted cached artifact file on disk must raise SecurityViolationError on cache hit."""
    from app.stages.artifact_errors import SecurityViolationError
    # 1. Execute stage successfully to create cached artifact
    # 2. Tamper with file on disk (overwrite bytes)
    # 3. Second execution with same fingerprint must detect corruption and raise SecurityViolationError
```

- [x] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_stage_runtime.py -k test_stage_runtime_verifies_cached_artifact_integrity`
Expected: FAIL (currently reads raw bytes without verifying checksum on cache hit)

- [x] **Step 3: Update `StageContext` and `create_stage_context`**

In `app/stages/context.py`:
Add `job_payload: Mapping[str, Any] = Field(default_factory=dict, description="Job input parameters")`.

In `app/stages/runtime.py`:
Pass `job_payload=job.input_payload or {}` in `create_stage_context()`.

- [x] **Step 4: Update `StageRuntime.execute_stage()` to verify integrity**

In `app/stages/runtime.py`:
```python
if cached_artifact is not None:
    # Enforce integrity verification against physical disk file
    verified_artifact = await self._repo.get_available(
        UUID(cached_artifact.id),
        verify_integrity=True,
    )
    raw_bytes = await self._store.read_artifact(Path(verified_artifact.storage_path))
    cached_output = output_type.model_validate_json(raw_bytes)
    return StageExecutionResult[TOutput](
        output=cached_output,
        artifact_id=UUID(verified_artifact.id),
        storage_path=verified_artifact.storage_path,
        checksum=verified_artifact.checksum or "",
        cached=True,
    )
```
Apply the same integrity verification to the `FingerprintConflictError` race-resolution branch.

- [x] **Step 5: Run tests and verify passing**

Run: `pytest tests/unit/test_stage_runtime.py`
Expected: PASS

- [x] **Step 6: Commit**

```bash
git add app/stages/context.py app/stages/runtime.py app/orchestration/worker.py tests/unit/test_stage_runtime.py
git commit -m "feat(stages): enforce cryptographic integrity verification on cached stage artifacts"
```

---

### Task 2: Implement `ResearchStageHandler` & Canonical Fingerprinting

**Files:**
- Create: `app/stages/research.py`
- Modify: `app/stages/__init__.py`
- Test: `tests/unit/test_research_stage.py`

**Interfaces:**
- `class ResearchStageHandler(StageHandler[ResearchRequest, ResearchResult]):`
  - `__init__(self, agent: ResearchAgent | None = None) -> None`
  - `stage_name -> str = "RESEARCH"`
  - `async def load_input(self, context: StageContext) -> ResearchRequest:`
    Reads `topic`, `target_duration_seconds`, `target_audience`, etc., from `context.job_payload`.
  - `async def execute(self, input_data: ResearchRequest, context: StageContext) -> ResearchResult:`
    Invokes `self.agent.run(input_data)` and returns `ResearchResult`.
  - `calculate_fingerprint(job: JobModel, context: StageContext) -> str:`
    Calls `StageFingerprintCalculator.for_research(topic, target_duration_seconds, model_name)`.

- [x] **Step 1: Write failing unit tests for `ResearchStageHandler`**

Create `tests/unit/test_research_stage.py`:
```python
import pytest
from app.stages.research import ResearchStageHandler
from app.stages.context import StageContext
from app.config.settings import Settings
import uuid

@pytest.mark.asyncio
async def test_research_stage_handler_load_input_and_execute():
    # Test loading from context.job_payload and executing through mock agent
```

- [x] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_research_stage.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.stages.research'`

- [x] **Step 3: Implement `ResearchStageHandler`**

Create `app/stages/research.py`:
- Pure stage executor implementing `StageHandler[ResearchRequest, ResearchResult]`.
- Implements `calculate_fingerprint` using `StageFingerprintCalculator.for_research`.
- Integrates with `ResearchAgent` (using injectable text model or agent mock for tests).

Export in `app/stages/__init__.py`.

- [x] **Step 4: Run test to verify passing**

Run: `pytest tests/unit/test_research_stage.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add app/stages/research.py app/stages/__init__.py tests/unit/test_research_stage.py
git commit -m "feat(stages): implement ResearchStageHandler with canonical fingerprinting"
```

---

### Task 3: Implement `ScriptStageHandler` & Durable Research Artifact Ingestion

**Files:**
- Create: `app/stages/script.py`
- Modify: `app/stages/__init__.py`
- Test: `tests/unit/test_script_stage.py`

**Interfaces:**
- `class ScriptStageHandler(StageHandler[ScriptRequest, ScriptResult]):`
  - `__init__(self, session_factory: async_sessionmaker[AsyncSession], store: ArtifactStore, agent: ScriptAgent | None = None) -> None`
  - `stage_name -> str = "SCRIPT"`
  - `async def load_input(self, context: StageContext) -> ScriptRequest:`
    1. Extracts `research_artifact_id` from `context.job_payload`.
    2. Queries `StageArtifactRepository.get_available(UUID(research_artifact_id), verify_integrity=True)`.
    3. Reads raw bytes via `store.read_artifact(Path(artifact.storage_path))`.
    4. Deserializes `ResearchResult.model_validate_json(raw_bytes)`.
    5. Returns `ScriptRequest(research=research_result, pacing_wpm=...)`.
  - `async def execute(self, input_data: ScriptRequest, context: StageContext) -> ScriptResult:`
    Invokes `self.agent.run(input_data)` and returns `ScriptResult`.
  - `calculate_fingerprint(job: JobModel, context: StageContext) -> str:`
    Calls `StageFingerprintCalculator.for_script(research_artifact_checksum, pacing_wpm, target_duration_seconds)`.

- [x] **Step 1: Write failing unit tests for `ScriptStageHandler`**

Create `tests/unit/test_script_stage.py`:
- Test `load_input()` successfully resolves and verifies upstream research artifact by ID.
- Test `load_input()` raises `SecurityViolationError` if research artifact file on disk was corrupted.
- Test `load_input()` raises `ArtifactUnavailableError` if research artifact is missing or not AVAILABLE.
- Test `execute()` produces valid `ScriptResult`.

- [x] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_script_stage.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.stages.script'`

- [x] **Step 3: Implement `ScriptStageHandler`**

Create `app/stages/script.py`:
- Takes `session_factory` and `store` in `__init__`.
- In `load_input()`, fetches and verifies upstream `ResearchResult` artifact strictly from durable storage.
- Implements `calculate_fingerprint` using `StageFingerprintCalculator.for_script`.
- Implements `execute()` returning `ScriptResult`.

Export in `app/stages/__init__.py`.

- [x] **Step 4: Run test to verify passing**

Run: `pytest tests/unit/test_script_stage.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add app/stages/script.py app/stages/__init__.py tests/unit/test_script_stage.py
git commit -m "feat(stages): implement ScriptStageHandler with durable artifact resolution and integrity checks"
```

---

### Task 4: Artifact-Driven Orchestration in `WorkflowOrchestrator` (Gap 3)

**Files:**
- Modify: `app/orchestration/orchestrator.py:205-240`
- Test: `tests/unit/test_orchestration_state.py`

**Interfaces:**
- `WorkflowOrchestrator.advance_workflow(workflow_id: str)`:
  - When `current_stage == JobStage.RESEARCH` completes:
    Extracts `_artifact_id` and `_checksum` from `r_job.output_payload`.
    Creates `script` job with:
    ```python
    input_payload = {
        "workflow_id": str(workflow_id),
        "topic": wf.topic,
        "research_artifact_id": r_job.output_payload["_artifact_id"],
        "research_artifact_checksum": r_job.output_payload["_checksum"],
        "target_duration_seconds": 60,
    }
    ```
    (Zero raw `research_output` dictionaries passed in `input_payload`).

- [x] **Step 1: Write unit test verifying `advance_workflow` creates artifact-driven downstream jobs**

In `tests/unit/test_orchestration_state.py`:
Verify that when research job completes with `_artifact_id` and `_checksum`, the generated script job's `input_payload` contains `research_artifact_id` and `research_artifact_checksum`, and does NOT contain `research_output`.

- [x] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_orchestration_state.py -k test_advance_workflow_creates_artifact_driven_job`
Expected: FAIL

- [x] **Step 3: Update `WorkflowOrchestrator.advance_workflow`**

In `app/orchestration/orchestrator.py`:
Refactor stage advancement from `RESEARCH` to `SCRIPT` to pass only durable artifact identifiers and checksums.

- [x] **Step 4: Run test to verify passing**

Run: `pytest tests/unit/test_orchestration_state.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add app/orchestration/orchestrator.py tests/unit/test_orchestration_state.py
git commit -m "refactor(orchestrator): transition stage job creation to durable artifact references"
```

---

### Task 5: End-to-End PostgreSQL 16 Pipeline Integration: Research $\to$ Script (Gap 4)

**Files:**
- Create: `tests/integration/test_research_script_pipeline_postgres.py`

**Test Cases:**
1. **Full Research $\to$ Script Pipeline Execution**:
   - Create workflow $\to$ Claim & execute Research stage via `JobWorker` $\to$ Durable `ResearchResult` artifact written and finalized.
   - Advance workflow $\to$ Script job created with `research_artifact_id` and `_checksum`.
   - Claim & execute Script stage via `JobWorker` $\to$ `ScriptStageHandler` loads research artifact from disk with integrity check $\to$ Durable `ScriptResult` artifact written and finalized.
2. **Deterministic Stage Idempotency**:
   - Re-running Script stage with identical research artifact checksum and pacing skips execution and reuses cached artifact.
3. **Failure & Recovery Verification**:
   - **Corrupted Upstream Artifact**: Tampered research artifact on disk fails with `SecurityViolationError` during Script `load_input()`.
   - **Corrupted Cached Artifact**: Tampered cached script artifact on disk triggers `SecurityViolationError` during `StageRuntime` cache hit.
   - **Stale Lease**: Worker with expired lease token fails to finalize script artifact.

- [x] **Step 1: Write integration tests in `tests/integration/test_research_script_pipeline_postgres.py`**
- [x] **Step 2: Run tests against PostgreSQL 16**

Run: `pytest -v tests/integration/test_research_script_pipeline_postgres.py --no-cov`
Expected: PASS

- [x] **Step 3: Run static checks & full regression suite**

Run:
`python -m compileall app alembic tests`
`ruff check app alembic tests`
`ruff format --check app alembic tests`
`alembic check`
`pytest`
Expected: 0 errors, all tests pass, line coverage $\ge 90.0\%$.

- [x] **Step 4: Commit**

```bash
git add tests/integration/test_research_script_pipeline_postgres.py
git commit -m "test(integration): verify Research -> Script artifact pipeline, idempotency, and recovery on PostgreSQL 16"
```

---

## Plan Self-Review Checklist

- **1. Spec & Gaps Coverage:**
  - Gap 1: Canonical stage-specific fingerprints (Tasks 1, 2, 3).
  - Gap 2: Cached artifact cryptographic integrity verification (Task 1).
  - Gap 3: Authoritative inter-stage contract via durable artifacts (Task 4).
  - Gap 4: Recovery, corruption, and lease failure verification (Task 5).
- **2. Scope Control:**
  - Phase 9.1 technical debt remains frozen.
  - No real external AI provider calls (capability test doubles used).
- **3. Typing:**
  - Zero `Any` in public interfaces, handler contracts, or DTOs.
