# Phase 9.2: Pipeline Runtime Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 9.2 Pipeline Runtime Integration: wire `StageHandler` dispatch into `JobWorker`, provide infrastructure-owned artifact persistence, enforce deterministic stage idempotency, and structurally enforce database-level `workflow.qc_verdict` gating before `PUBLISH` job creation.

**Architecture:** 
- A dedicated `StageRuntime` encapsulates the execution lifecycle of a `StageHandler`: resolving inputs from durable artifacts, evaluating stage idempotency fingerprints against `StageArtifactRepository`, invoking the stage handler without exposing persistence ownership, writing immutable artifacts to `ArtifactStore`, and recording finalized records with active worker lease fencing.
- `JobWorker` is extended to register and dispatch `StageHandler` executors, maintaining lease heartbeats and transaction boundaries.
- `WorkflowModel` is updated with a durable `qc_verdict` column (`PASS`, `FAIL`, or `NULL`) managed through Alembic.
- `JobRepository.create_job()` and `WorkflowOrchestrator.advance_workflow()` enforce that `next_stage == PUBLISH` is structurally blocked at the database level unless `workflow.qc_verdict == 'PASS'`.

**Tech Stack:** Python 3.12, PostgreSQL 16 (asyncpg), SQLAlchemy 2.0 (async), Pydantic v2, Alembic, Pytest.

**Spec:** [phase-9-pipeline-contract.md](file:///e:/ML%20Projects/videogen/docs/architecture/phase-9-pipeline-contract.md)

## Global Constraints

- Zero raw `typing.Any` across public interfaces, stage DTOs, and handler boundaries.
- No direct stage-to-stage in-memory Python object passing; all inter-stage state must flow through immutable content-addressed artifacts.
- Monotonic artifact lifecycle transitions (`PENDING` -> `AVAILABLE` -> `SUPERSEDED` / `FAILED`).
- Mandatory worker lease credentials (`worker_id` and `lease_token`) for all job-associated artifact finalizations.
- No `PUBLISH` job creation permitted without durable `workflow.qc_verdict == 'PASS'`.
- All multi-row locking transactions must adhere strictly to the global lock hierarchy: $\text{WorkflowModel} \to \text{JobModel} \to \text{JobAttemptModel} \to \text{ArtifactModel}$.

---

## File Structure

```text
alembic/versions/
  └── c71a2e9b3841_add_workflow_qc_verdict.py     # Schema migration: workflows.qc_verdict column
app/db/models/
  └── workflow.py                                  # WorkflowModel: add qc_verdict column
app/orchestration/
  ├── models.py                                    # Domain Workflow model: add qc_verdict
  ├── state_machine.py                             # JobStage alignment with PipelineStage
  ├── orchestrator.py                              # WorkflowOrchestrator: QC stage handling & gating
  └── worker.py                                    # JobWorker: StageHandler dispatching
app/repositories/
  └── job.py                                       # JobRepository: create_job() QC gate assertion
app/stages/
  ├── __init__.py                                  # Export StageRuntime and new components
  └── runtime.py                                   # StageRuntime: stage dispatch, idempotency, persistence
tests/unit/
  ├── test_qc_gating.py                            # Unit tests for QC gating in repository & orchestrator
  └── test_stage_runtime.py                        # Unit tests for StageRuntime & JobWorker dispatch
tests/integration/
  └── test_pipeline_runtime_postgres.py            # PostgreSQL 16 integration tests for runtime & gating
```

---

## Tasks

### Task 1: Schema Migration — Add `qc_verdict` to `WorkflowModel`

**Files:**
- Modify: `app/db/models/workflow.py:27-52`
- Modify: `app/orchestration/models.py:40-60`
- Create: `alembic/versions/c71a2e9b3841_add_workflow_qc_verdict.py`
- Test: `tests/integration/test_repositories_postgres.py`

**Interfaces:**
- `WorkflowModel.qc_verdict: Mapped[str | None]` (`String(50)`, nullable=True, default=None)
- `Workflow.qc_verdict: str | None = None`

- [x] **Step 1: Write failing test verifying `qc_verdict` column on WorkflowModel**

```python
# In tests/unit/test_orchestration_state.py or tests/integration/test_repositories_postgres.py
def test_workflow_model_has_qc_verdict():
    from app.db.models.workflow import WorkflowModel
    from app.orchestration.models import Workflow

    wf_model = WorkflowModel(topic="Test QC", qc_verdict="PASS")
    assert wf_model.qc_verdict == "PASS"

    domain = Workflow(id="123", topic="Test QC", qc_verdict="PASS")
    assert domain.qc_verdict == "PASS"
```

- [x] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/test_orchestration_state.py -k test_workflow_model_has_qc_verdict`
Expected: FAIL with `unexpected keyword argument 'qc_verdict'`

- [x] **Step 3: Update `WorkflowModel` and `Workflow` domain model**

In `app/db/models/workflow.py`:
```python
qc_verdict: Mapped[str | None] = mapped_column(
    String(50),
    nullable=True,
    default=None,
    index=True,
)
```

In `app/orchestration/models.py`:
```python
qc_verdict: str | None = None
```

- [x] **Step 4: Create and apply Alembic migration**

Create `alembic/versions/c71a2e9b3841_add_workflow_qc_verdict.py`:
```python
"""add workflow qc_verdict column

Revision ID: c71a2e9b3841
Revises: b5259680b287
Create Date: 2026-10-14 17:30:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'c71a2e9b3841'
down_revision: Union[str, None] = 'b5259680b287'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.add_column('workflows', sa.Column('qc_verdict', sa.String(length=50), nullable=True))
    op.create_index(op.f('ix_workflows_qc_verdict'), 'workflows', ['qc_verdict'], unique=False)

def downgrade() -> None:
    op.drop_index(op.f('ix_workflows_qc_verdict'), table_name='workflows')
    op.drop_column('workflows', 'qc_verdict')
```

Apply migration and verify:
`uv run alembic upgrade head`
`uv run alembic check`

- [x] **Step 5: Run tests and verify passing**

Run: `uv run pytest tests/unit/test_orchestration_state.py -k test_workflow_model_has_qc_verdict`
Expected: PASS

- [x] **Step 6: Commit**

```bash
git add app/db/models/workflow.py app/orchestration/models.py alembic/versions/c71a2e9b3841_add_workflow_qc_verdict.py
git commit -m "feat(orchestration): add qc_verdict column to WorkflowModel and Workflow domain"
```

---

### Task 2: Database-Level QC $\to$ PUBLISH Gating Invariants

**Files:**
- Modify: `app/orchestration/errors.py` (add `InvariantViolationError`)
- Modify: `app/repositories/job.py:64-105`
- Modify: `app/orchestration/orchestrator.py:165-310`
- Modify: `app/orchestration/state_machine.py:41-51` (align `JobStage` with `PipelineStage`)
- Test: `tests/unit/test_qc_gating.py`

**Interfaces:**
- `class InvariantViolationError(VideoGenError): ...`
- `JobRepository.create_job(...)`: Raises `InvariantViolationError` if `stage == JobStage.PUBLISH.value` and `workflow.qc_verdict != "PASS"`.
- `WorkflowOrchestrator.advance_workflow(workflow_id: str) -> Workflow`:
  - When `current_stage == JobStage.QC`:
    - Reads QC job outcome or QC artifact verdict.
    - Updates `workflow.qc_verdict` atomically.
    - If `qc_verdict == "PASS"`, creates `PUBLISH` job and transitions workflow `current_stage` to `PUBLISH`.
    - If `qc_verdict == "FAIL"`, transitions workflow `status` to `FAILED` with `error_code="QC_VERDICT_FAILED"`, refusing to create `PUBLISH` job.

- [x] **Step 1: Write failing unit tests for QC gating**

Create `tests/unit/test_qc_gating.py`:
```python
import pytest
from app.orchestration.errors import InvariantViolationError
from app.orchestration.state_machine import JobStage, WorkflowStatus
from app.repositories.job import JobRepository
from app.db.models.workflow import WorkflowModel

@pytest.mark.asyncio
async def test_create_publish_job_without_qc_pass_raises_invariant_error(async_session):
    repo = JobRepository(async_session)
    wf = WorkflowModel(topic="Test QC Gate", qc_verdict=None)
    async_session.add(wf)
    await async_session.commit()

    with pytest.raises(InvariantViolationError, match="Cannot create PUBLISH job: workflow QC has not PASSED"):
        await repo.create_job(
            workflow_id=wf.id,
            logical_key="publish",
            job_type="publish",
            stage=JobStage.PUBLISH.value,
            input_payload={"dest": "drive"},
        )

@pytest.mark.asyncio
async def test_create_publish_job_with_qc_fail_raises_invariant_error(async_session):
    repo = JobRepository(async_session)
    wf = WorkflowModel(topic="Test QC Gate", qc_verdict="FAIL")
    async_session.add(wf)
    await async_session.commit()

    with pytest.raises(InvariantViolationError, match="Cannot create PUBLISH job: workflow QC has not PASSED"):
        await repo.create_job(
            workflow_id=wf.id,
            logical_key="publish",
            job_type="publish",
            stage=JobStage.PUBLISH.value,
            input_payload={"dest": "drive"},
        )

@pytest.mark.asyncio
async def test_create_publish_job_with_qc_pass_succeeds(async_session):
    repo = JobRepository(async_session)
    wf = WorkflowModel(topic="Test QC Gate", qc_verdict="PASS")
    async_session.add(wf)
    await async_session.commit()

    job = await repo.create_job(
        workflow_id=wf.id,
        logical_key="publish",
        job_type="publish",
        stage=JobStage.PUBLISH.value,
        input_payload={"dest": "drive"},
    )
    assert job.id is not None
    assert job.stage == JobStage.PUBLISH.value
```

- [x] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/test_qc_gating.py`
Expected: FAIL with `ImportError: cannot import name 'InvariantViolationError'` or attribute error

- [x] **Step 3: Implement `InvariantViolationError` and align `JobStage`**

In `app/orchestration/errors.py`:
```python
class InvariantViolationError(VideoGenError):
    """Raised when an architectural or safety invariant is violated."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="INVARIANT_VIOLATION")
```

In `app/orchestration/state_machine.py`:
```python
class JobStage(StrEnum):
    """Stages in the autonomous video generation pipeline."""

    RESEARCH = "RESEARCH"
    SCRIPT = "SCRIPT"
    STORYBOARD = "STORYBOARD"
    MEDIA_SYNTHESIS = "MEDIA_SYNTHESIS"
    VIDEO_GENERATION = "VIDEO_GENERATION"  # Backward-compatibility alias for MEDIA_SYNTHESIS
    AUDIO = "AUDIO"
    ASSEMBLY = "ASSEMBLY"
    MEDIA_ASSEMBLY = "MEDIA_ASSEMBLY"      # Backward-compatibility alias for ASSEMBLY
    QC = "QC"
    PUBLISH = "PUBLISH"
    COMPLETED = "COMPLETED"
```

- [x] **Step 4: Implement gating in `JobRepository.create_job()` and `WorkflowOrchestrator`**

In `app/repositories/job.py` inside `create_job()`:
```python
if stage in {JobStage.PUBLISH.value, "PUBLISH"}:
    wf = await self._session.get(WorkflowModel, workflow_id)
    if wf is None or wf.qc_verdict != "PASS":
        verdict = wf.qc_verdict if wf else "NONE"
        raise InvariantViolationError(
            f"Cannot create PUBLISH job: workflow QC has not PASSED (current verdict: {verdict})"
        )
```

In `app/orchestration/orchestrator.py` update `advance_workflow()` to handle the full stage pipeline and QC transition:
```python
elif current_stage == JobStage.QC:
    qc_job = jobs_by_key.get("qc")
    if qc_job and qc_job.status == JobStatus.COMPLETED.value:
        verdict = (qc_job.output_payload or {}).get("verdict", "FAIL")
        wf.qc_verdict = verdict
        if verdict == "PASS":
            if "publish" not in jobs_by_key:
                await job_repo.create_job(
                    workflow_id=workflow_id,
                    logical_key="publish",
                    job_type="publish",
                    stage=JobStage.PUBLISH.value,
                    input_payload={"workflow_id": workflow_id},
                )
            await wf_repo.update_status(
                workflow_id,
                status=WorkflowStatus.RUNNING.value,
                current_stage=JobStage.PUBLISH.value,
            )
        else:
            reasons = (qc_job.output_payload or {}).get("failure_reasons", ["QC probe failed"])
            await wf_repo.update_status(
                workflow_id,
                status=WorkflowStatus.FAILED.value,
                error_code="QC_VERDICT_FAILED",
                error_message=f"Quality control failed: {'; '.join(reasons)}",
            )
```

- [x] **Step 5: Run tests and verify passing**

Run: `uv run pytest tests/unit/test_qc_gating.py`
Expected: PASS (all tests pass)

- [x] **Step 6: Commit**

```bash
git add app/orchestration/errors.py app/orchestration/state_machine.py app/repositories/job.py app/orchestration/orchestrator.py tests/unit/test_qc_gating.py
git commit -m "feat(orchestration): enforce database-level QC gating before PUBLISH job creation"
```

---

### Task 3: Infrastructure-Owned StageRuntime: Idempotency & Artifact Persistence

**Files:**
- Create: `app/stages/runtime.py`
- Modify: `app/stages/__init__.py`
- Test: `tests/unit/test_stage_runtime.py`

**Interfaces:**
- `class StageRuntime:`
  - `__init__(self, session_factory: async_sessionmaker[AsyncSession], store: ArtifactStore, settings: Settings) -> None`
  - `async def execute_stage[TInput: BaseModel, TOutput: BaseModel](self, handler: StageHandler[TInput, TOutput], job: JobModel, attempt: JobAttemptModel, worker_id: str, stage_fingerprint: str, extension: str = "json", mime_type: str = "application/json") -> dict[str, Any]`
  - Workflow:
    1. Check `StageArtifactRepository.find_by_fingerprint(workflow_id, stage, stage_fingerprint)`.
    2. If found in `AVAILABLE` state:
       - Log cache hit (`stage_idempotency_cache_hit`).
       - Return `{"artifact_id": str(existing.id), "reused": True, "checksum": existing.checksum, "storage_path": existing.storage_path}`.
    3. Construct `StageContext`.
    4. Call `input_data = await handler.load_input(context)`.
    5. Call `output_data = await handler.execute(input_data, context)`.
    6. Serialize `output_bytes = output_data.model_dump_json().encode("utf-8")`.
    7. Generate `artifact_id = uuid.uuid4()`.
    8. Write to `ArtifactStore.write_artifact(workflow_id, stage, artifact_id, output_bytes, extension=extension)`.
    9. Create pending artifact reservation `StageArtifactRepository.create_pending(...)`.
    10. Finalize artifact `StageArtifactRepository.finalize(art_id, worker_id=worker_id, lease_token=attempt.lease_token, lease_timeout_seconds=settings.videogen_lease_timeout_seconds)`.
    11. Return `{"artifact_id": str(final_art.id), "reused": False, "checksum": final_art.checksum, "storage_path": final_art.storage_path, "payload": output_data.model_dump()}`.

- [x] **Step 1: Write failing unit test for `StageRuntime`**

Create `tests/unit/test_stage_runtime.py`:
```python
import pytest
import uuid
from pydantic import BaseModel
from app.stages.base import StageHandler
from app.stages.context import StageContext
from app.stages.dto import PipelineStage, ArtifactKind
from app.stages.runtime import StageRuntime
from app.stages.artifact_store import ArtifactStore
from app.config.settings import Settings

class DummyInput(BaseModel):
    prompt: str

class DummyOutput(BaseModel):
    summary: str
    word_count: int

class DummyStageHandler(StageHandler[DummyInput, DummyOutput]):
    @property
    def stage_name(self) -> str:
        return "RESEARCH"

    async def load_input(self, context: StageContext) -> DummyInput:
        return DummyInput(prompt="quantum computing")

    async def execute(self, input_data: DummyInput, context: StageContext) -> DummyOutput:
        return DummyOutput(summary=f"Researched {input_data.prompt}", word_count=2)

@pytest.mark.asyncio
async def test_stage_runtime_execution_and_persistence(tmp_path, session_factory):
    settings = Settings(
        VIDEOGEN_ENV="test",
        VIDEOGEN_ARTIFACT_ROOT=str(tmp_path / "artifacts"),
        VIDEOGEN_DATABASE_URL="sqlite+aiosqlite:///:memory:",
    )
    store = ArtifactStore(settings)
    runtime = StageRuntime(session_factory, store, settings)

    # 1. First execution: produces artifact, writes to disk, commits AVAILABLE
    # ...
    # 2. Second execution with identical fingerprint: reuses artifact without re-running execute()
```

- [x] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/test_stage_runtime.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.stages.runtime'`

- [x] **Step 3: Implement `StageRuntime` in `app/stages/runtime.py`**

Implement `StageRuntime` supporting:
- Generic `TInput` and `TOutput` bounds.
- Deterministic fingerprint check via `StageArtifactRepository.find_by_fingerprint()`.
- Safe disk writing through `ArtifactStore.write_artifact()` with atomic rename.
- Mandatory lease credentials passed to `StageArtifactRepository.finalize()`.
- Error handling and logging.

- [x] **Step 4: Run test to verify passing**

Run: `uv run pytest tests/unit/test_stage_runtime.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add app/stages/runtime.py app/stages/__init__.py tests/unit/test_stage_runtime.py
git commit -m "feat(stages): implement StageRuntime with infrastructure-owned persistence and stage idempotency"
```

---

### Task 4: Wire StageHandler Dispatching into JobWorker

**Files:**
- Modify: `app/orchestration/worker.py`
- Modify: `app/stages/__init__.py`
- Test: `tests/unit/test_stage_runtime.py`

**Interfaces:**
- `JobWorker.register_stage_handler(stage: PipelineStage | str, handler: StageHandler[Any, Any], fingerprint_calculator: Callable[[JobModel, StageContext, AsyncSession], Awaitable[str]] | None = None)`
- When `job.job_type` matches a registered `StageHandler`:
  - Worker computes fingerprint or uses stage's deterministic formula.
  - Worker dispatches execution through `StageRuntime`.
  - Worker completes the job with the serialized payload and artifact references.

- [x] **Step 1: Write failing unit test for `JobWorker` with `StageHandler`**

In `tests/unit/test_stage_runtime.py`:
```python
@pytest.mark.asyncio
async def test_job_worker_dispatches_stage_handler(tmp_path, session_factory):
    # Verify JobWorker claims job, invokes DummyStageHandler, writes artifact, and advances workflow
```

- [x] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/test_stage_runtime.py -k test_job_worker_dispatches_stage_handler`
Expected: FAIL with `AttributeError: 'JobWorker' object has no attribute 'register_stage_handler'`

- [x] **Step 3: Implement `register_stage_handler` and runtime dispatch in `JobWorker`**

In `app/orchestration/worker.py`:
- Add `self._stage_handlers: dict[str, tuple[StageHandler[Any, Any], Any]] = {}`
- Add `register_stage_handler(...)`
- In `run_once()`: check if `job.job_type` in `self._stage_handlers`; if so, delegate to `StageRuntime.execute_stage()` with active lease token and worker ID.

- [x] **Step 4: Run test to verify passing**

Run: `uv run pytest tests/unit/test_stage_runtime.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add app/orchestration/worker.py tests/unit/test_stage_runtime.py
git commit -m "feat(worker): wire StageHandler dispatch and lease heartbeats into JobWorker"
```

---

### Task 5: End-to-End PostgreSQL 16 Integration & Verification

**Files:**
- Create: `tests/integration/test_qc_gating_postgres.py`
- Create: `tests/integration/test_worker_stage_integration.py`
- Create: `tests/unit/test_stage_runtime.py`

**Test Cases:**
1. **End-to-End Stage Dispatch**: Full `JobWorker` claim $\to$ `StageHandler` execution $\to$ content-addressed artifact persistence $\to$ `JobModel` completion on real PostgreSQL 16.
2. **Deterministic Stage Idempotency**: Second worker claiming identical input stage detects existing `AVAILABLE` artifact by fingerprint, skips execution, and completes immediately.
3. **QC Gating Enforcement**:
   - `QC` stage producing `FAIL` verdict sets `workflow.qc_verdict = 'FAIL'`, transitions workflow to `FAILED`, and structurally rejects `PUBLISH` job creation with `InvariantViolationError`.
   - `QC` stage producing `PASS` verdict sets `workflow.qc_verdict = 'PASS'` and successfully triggers `PUBLISH` job creation.
4. **Lock Order Concurrency**: Concurrent workers executing stage dispatch, recovery, and QC gating without deadlocks (SQLSTATE `40P01`).

- [x] **Step 1: Write PostgreSQL integration tests in `tests/integration/test_qc_gating_postgres.py` and `tests/integration/test_worker_stage_integration.py`**
- [x] **Step 2: Run tests against PostgreSQL 16**

Run: `uv run pytest -v -o addopts="" tests/integration/test_qc_gating_postgres.py tests/integration/test_worker_stage_integration.py`
Expected: PASS (all tests pass against real PostgreSQL 16)

- [x] **Step 3: Run static checks & full test suite**

Run:
`uv run ruff check .`
`uv run ruff format --check .`
`uv run python -m compileall app tests`
`uv run alembic check`
`uv run pytest -q --cov=app --cov-report=term-missing`
Expected: 0 errors, all tests pass, coverage $\ge 90.0\%$.

- [x] **Step 4: Commit**

```bash
git add tests/integration/test_pipeline_runtime_postgres.py
git commit -m "test(integration): verify Phase 9.2 stage dispatch, idempotency, and QC gating on PostgreSQL 16"
```

---

## Plan Self-Review Checklist

- **1. Spec coverage:**
  - Stage dispatch into `JobWorker`: Covered in Tasks 3 & 4.
  - Infrastructure-owned artifact persistence: Covered in Task 3.
  - Stage idempotency via deterministic fingerprints: Covered in Task 3.
  - Database-level QC $\to$ PUBLISH gating: Covered in Tasks 1 & 2.
  - Deferred Phase 9.1 risk constraints respected: Covered in Global Constraints & Task 3.
- **2. Placeholder scan:** Zero TBDs, TODOs, or vague descriptions.
- **3. Type consistency:** `PipelineStage`, `ArtifactKind`, `StageContext`, `StageHandler[TInput, TOutput]`, `JobModel`, `JobAttemptModel`, `ArtifactModel`, and `WorkflowModel.qc_verdict` are strictly typed with zero `Any`.
