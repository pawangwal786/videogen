# Phase 7.5 Implementation Plan: Orchestration Hardening & Durable Recovery

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate orchestration concurrency races, consolidate domain architecture, guarantee non-retry of ambiguous external submissions, and prove end-to-end crash recovery against PostgreSQL 16.

**Architecture:** Layered concurrency control combines workflow row-level locking (`SELECT FOR UPDATE`), isolated transaction savepoints (`session.begin_nested()`), and authoritative database unique constraints. Provider reconciliation cleanly separates `RESOLVED`, `CONFIRMED_ABSENT`, and `UNRESOLVED` states to prevent duplicate billable operations. Stale worker execution is strictly rejected via `(job_id, worker_id, lease_token)` validation.

**Tech Stack:** Python 3.12, PostgreSQL 16, SQLAlchemy 2.0 (Async), asyncpg, pytest, pytest-asyncio, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-15-phase-7-5-hardening-design.md`

## Global Constraints

- Never use SQLite fallback for integration tests; all integration tests must run against PostgreSQL 16.
- Production statement coverage must remain $\ge 90\%$.
- Code must be clean under `ruff check .` and `ruff format --check .`.
- Zero references to `app.orchestrator` may remain in the codebase.
- Manual reconciliation CLI is out of scope for Phase 7.5.
- Absence of `provider_operation_id` in Veo must always be classified as `UNRESOLVED`, never `CONFIRMED_ABSENT`.

---

### Task 1: Repository Concurrency Primitives & Savepoint Isolation

**Files:**
- Modify: `app/repositories/workflow.py`
- Modify: `app/repositories/job.py`
- Test: `tests/integration/test_repositories_postgres.py`

**Interfaces:**
- Produces: `WorkflowRepository.get_workflow_for_update(workflow_id: str) -> WorkflowModel | None`
- Produces: `WorkflowRepository.get_by_idempotency_key(idempotency_key: str) -> WorkflowModel | None`
- Produces: `JobRepository.get_by_logical_key(workflow_id: str, logical_key: str) -> JobModel | None`
- Modifies: `WorkflowRepository.create_workflow` to isolate flush in `async with session.begin_nested()` and recover `idempotency_key` collisions while propagating unrelated constraint errors.
- Modifies: `JobRepository.create_job` to isolate flush in `async with session.begin_nested()` and recover `(workflow_id, logical_key)` collisions.

- [ ] **Step 1: Write failing integration test for concurrent job creation**

In `tests/integration/test_repositories_postgres.py`:
```python
@pytest.mark.asyncio
async def test_concurrent_create_same_logical_job(pg_engine: AsyncEngine):
    session_factory = async_sessionmaker(bind=pg_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        wf = await wf_repo.create_workflow(topic="Concurrent Job Test")
        await session.commit()
        wf_id = wf.id

    async def create_job_task():
        async with session_factory() as session:
            repo = JobRepository(session)
            job = await repo.create_job(
                workflow_id=wf_id,
                logical_key="test_job",
                job_type="research",
                stage="RESEARCH",
                input_payload={"test": 1},
            )
            await session.commit()
            return job.id

    results = await asyncio.gather(create_job_task(), create_job_task())
    assert results[0] == results[1]

    async with session_factory() as session:
        repo = JobRepository(session)
        jobs = await repo.list_jobs_for_workflow(wf_id)
        assert len(jobs) == 1
```

- [ ] **Step 2: Run test to verify expected failure / current behavior**

Run: `uv run pytest tests/integration/test_repositories_postgres.py::test_concurrent_create_same_logical_job -v`

- [ ] **Step 3: Implement savepoint isolation in JobRepository and WorkflowRepository**

In `app/repositories/job.py`:
```python
    async def get_by_logical_key(
        self,
        workflow_id: str,
        logical_key: str,
    ) -> JobModel | None:
        stmt = select(JobModel).where(
            JobModel.workflow_id == workflow_id,
            JobModel.logical_key == logical_key,
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def create_job(
        self,
        workflow_id: str,
        logical_key: str,
        job_type: str,
        stage: str,
        input_payload: dict[str, Any],
        max_attempts: int = 3,
        available_at: datetime | None = None,
    ) -> JobModel:
        existing = await self.get_by_logical_key(workflow_id, logical_key)
        if existing is not None:
            return existing

        job = JobModel(
            workflow_id=workflow_id,
            logical_key=logical_key,
            job_type=job_type,
            stage=stage,
            input_payload=input_payload,
            max_attempts=max_attempts,
            available_at=available_at or utc_now(),
            status=JobStatus.PENDING.value,
        )
        self._session.add(job)
        try:
            async with self._session.begin_nested():
                await self._session.flush()
        except IntegrityError as exc:
            existing = await self.get_by_logical_key(workflow_id, logical_key)
            if existing is not None:
                return existing
            raise exc
        return job
```

In `app/repositories/workflow.py`:
```python
    async def get_workflow_for_update(self, workflow_id: str) -> WorkflowModel | None:
        stmt = select(WorkflowModel).where(WorkflowModel.id == workflow_id).with_for_update()
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_idempotency_key(self, idempotency_key: str) -> WorkflowModel | None:
        stmt = select(WorkflowModel).where(WorkflowModel.idempotency_key == idempotency_key)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_workflow(
        self,
        topic: str,
        idempotency_key: str | None = None,
    ) -> WorkflowModel:
        if idempotency_key is not None:
            existing = await self.get_by_idempotency_key(idempotency_key)
            if existing is not None:
                if existing.topic == topic:
                    return existing
                raise IdempotencyConflictError(idempotency_key, existing.topic, topic)

        workflow = WorkflowModel(
            topic=topic,
            idempotency_key=idempotency_key,
            status=WorkflowStatus.PENDING.value,
            current_stage=JobStage.RESEARCH.value,
        )
        self._session.add(workflow)
        try:
            async with self._session.begin_nested():
                await self._session.flush()
        except IntegrityError as exc:
            if idempotency_key is not None:
                existing = await self.get_by_idempotency_key(idempotency_key)
                if existing is not None:
                    if existing.topic == topic:
                        return existing
                    raise IdempotencyConflictError(idempotency_key, existing.topic, topic) from exc
            raise exc
        return workflow
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_repositories_postgres.py::test_concurrent_create_same_logical_job -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add app/repositories/job.py app/repositories/workflow.py tests/integration/test_repositories_postgres.py
git commit -m "feat(repositories): add workflow row lock and nested savepoint concurrency protection"
```

---

### Task 2: Idempotent WorkflowOrchestrator Stage Advancement

**Files:**
- Modify: `app/orchestration/orchestrator.py`
- Test: `tests/integration/test_orchestrator_worker.py`

**Interfaces:**
- Consumes: `WorkflowRepository.get_workflow_for_update`
- Consumes: `JobRepository.create_job` (nested savepoint safe)
- Produces: Idempotent `WorkflowOrchestrator.advance_workflow(workflow_id: str) -> Workflow`

- [ ] **Step 1: Write failing test for concurrent workflow advancement**

In `tests/integration/test_orchestrator_worker.py`:
```python
@pytest.mark.asyncio
async def test_advance_workflow_concurrent_idempotency(pg_engine: AsyncEngine):
    session_factory = async_sessionmaker(bind=pg_engine, class_=AsyncSession, expire_on_commit=False)
    orchestrator = WorkflowOrchestrator(session_factory)

    wf = await orchestrator.create_workflow(topic="Parallel Advance Test")
    wf_id = wf.id

    # Complete the research job
    async with session_factory() as session:
        job_repo = JobRepository(session)
        claim = await job_repo.claim_next_job(worker_id="test-worker")
        assert claim is not None
        job, attempt = claim
        await job_repo.record_completed(job.id, "test-worker", attempt.lease_token, {"findings": "verified"})
        await session.commit()

    # Advance workflow concurrently from 10 callers
    async def advance_task():
        return await orchestrator.advance_workflow(wf_id)

    results = await asyncio.gather(*[advance_task() for _ in range(10)])

    # All callers receive coherent workflow state
    for res in results:
        assert res.status == WorkflowStatus.RUNNING
        assert res.current_stage == JobStage.SCRIPT

    # Verify exactly one script job was created in the database
    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(wf_id)
        script_jobs = [j for j in jobs if j.logical_key == "script"]
        assert len(script_jobs) == 1
```

- [ ] **Step 2: Run test to verify failure / baseline**

Run: `uv run pytest tests/integration/test_orchestrator_worker.py::test_advance_workflow_concurrent_idempotency -v`

- [ ] **Step 3: Update `advance_workflow` with row locking and defensive checks**

In `app/orchestration/orchestrator.py`:
- Use `wf = await wf_repo.get_workflow_for_update(workflow_id)` at the start.
- If `wf.status` is in `{COMPLETED, FAILED, CANCELLED}`, immediately return `Workflow.model_validate(wf)`.
- Pre-check downstream jobs before creating them in every stage:
  - `RESEARCH`: if completed, check `jobs_by_key.get("script") is None` before calling `create_job`. Always update `wf.current_stage = JobStage.SCRIPT.value`.
  - `SCRIPT`: if completed, check `jobs_by_key.get("storyboard") is None` before calling `create_job`. Always update `wf.current_stage = JobStage.STORYBOARD.value`.
  - `STORYBOARD`: if completed, for each shot in shots, check if `video:shot:{shot_num}` exists before calling `create_job`. Update `wf.current_stage = JobStage.VIDEO_GENERATION.value`.
  - `VIDEO_GENERATION`: if all video jobs completed, check if `jobs_by_key.get("media_assembly") is None` before calling `create_job`. Update `wf.current_stage = JobStage.MEDIA_ASSEMBLY.value`.
  - `MEDIA_ASSEMBLY`: if assembly job completed, update status to `COMPLETED` and stage to `COMPLETED`.
  - In all stages, if a prerequisite job failed, update workflow status to `FAILED` with matching `error_code` and `error_message`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_orchestrator_worker.py::test_advance_workflow_concurrent_idempotency -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add app/orchestration/orchestrator.py tests/integration/test_orchestrator_worker.py
git commit -m "feat(orchestration): make workflow advancement serialized and idempotent"
```

---

### Task 3: Reconciliation Contract & Provider Reconciler

**Files:**
- Modify: `app/orchestration/recovery.py`
- Modify: `app/video/reconciler.py`
- Test: `tests/unit/test_video_service.py`

**Interfaces:**
- Produces: `ReconciliationStatus` enum (`RESOLVED`, `CONFIRMED_ABSENT`, `UNRESOLVED`)
- Produces: `ReconciliationOutcome` dataclass
- Modifies: `VeoProviderReconciler.reconcile_submission` to return `ReconciliationOutcome`

- [ ] **Step 1: Write unit tests for VeoProviderReconciler status mapping**

In `tests/unit/test_video_service.py`:
```python
@pytest.mark.asyncio
async def test_veo_reconciler_returns_unresolved_when_operation_id_missing():
    mock_model = MagicMock(spec=VideoModel)
    reconciler = VeoProviderReconciler(mock_model)
    outcome = await reconciler.reconcile_submission(
        submission_token="test_token",
        provider_operation_id=None,
    )
    assert outcome.status == ReconciliationStatus.UNRESOLVED
    assert outcome.provider_operation_id is None

@pytest.mark.asyncio
async def test_veo_reconciler_returns_resolved_when_operation_active():
    mock_model = MagicMock(spec=VideoModel)
    mock_model.get_operation_status = AsyncMock(
        return_value=VideoOperation(operation_id="ops/123", status="PROCESSING")
    )
    reconciler = VeoProviderReconciler(mock_model)
    outcome = await reconciler.reconcile_submission(
        submission_token="test_token",
        provider_operation_id="ops/123",
    )
    assert outcome.status == ReconciliationStatus.RESOLVED
    assert outcome.provider_operation_id == "ops/123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_video_service.py -k test_veo_reconciler -v`

- [ ] **Step 3: Implement ReconciliationStatus and update VeoProviderReconciler**

In `app/orchestration/recovery.py` (or shared module imported by reconciler):
```python
class ReconciliationStatus(str, Enum):
    RESOLVED = "resolved"
    CONFIRMED_ABSENT = "confirmed_absent"
    UNRESOLVED = "unresolved"

@dataclass(frozen=True)
class ReconciliationOutcome:
    status: ReconciliationStatus
    provider_operation_id: str | None = None
    error_message: str | None = None
```

In `app/video/reconciler.py`:
Implement `reconcile_submission` returning `ReconciliationOutcome`. When `provider_operation_id` is missing, return `ReconciliationStatus.UNRESOLVED`. When check fails or times out, return `ReconciliationStatus.UNRESOLVED`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_video_service.py -k test_veo_reconciler -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add app/orchestration/recovery.py app/video/reconciler.py tests/unit/test_video_service.py
git commit -m "feat(video): implement explicit ReconciliationStatus outcome contract"
```

---

### Task 4: Recovery Worker Ambiguity & Safety Semantics

**Files:**
- Modify: `app/orchestration/recovery.py`
- Test: `tests/integration/test_recovery_worker.py`

**Interfaces:**
- Consumes: `ReconciliationOutcome`
- Guarantees: `UNRESOLVED` transitions attempt to `FAILED` with `AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION`, no automated retry, and diagnostic metadata recorded.

- [ ] **Step 1: Write integration test for zero-retry on unresolved submission**

In `tests/integration/test_recovery_worker.py`:
```python
@pytest.mark.asyncio
async def test_recovery_does_not_retry_unresolved_submission(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    session_factory = async_sessionmaker(bind=pg_engine, class_=AsyncSession, expire_on_commit=False)
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Unresolved Zero Retry")
    job = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="video:shot:1",
        job_type="video_generation",
        stage=JobStage.VIDEO_GENERATION.value,
        input_payload={"prompt": "prompt"},
        max_attempts=3,
    )
    job_id = job.id
    await db_session.commit()

    # Claim job and record submission pending without provider_operation_id
    claim = await job_repo.claim_next_job(worker_id="crashed-worker")
    assert claim is not None
    _, attempt = claim
    await job_repo.record_submission_pending(
        job_id=job_id,
        worker_id="crashed-worker",
        lease_token=attempt.lease_token,
        provider="veo",
    )
    # Expire the lease
    attempt.heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    await db_session.commit()

    # Recovery worker with Veo reconciler returning UNRESOLVED
    class MockUnresolvedReconciler:
        async def reconcile_submission(self, submission_token, provider_operation_id):
            return ReconciliationOutcome(status=ReconciliationStatus.UNRESOLVED)

    worker = RecoveryWorker(session_factory=session_factory, lease_timeout_seconds=30.0)
    worker.register_reconciler("veo", MockUnresolvedReconciler())

    report = await worker.run_once()
    assert report.expired_attempts_detected == 1
    assert report.jobs_terminally_failed == 1
    assert report.jobs_rescheduled == 0

    db_session.expire_all()
    updated_job = await job_repo.get_job(job_id, load_attempts=True)
    assert updated_job is not None
    assert updated_job.status == JobStatus.FAILED.value
    assert updated_job.error_code == "AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION"
    assert updated_job.available_at is None
    assert len(updated_job.attempts) == 1
    assert updated_job.attempts[0].status == AttemptStatus.FAILED.value
    assert updated_job.attempts[0].attempt_metadata["provider"] == "veo"

    # Verify advance_workflow marks workflow as FAILED
    orchestrator = WorkflowOrchestrator(session_factory)
    updated_wf = await orchestrator.advance_workflow(wf.id)
    assert updated_wf.status == WorkflowStatus.FAILED
    assert updated_wf.error_code == "AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/integration/test_recovery_worker.py::test_recovery_does_not_retry_unresolved_submission -v`

- [ ] **Step 3: Update `RecoveryWorker._handle_expired_attempt`**

In `app/orchestration/recovery.py`:
- Call `reconciler.reconcile_submission(...)`.
- If `outcome.status == ReconciliationStatus.RESOLVED`:
  - `attempt.provider_operation_id = outcome.provider_operation_id`
  - `attempt.status = AttemptStatus.RUNNING.value`
  - `job.status = JobStatus.RUNNING.value`
- If `outcome.status == ReconciliationStatus.CONFIRMED_ABSENT`:
  - Mark `attempt.status = AttemptStatus.EXPIRED.value`
  - Reschedule if `attempt_number < max_attempts`, else fail.
- If `outcome.status == ReconciliationStatus.UNRESOLVED`:
  - `attempt.status = AttemptStatus.FAILED.value`
  - `attempt.completed_at = now`
  - `attempt.attempt_metadata = { "error_code": "AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION", "provider": attempt.provider, "submission_token": attempt.submission_token, "provider_operation_id": attempt.provider_operation_id }`
  - `job.status = JobStatus.FAILED.value`
  - `job.completed_at = now`
  - `job.error_code = "AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION"`
  - `job.error_message = f"Worker crashed during ambiguous submission. Provider '{attempt.provider}' cannot resolve status. Manual reconciliation required."`
  - `job.available_at = None`

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_recovery_worker.py::test_recovery_does_not_retry_unresolved_submission -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add app/orchestration/recovery.py tests/integration/test_recovery_worker.py
git commit -m "feat(orchestration): enforce zero retry and terminal failure on unresolved submissions"
```

---

### Task 5: Package Consolidation & Reference Audit

**Files:**
- Delete: `app/orchestrator/state.py`
- Delete: `app/orchestrator/task.py`
- Delete: `app/orchestrator/workflow.py`
- Delete: `app/orchestrator/__init__.py`
- Modify: `tests/unit/test_workflow.py`

- [ ] **Step 1: Replace `tests/unit/test_workflow.py` with tests for `app.orchestration.models`**

Update `tests/unit/test_workflow.py`:
```python
"""Unit tests for orchestration domain models."""

from app.orchestration.models import Workflow
from app.orchestration.state_machine import JobStage, WorkflowStatus


def test_workflow_model_defaults():
    wf = Workflow(
        id="wf-123",
        topic="AI in Production",
        status=WorkflowStatus.PENDING,
        current_stage=JobStage.RESEARCH,
    )
    assert wf.status == WorkflowStatus.PENDING
    assert wf.topic == "AI in Production"
    assert wf.current_stage == JobStage.RESEARCH
```

- [ ] **Step 2: Delete legacy `app/orchestrator/` files**

Remove `app/orchestrator/state.py`, `task.py`, `workflow.py`, `__init__.py`.

- [ ] **Step 3: Audit all repository files for remaining references**

Run: `git grep "app.orchestrator"`
Expected: Zero occurrences.

- [ ] **Step 4: Run test suite to verify no broken imports**

Run: `uv run pytest tests/unit/test_workflow.py -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add -u app/orchestrator/ tests/unit/test_workflow.py
git commit -m "refactor(orchestration): remove legacy app/orchestrator package and audit references"
```

---

### Task 6: Worker Shutdown & Stale Lease Rejection Integration Test

**Files:**
- Modify: `tests/integration/test_orchestrator_worker.py`

- [ ] **Step 1: Write integration test verifying stale lease token rejection**

In `tests/integration/test_orchestrator_worker.py`:
```python
@pytest.mark.asyncio
async def test_worker_shutdown_preserves_attempt_and_blocks_stale_mutation(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    session_factory = async_sessionmaker(bind=pg_engine, class_=AsyncSession, expire_on_commit=False)
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Stale Lease Token Test")
    job = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="script",
        job_type="script",
        stage=JobStage.SCRIPT.value,
        input_payload={"topic": "Lease Test"},
        max_attempts=3,
    )
    job_id = job.id
    await db_session.commit()

    # Worker A claims job
    claim_a = await job_repo.claim_next_job(worker_id="worker-A")
    assert claim_a is not None
    job_a, attempt_a = claim_a
    token_a = attempt_a.lease_token

    # Simulate Worker A stopping / lease expiring
    attempt_a.heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    await db_session.commit()

    # Recovery worker reclaims the expired attempt
    rec_worker = RecoveryWorker(session_factory=session_factory, lease_timeout_seconds=30.0)
    await rec_worker.run_once()

    # Worker B claims the job
    claim_b = await job_repo.claim_next_job(worker_id="worker-B")
    assert claim_b is not None
    job_b, attempt_b = claim_b
    token_b = attempt_b.lease_token

    assert token_a != token_b
    assert attempt_b.attempt_number == 2

    # Verify stale Worker A cannot mutate or complete the job with its old lease_token
    async with session_factory() as session:
        stale_repo = JobRepository(session)
        with pytest.raises(LeaseConflictError):
            await stale_repo.record_completed(
                job_id=job_id,
                worker_id="worker-A",
                lease_token=token_a,
                output_payload={"stale": True},
            )

    # Verify Worker B can complete the job cleanly
    async with session_factory() as session:
        valid_repo = JobRepository(session)
        completed_job = await valid_repo.record_completed(
            job_id=job_id,
            worker_id="worker-B",
            lease_token=token_b,
            output_payload={"valid": True},
        )
        await session.commit()
        assert completed_job.status == JobStatus.COMPLETED.value
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_orchestrator_worker.py::test_worker_shutdown_preserves_attempt_and_blocks_stale_mutation -v`
Expected: PASS

- [ ] **Step 3: Commit test**

```bash
git add tests/integration/test_orchestrator_worker.py
git commit -m "test(orchestration): verify worker shutdown and stale lease token mutation rejection"
```

---

### Task 7: Full Crash/Restart Veo Provider Recovery E2E Integration Test

**Files:**
- Modify: `tests/integration/test_orchestrator_worker.py`

- [ ] **Step 1: Write integration test for worker crash resuming persisted provider operation**

In `tests/integration/test_orchestrator_worker.py`:
```python
@pytest.mark.asyncio
async def test_worker_restart_resumes_persisted_provider_operation(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    session_factory = async_sessionmaker(bind=pg_engine, class_=AsyncSession, expire_on_commit=False)
    orchestrator = WorkflowOrchestrator(session_factory)

    wf = await orchestrator.create_workflow(topic="Veo Full Recovery Test")
    wf_id = wf.id

    # Create video generation job directly for the workflow
    async with session_factory() as session:
        job_repo = JobRepository(session)
        wf_repo = WorkflowRepository(session)
        await wf_repo.update_status(wf_id, status=WorkflowStatus.RUNNING.value, current_stage=JobStage.VIDEO_GENERATION.value)
        job = await job_repo.create_job(
            workflow_id=wf_id,
            logical_key="video:shot:1",
            job_type="video_generation",
            stage=JobStage.VIDEO_GENERATION.value,
            input_payload={"prompt": "durable video prompt"},
            max_attempts=3,
        )
        await session.commit()
        job_id = job.id

    # Worker A claims job, starts execution, persists operation ID, then crashes
    claim_a = await job_repo.claim_next_job(worker_id="worker-A")
    assert claim_a is not None
    _, attempt_a = claim_a

    async with session_factory() as session:
        repo = JobRepository(session)
        await repo.record_submission_pending(
            job_id=job_id,
            worker_id="worker-A",
            lease_token=attempt_a.lease_token,
            provider="veo",
            submission_token=f"{wf_id}:video:shot:1:1",
        )
        await repo.record_submitted(
            job_id=job_id,
            worker_id="worker-A",
            lease_token=attempt_a.lease_token,
            provider_operation_id="operations/veo-live-999",
        )
        # Backdate heartbeat to simulate crash
        att = await session.get(JobAttemptModel, attempt_a.id)
        assert att is not None
        att.heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=120)
        await session.commit()

    # RecoveryWorker runs with Reconciler that resolves the operation ID
    class LiveVeoReconciler:
        async def reconcile_submission(self, submission_token, provider_operation_id):
            assert provider_operation_id == "operations/veo-live-999"
            return ReconciliationOutcome(
                status=ReconciliationStatus.RESOLVED,
                provider_operation_id=provider_operation_id,
            )

    rec_worker = RecoveryWorker(session_factory=session_factory, lease_timeout_seconds=30.0)
    rec_worker.register_reconciler("veo", LiveVeoReconciler())
    report = await rec_worker.run_once()
    assert report.expired_attempts_detected == 1
    assert report.jobs_rescheduled == 1

    # Worker B resumes execution, polls to completion, persists artifact
    worker_b = JobWorker(session_factory=session_factory)
    class ResumingVideoHandler:
        async def execute(self, job, attempt, worker):
            assert attempt.provider_operation_id == "operations/veo-live-999"
            # Simulate polling provider operation -> completed
            return {"video_uri": "gs://bucket/shot1.mp4", "duration_sec": 5.0}

    worker_b.register_handler("video_generation", ResumingVideoHandler())
    worked = await worker_b.run_once()
    assert worked is True

    # Advance workflow to media assembly
    advanced_wf = await orchestrator.advance_workflow(wf_id)
    assert advanced_wf.current_stage == JobStage.MEDIA_ASSEMBLY

    # Verify media_assembly job was spawned
    async with session_factory() as session:
        repo = JobRepository(session)
        jobs = await repo.list_jobs_for_workflow(wf_id)
        assert any(j.logical_key == "media_assembly" for j in jobs)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_orchestrator_worker.py::test_worker_restart_resumes_persisted_provider_operation -v`
Expected: PASS

- [ ] **Step 3: Commit test**

```bash
git add tests/integration/test_orchestrator_worker.py
git commit -m "test(orchestration): verify worker crash recovery resumes persisted provider operation"
```

---

### Task 8: Full Quality Gate Verification

**Files:**
- Entire repository

- [ ] **Step 1: Run full test suite with coverage**

Run: `uv run pytest --cov=app --cov-report=term-missing`
Assert:
- All tests pass (0 failures, 0 errors).
- Statement coverage $\ge 90\%$.

- [ ] **Step 2: Run Ruff linter and formatter**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`
Assert: 0 errors/warnings.

- [ ] **Step 3: Verify git status and commit cleanliness**

Run: `git status`
Assert: Working tree clean.
