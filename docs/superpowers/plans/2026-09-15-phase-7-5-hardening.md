# Phase 7.5 Implementation Plan: Orchestration Hardening & Durable Recovery

**Goal:** Eliminate orchestration concurrency races, consolidate domain architecture, guarantee non-retry of ambiguous external submissions, and prove end-to-end crash recovery against PostgreSQL 16.

**Architecture:** Layered concurrency control combines workflow row-level locking (`SELECT FOR UPDATE`), isolated transaction savepoints (`session.begin_nested()`), and authoritative database unique constraints. Provider reconciliation cleanly separates `RESOLVED`, `CONFIRMED_ABSENT`, and `UNRESOLVED` states to prevent duplicate billable operations. Stale worker execution is strictly rejected via `(job_id, worker_id, lease_token)` validation.

**Tech Stack:** Python 3.12, PostgreSQL 16, SQLAlchemy 2.0 (Async), asyncpg, pytest, pytest-asyncio, Ruff, Alembic.

**Spec:** `docs/superpowers/specs/2026-09-15-phase-7-5-hardening-design.md`

## Global Constraints

- Never use SQLite fallback for integration tests; all integration tests must run against PostgreSQL 16 on port 5433.
- Production statement coverage must remain $\ge 90\%$.
- Code must be clean under `ruff check .` and `ruff format --check .`.
- Zero references to `app.orchestrator` may remain in the codebase.
- Manual reconciliation CLI is out of scope for Phase 7.5.
- Absence of `provider_operation_id` in Veo must always be classified as `UNRESOLVED`, never `CONFIRMED_ABSENT`.
- In concurrency tests, each concurrent task must use an independent `AsyncSession` / transaction from `session_factory()`.
- Savepoint error handling must strictly check for expected unique constraint violations; foreign key, check, and NOT NULL violations must propagate.
- Verify no schema drift via `alembic check`.

---

### Task 1: Repository Concurrency Primitives & Strict Savepoint Isolation

**Files:**
- Modify: `app/repositories/workflow.py`
- Modify: `app/repositories/job.py`
- Test: `tests/integration/test_repositories_postgres.py`

**Interfaces:**
- Produces: `WorkflowRepository.get_workflow_for_update(workflow_id: str) -> WorkflowModel | None`
- Produces: `WorkflowRepository.get_by_idempotency_key(idempotency_key: str) -> WorkflowModel | None`
- Produces: `JobRepository.get_by_logical_key(workflow_id: str, logical_key: str) -> JobModel | None`
- Modifies: `WorkflowRepository.create_workflow` to isolate flush in `async with session.begin_nested()`, recover only on idempotency_key unique collision, and propagate foreign key or other constraint failures.
- Modifies: `JobRepository.create_job` to isolate flush in `async with session.begin_nested()`, recover only on `(workflow_id, logical_key)` unique collision, and propagate foreign key or other constraint failures.

- [ ] **Step 1: Write integration tests for concurrency and constraint propagation**

In `tests/integration/test_repositories_postgres.py`:
- `test_concurrent_create_same_logical_job`: 5 independent sessions creating the same `(workflow_id, logical_key)`. Asserts same persisted `job.id`, exactly 1 row in DB, 0 `PendingRollbackError`.
- `test_concurrent_create_same_idempotency_key`: 5 independent sessions creating the same `idempotency_key`. Asserts same persisted `workflow.id`, exactly 1 row in DB, 0 `PendingRollbackError`.
- `test_create_job_unrelated_integrity_error_propagates`: `create_job(workflow_id="00000000-0000-0000-0000-000000000000", logical_key="invalid_fk", ...)` raises `IntegrityError` (foreign key violation) and is not swallowed.

- [ ] **Step 2: Run tests to verify expected failures**

Run: `uv run pytest tests/integration/test_repositories_postgres.py -k "test_concurrent or test_create_job_unrelated" -v`

- [ ] **Step 3: Implement strict savepoint isolation in JobRepository and WorkflowRepository**

In `app/repositories/job.py`:
```python
def is_unique_violation(exc: IntegrityError, constraint_hint: str | None = None) -> bool:
    orig = getattr(exc, "orig", None)
    sqlstate = (
        getattr(orig, "sqlstate", None)
        or getattr(getattr(orig, "pgcode", None), "__str__", lambda: "")()
    )
    if sqlstate == "23505":  # PostgreSQL unique_violation
        if constraint_hint and hasattr(orig, "diag") and orig.diag.constraint_name:
            return constraint_hint in orig.diag.constraint_name
        return True
    return False
```
Use savepoint in `create_job` and `create_workflow`. If `IntegrityError` is NOT a unique violation on the expected constraint, re-raise immediately. If it is, look up the existing row. If found, return it; otherwise re-raise.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/integration/test_repositories_postgres.py -k "test_concurrent or test_create_job_unrelated" -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add app/repositories/job.py app/repositories/workflow.py tests/integration/test_repositories_postgres.py
git commit -m "feat(repositories): add strict savepoint concurrency and row locking"
```

---

### Task 2: Idempotent WorkflowOrchestrator Stage Advancement

**Files:**
- Modify: `app/orchestration/orchestrator.py`
- Test: `tests/integration/test_orchestrator_worker.py`

**Interfaces:**
- Consumes: `WorkflowRepository.get_workflow_for_update`
- Consumes: `JobRepository.create_job` (savepoint isolated)
- Produces: Idempotent `WorkflowOrchestrator.advance_workflow(workflow_id: str) -> Workflow`

- [ ] **Step 1: Write integration test for concurrent workflow advancement with separate sessions**

In `tests/integration/test_orchestrator_worker.py`:
- `test_advance_workflow_concurrent_idempotency`: 10 independent workers, each with its own `AsyncSession` / `session_factory()`, call `advance_workflow(wf_id)` in parallel.
- Asserts:
  - 0 unique constraint exceptions.
  - Exactly 1 downstream job in database (`script`).
  - Coherent final workflow state (`RUNNING`, current stage `SCRIPT`).

- [ ] **Step 2: Run test to verify failure / current behavior**

Run: `uv run pytest tests/integration/test_orchestrator_worker.py -k test_advance_workflow_concurrent_idempotency -v`

- [ ] **Step 3: Update `advance_workflow` with row locking and defensive pre-checks**

In `app/orchestration/orchestrator.py`:
- Acquire row lock: `wf = await wf_repo.get_workflow_for_update(workflow_id)`.
- If `wf.status` in terminal set (`COMPLETED`, `FAILED`, `CANCELLED`), return immediately.
- For each stage check:
  - Verify if downstream logical job already exists before creating.
  - Advance `current_stage` and `status` cleanly.
  - If any prerequisite job failed, set `workflow.status = WorkflowStatus.FAILED` with matching error codes.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_orchestrator_worker.py -k test_advance_workflow_concurrent_idempotency -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add app/orchestration/orchestrator.py tests/integration/test_orchestrator_worker.py
git commit -m "feat(orchestration): make workflow advancement serialized and idempotent"
```

---

### Task 3: Reconciliation Contract & Granular Provider Exception Handling

**Files:**
- Modify: `app/orchestration/recovery.py`
- Modify: `app/video/reconciler.py`
- Test: `tests/unit/test_video_service.py`

**Interfaces:**
- Produces: `ReconciliationStatus` enum (`RESOLVED`, `CONFIRMED_ABSENT`, `UNRESOLVED`)
- Produces: immutable `ReconciliationOutcome` dataclass
- Modifies: `VeoProviderReconciler.reconcile_submission` to return `ReconciliationOutcome` with specific provider error handling.

- [ ] **Step 1: Write unit tests for VeoProviderReconciler**

In `tests/unit/test_video_service.py`:
- `test_veo_reconciler_returns_unresolved_when_operation_id_missing`: absence of operation ID is always `UNRESOLVED`.
- `test_veo_reconciler_returns_resolved_when_operation_active`: operation found and active returns `RESOLVED`.
- `test_veo_reconciler_transient_error_propagates_or_marks_transient`: transient errors (e.g. 503, connection timeouts) propagate or are flagged rather than converting to terminal `UNRESOLVED`.

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/unit/test_video_service.py -k test_veo_reconciler -v`

- [ ] **Step 3: Implement ReconciliationStatus and update VeoProviderReconciler**

In `app/orchestration/recovery.py`:
```python
class ReconciliationStatus(str, Enum):
    RESOLVED = "resolved"  # operation ID identified; caller resumes polling
    CONFIRMED_ABSENT = "confirmed_absent"  # authoritative provider proof request never reached it
    UNRESOLVED = "unresolved"  # cannot authoritatively determine external status


@dataclass(frozen=True)
class ReconciliationOutcome:
    status: ReconciliationStatus
    provider_operation_id: str | None = None
    error_message: str | None = None
```

In `app/video/reconciler.py`:
- Missing operation ID returns `UNRESOLVED`.
- Active operation returns `RESOLVED`.
- Differentiate transient provider errors from terminal errors. Never return `CONFIRMED_ABSENT` for Veo.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_video_service.py -k test_veo_reconciler -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add app/orchestration/recovery.py app/video/reconciler.py tests/unit/test_video_service.py
git commit -m "feat(video): implement explicit ReconciliationOutcome and Veo reconciler semantics"
```

---

### Task 4: Recovery Worker Ambiguity & Safety Semantics (Zero-Retry)

**Files:**
- Modify: `app/orchestration/recovery.py`
- Test: `tests/integration/test_recovery_worker.py`

**Interfaces:**
- Guarantees: `UNRESOLVED` transitions attempt to `FAILED` with `AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION`, no automated retry, diagnostic metadata recorded, and state remains permanently terminal across repeated recovery runs.

- [ ] **Step 1: Write integration test for persistent zero-retry on unresolved submission**

In `tests/integration/test_recovery_worker.py`:
- `test_recovery_does_not_retry_unresolved_submission`:
  - Worker A claims job, calls `record_submission_pending` without operation ID.
  - Heartbeat expires.
  - RecoveryWorker runs with `MockUnresolvedReconciler`:
    - `job.status == FAILED`, `error_code == AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION`.
    - `available_at is None`.
    - `attempt.status == FAILED`, `attempt_metadata` contains diagnostic details (`provider`, `submission_token`).
  - Run RecoveryWorker a SECOND time:
    - Asserts attempt count remains exactly 1.
    - No new attempts created.
    - Job remains `FAILED`.
  - `advance_workflow()` transitions workflow to `FAILED`.

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/integration/test_recovery_worker.py -k test_recovery_does_not_retry_unresolved_submission -v`

- [ ] **Step 3: Update `RecoveryWorker._handle_expired_attempt`**

In `app/orchestration/recovery.py`:
- `outcome.status == ReconciliationStatus.RESOLVED`: keep attempt `RUNNING`, update `provider_operation_id`.
- `outcome.status == ReconciliationStatus.CONFIRMED_ABSENT`: mark attempt `EXPIRED`, reschedule if `attempt_number < max_attempts`.
- `outcome.status == ReconciliationStatus.UNRESOLVED`:
  - Mark `attempt.status = AttemptStatus.FAILED.value`.
  - Set `attempt.attempt_metadata = {"error_code": "AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION", "provider": attempt.provider, "submission_token": attempt.submission_token, "provider_operation_id": attempt.provider_operation_id}`.
  - Mark `job.status = JobStatus.FAILED.value`, `job.error_code = "AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION"`, `job.available_at = None`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_recovery_worker.py -k test_recovery_does_not_retry_unresolved_submission -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add app/orchestration/recovery.py tests/integration/test_recovery_worker.py
git commit -m "feat(orchestration): enforce persistent zero-retry on unresolved external submissions"
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

Update `tests/unit/test_workflow.py` to validate `Workflow` and `Job` models from `app.orchestration.models`.

- [ ] **Step 2: Delete legacy `app/orchestrator/` files**

Remove `app/orchestrator/` directory and contents.

- [ ] **Step 3: Audit all repository files for remaining references**

Run: `git grep "app.orchestrator"`
Run: `git grep "from app.orchestrator"`
Run: `git grep "import app.orchestrator"`
Assert: 0 matches across the entire repository.

- [ ] **Step 4: Run unit tests to verify no broken imports**

Run: `uv run pytest tests/unit/ -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add -u app/orchestrator/ tests/unit/test_workflow.py
git commit -m "refactor(orchestration): remove legacy app/orchestrator package and audit references"
```

---

### Task 6: Worker Shutdown & Stale Lease Token Protection Test

**Files:**
- Modify: `tests/integration/test_orchestrator_worker.py`

- [ ] **Step 1: Write integration test verifying stale lease token rejection and DB fencing**

In `tests/integration/test_orchestrator_worker.py`:
- `test_worker_shutdown_preserves_attempt_and_blocks_stale_mutation`:
  1. Worker A claims job (receives `token_a`, attempt 1).
  2. Simulate Worker A shutdown / lease expiry (backdate heartbeat).
  3. RecoveryWorker runs -> attempt 1 marked `EXPIRED`, job rescheduled.
  4. Worker B claims job (receives `token_b`, `token_a != token_b`, attempt 2).
  5. Worker A attempts `record_completed(token_a)` -> raises `LeaseConflictError`.
  6. Assert in DB: attempt 2 remains in `RUNNING` status owned by `worker-B`, untouched by Worker A!
  7. Worker B executes `record_completed(token_b)` -> successfully transitions job to `COMPLETED`.

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_orchestrator_worker.py -k test_worker_shutdown_preserves_attempt_and_blocks_stale_mutation -v`
Expected: PASS

- [ ] **Step 3: Commit test**

```bash
git add tests/integration/test_orchestrator_worker.py
git commit -m "test(orchestration): verify worker shutdown and DB-level stale lease mutation fencing"
```

---

### Task 7: Full Crash/Restart Veo Provider Recovery E2E Integration Test

**Files:**
- Modify: `tests/integration/test_orchestrator_worker.py`

- [ ] **Step 1: Write deterministic crash-recovery integration test**

In `tests/integration/test_orchestrator_worker.py`:
- `test_worker_restart_resumes_persisted_provider_operation`:
  1. Create workflow + video job.
  2. Worker A claims job, calls `record_submission_pending`, calls `record_submitted(provider_operation_id="operations/veo-live-101")`, and commits to PostgreSQL.
  3. Deterministic crash point: Worker A disappears / task cancelled without running any cleanup.
  4. Backdate heartbeat to simulate lease expiration.
  5. RecoveryWorker runs with `LiveVeoReconciler` (`RESOLVED`). Reschedules job to `RUNNING`.
  6. Worker B resumes execution, polls provider operation to completion, records output payload.
  7. `advance_workflow()` advances workflow to `MEDIA_ASSEMBLY` and spawns `media_assembly` job.

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_orchestrator_worker.py -k test_worker_restart_resumes_persisted_provider_operation -v`
Expected: PASS

- [ ] **Step 3: Commit test**

```bash
git add tests/integration/test_orchestrator_worker.py
git commit -m "test(orchestration): verify deterministic worker crash recovery resumes persisted provider operation"
```

---

### Task 8: Full Quality Gate Verification

**Files:**
- Entire repository

- [ ] **Step 1: Run full test suite with coverage against PostgreSQL 16**

Run: `uv run pytest --cov=app --cov-report=term-missing`
Assert:
- 0 failures, 0 errors.
- Statement coverage $\ge 90\%$.

- [ ] **Step 2: Verify Alembic migration cleanliness (no schema drift)**

Run: `uv run alembic check` (or verify autogenerate produces 0 diffs).

- [ ] **Step 3: Run Ruff linter and formatter**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`
Assert: 0 errors/warnings.

- [ ] **Step 4: Verify git status and commit cleanliness**

Run: `git status`
Assert: Working tree clean.
