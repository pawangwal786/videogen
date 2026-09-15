# Phase 7.5 Design Specification: Orchestration Hardening & Durable Recovery

**Document:** `docs/superpowers/specs/2026-09-15-phase-7-5-hardening-design.md`  
**Date:** 2026-09-15  
**Author:** Principal Software Engineer  
**Status:** Approved for Implementation Planning  
**Target:** VideoGen Orchestration Runtime & Durability Gate

---

## 1. Executive Summary & Goals

Phase 7 established the core persistence foundation for VideoGen using PostgreSQL 16, Alembic migrations, SQLAlchemy repositories, state machine definitions, atomic `SKIP LOCKED` job claiming, and lease-token ownership.

However, an architectural audit before exposing the FastAPI control plane (Phase 8) identified five critical orchestration issues:
1. **Concurrent Stage Advancement:** Unlocked concurrent calls to `advance_workflow()` could race to insert downstream jobs, producing `IntegrityError` or out-of-sync workflow stages.
2. **Package Duplication:** The repository retained the legacy scaffold `app/orchestrator/` alongside the active `app/orchestration/` runtime.
3. **Unverified Provider Restart Recovery:** While unit tests verified individual components, the end-to-end restart-and-reconcile cycle was unproven against real PostgreSQL.
4. **Unverified Worker Interruption:** Shutdown and lease expiration behaviors lacked an integration test verifying that a reclaimed job rejects stale worker updates.
5. **Unsafe Ambiguity Semantics:** A worker crashing during `SUBMISSION_PENDING` without a persisted `provider_operation_id` was treated as "confirmed absent" and rescheduled, creating a double-billing risk on external video providers (Veo).

Phase 7.5 resolves these five issues and proves durable recovery against PostgreSQL 16 before any API work begins.

---

## 2. Invariants & Guarantees

### 2.1 Concurrency & Idempotency
- **Workflow Serialization:** Workflow stage transitions are serialized per workflow using PostgreSQL row-level locking (`SELECT ... FOR UPDATE` via `WorkflowRepository.get_workflow_for_update`).
- **Defensive Pre-check:** `WorkflowOrchestrator.advance_workflow` verifies whether downstream logical jobs already exist before spawning them.
- **Isolated Savepoints:** `JobRepository.create_job` and `WorkflowRepository.create_workflow` isolate creation inside `session.begin_nested()`. If a uniqueness conflict occurs, only the savepoint rolls back; the outer transaction remains valid and retrieves the existing row. Unrelated database constraint errors (foreign keys, null violations) are not masked and propagate normally.
- **Database Authoritative Boundary:** `UNIQUE(workflow_id, logical_key)` and `UNIQUE(job_id, attempt_number)` remain the authoritative invariant.

### 2.2 Lease Ownership & Stale Worker Protection
- **Ownership Tuple:** Execution ownership is strictly defined by `(job_id, worker_id, lease_token)`.
- **Stale Worker Mutation Prevention:** An attempt claimed by Worker B with `lease_token_B` cannot be updated, completed, or failed by Worker A presenting expired `lease_token_A`.

### 2.3 Provider Reconciliation & Ambiguity Safety
- **The Golden Safety Rule:** *Never automatically retry an unresolved external submission when doing so could create a second billable provider operation.*
- **Reconciliation Outcome Classification:**
  - `RESOLVED`: The external `provider_operation_id` has been identified. Caller may resume polling its status. (Does *not* imply the operation has completed).
  - `CONFIRMED_ABSENT`: The provider supplied authoritative evidence that the request was never received/created. Safe to retry if `attempt_number < max_attempts`.
  - `UNRESOLVED`: The provider cannot authoritatively confirm whether the submission was accepted (e.g. Veo has no token-search API and `provider_operation_id` is missing).
- **Terminal Ambiguity Handling:**
  - `UNRESOLVED` transitions the attempt to `AttemptStatus.FAILED` with `error_code="AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION"`.
  - Structured diagnostic metadata is persisted on the attempt (`provider`, `submission_token`, `provider_operation_id: null`).
  - The job transitions to `JobStatus.FAILED` with `available_at = NULL` (no automatic retry).
  - The workflow transitions to `WorkflowStatus.FAILED` with clear operator diagnostics.

---

## 3. Detailed Architectural Changes

### 3.1 Repository & Orchestration Concurrency

#### `app/repositories/workflow.py`
Add row-locking retrieval:
```python
async def get_workflow_for_update(self, workflow_id: str) -> WorkflowModel | None:
    """Retrieve workflow with an exclusive row-level lock (FOR UPDATE)."""
    stmt = (
        select(WorkflowModel)
        .where(WorkflowModel.id == workflow_id)
        .with_for_update()
    )
    result = await self._session.execute(stmt)
    return result.scalar_one_or_none()
```
In `create_workflow`, wrap the insert in `async with self._session.begin_nested():` and catch `IntegrityError` outside the nested context to look up the existing idempotency key.

#### `app/repositories/job.py`
In `create_job`, wrap the initial insert in `async with self._session.begin_nested():`:
```python
try:
    async with self._session.begin_nested():
        self._session.add(job)
        await self._session.flush()
except IntegrityError as exc:
    # Only recover if this is a uniqueness collision on (workflow_id, logical_key)
    existing = await self.get_by_logical_key(workflow_id, logical_key)
    if existing is not None:
        return existing
    raise exc
```

#### `app/orchestration/orchestrator.py`
Update `advance_workflow(workflow_id: str)`:
1. Acquire row lock via `await wf_repo.get_workflow_for_update(workflow_id)`.
2. Short-circuit if workflow status is already terminal (`COMPLETED`, `FAILED`, `CANCELLED`).
3. Query all existing jobs for the workflow.
4. For each stage check:
   - Check if downstream job already exists in `jobs_by_key`.
   - If downstream job does not exist, invoke `job_repo.create_job(...)`.
   - Update `workflow.current_stage` and `workflow.status` transactionally.
   - If any prerequisite job has failed, transition `workflow.status = WorkflowStatus.FAILED` with the corresponding `error_code` and `error_message`.
5. Commit the transaction and return the validated `Workflow` model.

---

### 3.2 Reconciliation Protocol & Recovery Worker

#### `app/orchestration/recovery.py` & `app/video/reconciler.py`
Define the reconciliation contract:
```python
class ReconciliationStatus(str, Enum):
    RESOLVED = "resolved"              # provider_operation_id found; caller resumes polling
    CONFIRMED_ABSENT = "confirmed_absent"  # Authoritative proof provider never received request
    UNRESOLVED = "unresolved"          # External status cannot be authoritatively determined

@dataclass(frozen=True)
class ReconciliationOutcome:
    status: ReconciliationStatus
    provider_operation_id: str | None = None
    error_message: str | None = None
```

#### `VeoProviderReconciler` (`app/video/reconciler.py`)
```python
async def reconcile_submission(
    self,
    submission_token: str | None,
    provider_operation_id: str | None,
) -> ReconciliationOutcome:
    if provider_operation_id:
        try:
            op = await self._video_model.get_operation_status(provider_operation_id)
            return ReconciliationOutcome(
                status=ReconciliationStatus.RESOLVED,
                provider_operation_id=op.operation_id,
            )
        except Exception as exc:
            return ReconciliationOutcome(
                status=ReconciliationStatus.UNRESOLVED,
                error_message=f"Failed querying provider operation {provider_operation_id}: {exc}",
            )

    # Veo has no client submission token query capability. Absence of operation ID is UNRESOLVED.
    return ReconciliationOutcome(
        status=ReconciliationStatus.UNRESOLVED,
        error_message=f"Submission token {submission_token} cannot be searched in Veo API without operation ID.",
    )
```

#### `RecoveryWorker` (`app/orchestration/recovery.py`)
When evaluating an expired attempt in `SUBMISSION_PENDING`:
1. Call `reconciler.reconcile_submission(...)`.
2. If `outcome.status == ReconciliationStatus.RESOLVED`:
   - Keep attempt alive: `attempt.provider_operation_id = outcome.provider_operation_id`, `attempt.status = AttemptStatus.RUNNING`, `job.status = JobStatus.RUNNING`.
3. If `outcome.status == ReconciliationStatus.CONFIRMED_ABSENT`:
   - Mark `attempt.status = AttemptStatus.EXPIRED`.
   - If `attempt_number < job.max_attempts`: reschedule as `JobStatus.PENDING`.
   - Else: mark `JobStatus.FAILED` (`MAX_ATTEMPTS_EXCEEDED`).
4. If `outcome.status == ReconciliationStatus.UNRESOLVED`:
   - Mark `attempt.status = AttemptStatus.FAILED`.
   - Persist structured attempt metadata:
     ```python
     attempt.attempt_metadata = {
         "error_code": "AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION",
         "provider": attempt.provider,
         "submission_token": attempt.submission_token,
         "provider_operation_id": attempt.provider_operation_id,
     }
     ```
   - Mark `job.status = JobStatus.FAILED`, `job.error_code = "AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION"`.
   - Do **not** reschedule job.

---

### 3.3 Package Consolidation

1. Remove directory `app/orchestrator/`:
   - Delete `app/orchestrator/state.py`
   - Delete `app/orchestrator/task.py`
   - Delete `app/orchestrator/workflow.py`
   - Delete `app/orchestrator/__init__.py`
2. Remove/replace `tests/unit/test_workflow.py` with tests for `app/orchestration/models.py`.
3. Verify via `git grep` and static checks that zero references to `app.orchestrator` remain across source code, tests, and configuration.
4. Establish `app/orchestration/` as the sole owner of all workflow, job, attempt, and scheduling logic.

---

### 3.4 Operational Boundary for Manual Reconciliation

When a job reaches `AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION`:
1. **Operator Investigation:** The operator retrieves the attempt metadata containing `submission_token` and `provider`.
2. **Provider Audit:** The operator checks the provider console/logs for operations initiated around the recorded `created_at` timestamp with matching parameters.
3. **Future Operational Resolution:**
   - If an external operation was indeed created: An operational CLI/tooling script binds the identified `provider_operation_id` to the attempt and resets the job to `PENDING` with available polling.
   - If confirmed not created: The operator resets the job to `PENDING` for a clean attempt.
   - If abandoned: The workflow remains `FAILED` or is transitioned to `CANCELLED`.

---

## 4. Verification & Testing Matrix

All tests run against real PostgreSQL 16 (`videogen_test` on port 5433). SQLite fallback is strictly prohibited.

| Test Name | File | Verified Invariant |
| :--- | :--- | :--- |
| `test_concurrent_create_same_logical_job` | `tests/integration/test_repositories_postgres.py` | Two concurrent `create_job` calls for identical `(workflow_id, logical_key)` result in exactly one row, zero `PendingRollbackError`, and both callers receiving the identical model. |
| `test_advance_workflow_concurrent_idempotency` | `tests/integration/test_orchestrator_worker.py` | 10 concurrent `advance_workflow()` calls result in 0 unique constraint exceptions, 0 duplicate jobs, and a clean workflow state. |
| `test_worker_restart_resumes_persisted_provider_operation` | `tests/integration/test_orchestrator_worker.py` | Worker A starts video job $\rightarrow$ writes `provider_operation_id` $\rightarrow$ crashes $\rightarrow$ lease expires $\rightarrow$ Worker B / RecoveryWorker reclaims $\rightarrow$ reconciles operation $\rightarrow$ polls to completion $\rightarrow$ advances workflow. |
| `test_worker_shutdown_preserves_attempt_and_blocks_stale_mutation` | `tests/integration/test_orchestrator_worker.py` | Worker A receives `stop()` during execution $\rightarrow$ attempt remains persisted $\rightarrow$ lease expires $\rightarrow$ Worker B claims with new lease token $\rightarrow$ stale Worker A call with old lease token is rejected (`LeaseConflictError`). |
| `test_recovery_does_not_retry_unresolved_submission` | `tests/integration/test_recovery_worker.py` | Crash during `SUBMISSION_PENDING` without operation ID $\rightarrow$ reconciler returns `UNRESOLVED` $\rightarrow$ attempt and job become `FAILED` $\rightarrow$ initial `attempt_number == 1`, no new attempt created $\rightarrow$ workflow becomes `FAILED`. |

---

## 5. Acceptance Criteria

- [ ] All 5 integration tests pass against PostgreSQL 16.
- [ ] Statement coverage across production modules remains $\ge 90\%$.
- [ ] Ruff formatting and linting pass with zero errors.
- [ ] No references to `app.orchestrator` remain in the repository.
- [ ] No database schema migrations required (uses existing `AttemptStatus.FAILED` and `JobStatus.FAILED` with explicit error codes).
