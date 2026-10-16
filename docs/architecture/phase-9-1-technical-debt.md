# Phase 9.1 Technical Debt & Deferred Risk Register

This document records the deferred findings, correctness risks, and technical debt from Phase 9.1 (Artifact Infrastructure & Confinement) that are frozen and tracked for remediation during Phase 9.5 (Crash & Recovery Testing) and subsequent production-hardening milestones.

---

## 1. Tracked Technical Debt Items

| ID | Finding | Priority | Severity | Impact Analysis | Remediation Trigger | Tracking & Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TD-9.1-01** | Stale worker can publish an orphan file after lease expiry | High | High | If a worker's lease expires and recovery marks the artifact `FAILED`, a resumed stale worker can still write to an unused canonical path. The database rejects finalization, but an unreferenced orphan file remains on disk. | Phase 9.5 Crash & Recovery Testing (Publication Protocol Redesign) | **Constraint:** Phase 9.2 and subsequent phases MUST NOT introduce code that assumes database lease fencing fences filesystem writes. Quarantine detection is implemented as a runtime mitigation. |
| **TD-9.1-02** | Database and filesystem publication are not atomic | High | High | A database transaction cannot atomically commit a physical filesystem link or rename. Crashes occurring between filesystem write and database commit create divergent state that requires lease-aware recovery to resolve. | Phase 9.5 Crash & Recovery Testing (Two-phase commit / Attempt-staged manifests) | Reconciler checks checksum against DB before transitioning state; uncommitted writes must remain quarantined or reclaimed by recovery. |
| **TD-9.1-03** | Repository-wide lock hierarchy audit across future models | Medium | Medium | While the existing hierarchy ($\text{Workflow} \to \text{Job} \to \text{JobAttempt} \to \text{Artifact}$) is verified and tested against PostgreSQL 16, adding new repositories or cross-table queries could introduce deadlock cycles if the partial order is not maintained. | Pre-production Concurrency Hardening Pass | Any new multi-row transaction must be audited against the global lock inventory before merging. |
| **TD-9.1-04** | Physical cleanup failures and orphan temporary files | Medium | Medium | Streaming write/read error paths catch and log `os.unlink` and `os.close` failures while releasing semaphore permits. However, an unlinked temporary file is not guaranteed to be removed if the process or disk encounters catastrophic I/O faults. | Background Storage Janitor Daemon Implementation | `ArtifactStore.cleanup_failure_count` metric is exposed for observability and alarming. |
| **TD-9.1-05** | Additional adversarial race-condition regression tests | High | High | Existing integration tests cover standard concurrency, lease expiration, and cancellation. Adversarial scenarios involving sub-millisecond thread scheduling around executor shutdown remain to be expanded. | Phase 9.5 Crash & Recovery Testing | Add automated chaos and timing-injection regression tests alongside recovery test suite. |

---

## 2. Phase 9.2 Architectural Guardrails

To prevent technical debt from propagating into the pipeline runtime integration:
1. **No Filesystem Fencing Assumptions**: `JobWorker` and `StageHandler` runtime must never assume that acquiring a database lock prevents a rogue or stale thread from performing filesystem I/O.
2. **Explicit Quarantine & Verification**: When consuming artifacts from durable storage, callers must verify physical file existence and checksums through `StageArtifactRepository.get_available(verify_integrity=True)`.
3. **Idempotent Artifact Persistence**: All stage outputs must be written through `ArtifactStore` and finalized through `StageArtifactRepository.finalize()` with mandatory `worker_id` and `lease_token`.
