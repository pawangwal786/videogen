# VideoGen: Autonomous Agentic Video Generation Platform

VideoGen is a production-grade, distributed autonomous video generation system. It combines multi-agent LLM reasoning (Gemini, OpenRouter) with state-of-the-art video generation models (Veo), cloud storage integration (Google Drive), and media assembly (FFmpeg).

The architecture separates an asynchronous **FastAPI Control Plane** from a resilient **Durable Worker Runtime** backed by PostgreSQL row locking, atomic recovery leases, and fenced orchestration state machines.

---

## What This Project Is NOT

To preserve architectural clarity and prevent boundary erosion, the following non-goals are strictly enforced:

- **The API does not perform video generation synchronously.** `POST /workflows` persists a durable workflow and initial job in PostgreSQL, returning `HTTP 201 Created` immediately. Content generation runs strictly asynchronously in background workers.
- **The API does not accept media uploads.** The control plane accepts only bounded JSON request bodies ($\le 1$ MiB). Binary media assets and intermediate video chunks are stored directly in configured blob/cloud storage.
- **The API does not own external provider credentials outside configuration.** Sensitive API keys and OAuth refresh tokens are managed strictly via environment configuration (`Settings`) and are never accepted as per-request user headers.
- **The Control Plane is not the media-processing runtime.** CPU/GPU-intensive FFmpeg encoding and provider polling are isolated to background worker pools.

---

## System Architecture

```text
                                       ┌────────────────────────────────┐
                                       │       Client Application       │
                                       └───────────────┬────────────────┘
                                                       │ HTTPS / JSON
                                                       ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         FastAPI Control Plane                                          │
│                                                                                                        │
│  [ CorrelationIdMiddleware ]    ──> Validates / generates clean X-Correlation-ID (^[a-zA-Z0-9._:-]{1,64}$)│
│  [ RequestSizeLimitMiddleware ] ──> ASGI body limiter: NORMAL -> REJECTED (HTTP 413, no downstream run)│
│  [ Authentication ]             ──> Constant-time token verification via AuthenticatedPrincipal hash   │
│  [ Rate Limiting ]              ──> Sliding-window limiter on POST /workflows (dynamic Retry-After)    │
│  [ Endpoints ]                  ──> /health, /ready, /workflows, /jobs, /artifacts                     │
└──────────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                               │ ACID Transaction (Serializable / Row Locks)
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PostgreSQL Durable State Store                                       │
│                                                                                                        │
│  • workflows      ──> Durable workflow status, current stage, and bounded idempotency key              │
│  • jobs           ──> Durable stage jobs (RESEARCH -> SCRIPT -> STORYBOARD -> MEDIA -> ASSEMBLY)       │
│  • job_attempts   ──> Worker leases, attempt heartbeats, provider operation IDs, fences                │
│  • artifacts      ──> Public artifact records with sanitized storage paths                             │
└───────────────────────┬─────────────────────────────────────────────────┬──────────────────────────────┘
                        │ FOR UPDATE SKIP LOCKED                          │ Fenced Leases & Heartbeats
                        ▼                                                 ▼
┌───────────────────────────────────────────────┐ ┌──────────────────────────────────────────────────────┐
│             Durable Job Workers               │ │                   Recovery Workers                   │
│                                               │ │                                                      │
│ • Claim executable jobs with atomic lease     │ │ • Scan expired attempts with FOR UPDATE SKIP LOCKED  │
│ • Execute pipeline agents & media assembly    │ │ • Reconcile external provider state out-of-band      │
│ • Periodic heartbeat renewal                  │ │ • Apply fenced DB transitions (RESOLVED / RESUBMIT)  │
│ • Detect cancellation fences before commit    │ │ • Prevent duplicate operations and orphaned attempts │
└───────────────────────────────────────────────┘ └──────────────────────────────────────────────────────┘
```

---

## Core Invariants

1. **Strict Asynchronous Boundary**: No LLM generation, video synthesis, or media transcoding occurs inside HTTP request handlers.
2. **Bounded Idempotency**:
   - `Idempotency-Key` header ($\le 255$ chars) uniquely identifies creation requests.
   - Replays with matching payloads return `HTTP 200 OK` referencing the existing workflow.
   - Replays with conflicting payloads trigger `HTTP 409 Conflict`.
   - 20 concurrent creation requests with identical keys serialize cleanly to **exactly 1 creator** (`201 Created`), **19 replays** (`200 OK`), and **1 initial research job**.
3. **Fenced Cancellation**:
   - `POST /workflows/{id}/cancel` acquires a PostgreSQL row lock on the workflow, transitions status to `CANCELLED`, and cancels active jobs.
   - Active worker attempts are fenced: any in-flight attempt attempting to commit against a cancelled workflow or job is rejected by database constraints.
4. **Crash Recovery & Reconciliation**:
   - Crashed workers leave expired leases. Recovery workers claim expired attempts via `FOR UPDATE SKIP LOCKED`.
   - External provider operations are queried out-of-band outside DB transactions.
   - Outcomes are committed using strict conditional SQL `UPDATE` queries matching `(attempt_id, recovery_worker_id, recovery_lease_token, status)`.

---

## State Machines

### Workflow Lifecycle
```text
[ PENDING ] ──> [ RUNNING ] ──> [ COMPLETED ]
     │               │
     ├───────────────┴────────> [ CANCELLED ]
     │               │
     └───────────────┴────────> [ FAILED ]
```

### Job Lifecycle
```text
[ PENDING ] ──> [ RUNNING ] ──> [ COMPLETED ]
     │               │
     ├───────────────┴────────> [ CANCELLED ]
     │               │
     └───────────────┴────────> [ FAILED ]
```

### Job Attempt Lifecycle & Lease Fencing
```text
[ SUBMISSION_PENDING ] ──> [ PROVIDER_PENDING ] ──> [ RUNNING ] ──> [ COMPLETED ]
          │                         │                     │
          ▼ (Crash / Timeout)       ▼ (Crash / Timeout)   ▼ (Crash / Timeout)
   Recovery Claimed          Provider Reconciled   Recovery Fenced
          │                         │                     │
          ▼                         ▼                     ▼
     [ RESUBMIT ]             [ RESOLVED ]          [ FAILED / CANCELLED ]
```

---

## API Specification

### Endpoints

| Method | Path | Auth Required | Description |
| :--- | :--- | :---: | :--- |
| `GET` | `/health` | No | Liveness probe (executes zero DB queries; returns `200 OK`). |
| `GET` | `/ready` | No | Readiness probe (executes `SELECT 1` on PostgreSQL; returns `200 OK` or `503 Service Unavailable`). |
| `POST` | `/workflows` | Yes | Create workflow + initial `RESEARCH` job (`201 Created` or `200 OK` for replays). |
| `GET` | `/workflows/{id}` | Yes | Retrieve workflow metadata and current stage. |
| `POST` | `/workflows/{id}/cancel` | Yes | Cancel an active workflow; fences all in-flight workers. |
| `GET` | `/workflows/{id}/jobs` | Yes | List all jobs and attempt history for a workflow. |
| `GET` | `/workflows/{id}/artifacts` | Yes | List generated artifacts projected via `PublicArtifactMetadata`. |

### Standard Error Envelope

All API errors return a consistent machine-readable envelope with correlation IDs present in both header and body:

```json
{
  "error": {
    "code": "IDEMPOTENCY_CONFLICT",
    "message": "Idempotency key 'abc' already exists with different payload.",
    "details": null,
    "correlation_id": "7585fdfd-4a1d-4009-8472-a7d7ffcc36cf"
  }
}
```

### Public Artifact Metadata Whitelist

Artifact metadata returned by `GET /workflows/{id}/artifacts` is projected through an explicit whitelist schema (`PublicArtifactMetadata`), dropping internal provider tokens and filesystem roots:

```json
{
  "id": "e6a2b845-8f6b-4e1a-8c7a-39704be7751b",
  "workflow_id": "2399eafb-2f4e-4c1f-9a5a-39704be7751b",
  "artifact_type": "video",
  "storage_path": "final_output.mp4",
  "mime_type": "video/mp4",
  "file_size_bytes": 10485760,
  "status": "available",
  "created_at": "2026-09-17T18:00:00Z",
  "artifact_metadata": {
    "duration_seconds": 60.0,
    "width": 1920,
    "height": 1080,
    "frame_rate": 30.0,
    "codec": "libx264",
    "audio_codec": "aac",
    "sample_rate": 44100,
    "bitrate_kbps": 2500,
    "format": "mp4"
  }
}
```

### Observability & Correlation ID Contract

All inbound HTTP requests are tracked and correlated across middleware, handlers, and background logs using the `X-Correlation-ID` header:
- **Whitespace Normalization**: Caller-provided correlation IDs are normalized with leading and trailing whitespace stripped (`raw.strip()`). For example, `"  req-batch-001  "` becomes `"req-batch-001"`.
- **Constrained Grammar**: The ID must strictly conform to the ASCII regex `^[a-zA-Z0-9._:-]{1,64}$` (1 to 64 alphanumeric characters, periods, underscores, colons, or hyphens).
- **Fallback Semantics**: If the header is missing, empty, whitespace-only, contains disallowed characters (e.g. control chars, spaces, unicode), or exceeds 64 characters, the server automatically discards it and generates a clean UUID4.
- **Propagation**: The normalized ID is attached to the request state, injected into the request-scoped `structlog` context for all log records, and echoed back on every HTTP response in the `X-Correlation-ID` header as well as in the standard error envelope.

---

## Rate Limiting Topology

The control plane implements an in-memory sliding-window rate limiter on `POST /workflows`.

- **Single API Process**: In-memory rate limiting enforces caller limits using SHA-256 token hashes and returns dynamic `Retry-After: <seconds>` headers on `429 Too Many Requests`.
- **Multiple API Replicas**: When horizontally scaled across multiple instances, independent in-memory state does not form a global limit. To scale horizontally, substitute a Redis-backed sliding window limiter behind the existing `InMemoryRateLimiter` abstraction.

---

## Configuration Reference

Settings are configured via environment variables or a local `.env` file using Pydantic Settings.

| Variable | Type | Default | Description / Production Constraint |
| :--- | :--- | :--- | :--- |
| `VIDEOGEN_ENV` | `str` | `development` | Environment mode (`development`, `test`, `production`). |
| `VIDEOGEN_LOG_LEVEL` | `str` | `INFO` | Logging level (`INFO`, `WARNING`, `ERROR`, `CRITICAL`). `DEBUG`/`TRACE` are rejected in production. |
| `VIDEOGEN_DATABASE_URL` | `str` | Localhost PG | Async PostgreSQL connection string. In production, loopback hosts (`localhost`, `127.0.0.1`, `[::1]`) and `postgres:postgres` default credentials are rejected. |
| `VIDEOGEN_API_AUTH_TOKEN` | `SecretStr` | `None` | Shared bearer auth token. **Mandatory** in production (fails closed at boot if missing or whitespace). |
| `API_RATE_LIMIT_PER_MINUTE` | `int` | `60` | Max workflow creation requests per client per minute ($\ge 1$). |
| `API_MAX_REQUEST_BODY_BYTES`| `int` | `1048576` | Maximum allowed request body size ($\ge 1024$ bytes; default 1 MiB). |
| `GEMINI_API_KEY` | `SecretStr` | `None` | Gemini API key. Mandatory in production if Gemini is primary or fallback provider. |
| `OPENROUTER_API_KEY` | `SecretStr` | `None` | OpenRouter API key. Mandatory in production if OpenRouter is primary or fallback provider. |
| `GOOGLE_DRIVE_CLIENT_ID` | `SecretStr` | `None` | OAuth2 Client ID. |
| `GOOGLE_DRIVE_CLIENT_SECRET`| `SecretStr` | `None` | OAuth2 Client Secret. |
| `GOOGLE_DRIVE_REFRESH_TOKEN`| `SecretStr` | `None` | OAuth2 Refresh Token. |
| `GOOGLE_DRIVE_ROOT_FOLDER_ID`| `str` | `None` | Root Folder ID. *Note*: In production, Drive settings must be all-absent (disabled) or all-present (enabled). Partial configuration is rejected. |

---

## Local Development Setup

### 1. Prerequisites
- Python 3.12+
- `uv` (fast Python package manager)
- Docker & Docker Compose
- FFmpeg and FFprobe installed and available on `PATH`

### 2. Database Setup
Start a local PostgreSQL 16 instance via Docker:

```bash
docker run -d \
  --name videogen-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=videogen \
  -p 5433:5432 \
  postgres:16-alpine
```

### 3. Installation & Migrations
Install dependencies and apply schema migrations:

```bash
# Sync virtual environment
uv sync

# Run database migrations
uv run alembic upgrade head
```

### 4. Running the Control Plane API
Start the FastAPI server via the canonical entrypoint:

```bash
uv run uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## Testing & Quality Gates

The test suite enforces full statement and branch coverage across the orchestration, persistence, and API control plane layers.

### Local Verification Commands

```bash
# Verify static linting
uv run ruff check .

# Verify code formatting
uv run ruff format --check .

# Verify bytecode compilation and import integrity
uv run python -m compileall app

# Apply database migrations
uv run alembic upgrade head

# Verify database schema parity (detects any unmigrated model changes)
uv run alembic check

# Run full test suite with branch and statement coverage (enforces >= 90.0% threshold)
uv run pytest --cov=app --cov-branch --cov-report=term-missing
```

### Automated Remote CI Pipeline

All quality gates are enforced automatically on every push and pull request to `main` via the GitHub Actions CI workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):
- Runs on `ubuntu-latest` against a dedicated **PostgreSQL 16** service container (`postgres:16-alpine`).
- Enforces frozen dependency synchronization via `uv sync --frozen --all-extras`.
- Sequentially executes Ruff linting, Ruff format check, `python -m compileall`, Alembic migrations, Alembic schema checks, and the complete Pytest suite with branch coverage enforcement ($\ge 90.0\%$).

---

## Deployment

### Recommended Deployment Command

Run the ASGI control plane using Uvicorn with worker processes configured for your host capacity:

```bash
VIDEOGEN_ENV=production \
VIDEOGEN_API_AUTH_TOKEN="your-strong-production-token" \
VIDEOGEN_DATABASE_URL="postgresql+asyncpg://app_user:strong_password@db-instance.internal:5432/videogen_prod" \
GEMINI_API_KEY="your-gemini-key" \
uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

> [!NOTE]
> **Operational Topology & Process Isolation**:
> The command above is a deployment configuration, not proof that 4-worker production operation has been benchmarked for your specific infrastructure.
> Operating with multiple Uvicorn worker processes has concrete architectural implications:
> 1. **Process Isolation**: Each Uvicorn worker runs as an independent OS process with its own Python interpreter, its own SQLAlchemy async engine, and its own database connection pool (`pool_size=10, max_overflow=5` by default). Four workers will open up to 60 concurrent database connections. Ensure your PostgreSQL instance's `max_connections` setting accommodates this total along with asynchronous background workers.
> 2. **Per-Process Rate Limiting**: The built-in in-memory rate limiter operates **per worker process** (e.g., 60 requests/minute per worker process), not globally synchronized across the cluster. If strict cluster-wide rate limiting is required across multi-worker or multi-host deployments, a shared Redis-backed rate limiter should be introduced behind the `RateLimiter` interface. For current control-plane workloads, per-process limits remain sufficient.