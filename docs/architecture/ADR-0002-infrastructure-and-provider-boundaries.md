# ADR-0002: Infrastructure Layer, Provider Boundaries, and Storage Abstraction

## Status

Accepted

## Context

Phase 2 establishes the production infrastructure layer connecting future VideoGen agents to external services:
1. LLM providers (Google Gemini, OpenRouter)
2. Cloud object persistence (Google Drive)

Agents must remain completely decoupled from third-party SDKs, authentication mechanisms, and network I/O details. Furthermore, failure modes must be explicitly categorized so the system can differentiate between retryable/transient network issues and permanent faults (such as authentication or configuration errors).

## Decisions

### 1. Unified Interface Boundaries
- Agents communicate exclusively with [TextModel](file:///e:/ML%20Projects/videogen/app/models/base.py#L4) via [ModelRouter](file:///e:/ML%20Projects/videogen/app/models/router.py#L22).
- Agents persist assets exclusively through [ArtifactStorage](file:///e:/ML%20Projects/videogen/app/storage/base.py#L8) using [GoogleDriveStorage](file:///e:/ML%20Projects/videogen/app/storage/google_drive.py#L26).
- No agent code may import `google.genai`, `openai`, or `googleapiclient`.

### 2. Clean Error Domain Separation
- Model errors derive from [ModelError](file:///e:/ML%20Projects/videogen/app/models/errors.py#L1):
  - [ModelConfigurationError](file:///e:/ML%20Projects/videogen/app/models/errors.py#L18) (`retryable=False`)
  - [ModelAuthenticationError](file:///e:/ML%20Projects/videogen/app/models/errors.py#L30) (`retryable=False`)
  - [ModelTimeoutError](file:///e:/ML%20Projects/videogen/app/models/errors.py#L42) (`retryable=True`)
  - [ModelRateLimitError](file:///e:/ML%20Projects/videogen/app/models/errors.py#L54) (`retryable=True`)
  - [ModelResponseError](file:///e:/ML%20Projects/videogen/app/models/errors.py#L66) (`retryable=False|True`)
- Storage errors derive from [StorageError](file:///e:/ML%20Projects/videogen/app/storage/errors.py#L1):
  - [StorageConfigurationError](file:///e:/ML%20Projects/videogen/app/storage/errors.py#L14)
  - [StorageAuthenticationError](file:///e:/ML%20Projects/videogen/app/storage/errors.py#L18)
  - [StorageNotFoundError](file:///e:/ML%20Projects/videogen/app/storage/errors.py#L22)
  - [StorageDuplicateError](file:///e:/ML%20Projects/videogen/app/storage/errors.py#L26)
  - [StorageOperationError](file:///e:/ML%20Projects/videogen/app/storage/errors.py#L30)

### 3. Conservative Fallback Policy (No Loops in Router)
- [ModelRouter](file:///e:/ML%20Projects/videogen/app/models/router.py#L22) evaluates normalized provider errors.
- If primary provider fails with a retryable transient error (`error.retryable is True`, e.g., timeout, rate limit, 5xx), the router executes the configured fallback provider once.
- If primary provider fails with a permanent error (`retryable is False`, e.g., authentication, invalid request parameters, bad model name), the router raises immediately. **Fallback is strictly forbidden on permanent errors** to prevent masking configuration bugs or silent expensive consumption.

### 4. Google Drive Asynchronous Thread Boundary
- Google Drive API (`googleapiclient`) is synchronous. All Drive SDK operations are delegated via `asyncio.to_thread` to ensure non-blocking event loop execution.
- Workflow folder structures (`AgenticVideo/workflows/<workflow_id>/<artifact_type>`) are queried and created idempotently.
- File uploads enforce an explicit overwrite policy: uploading an existing file when `overwrite=False` raises [StorageDuplicateError](file:///e:/ML%20Projects/videogen/app/storage/errors.py#L26).

### 5. Observability and Secret Sanitization
- Structured logging using `structlog` captures `provider`, `model`, `prompt_length`, `duration_ms`, `success`, and `error_type`.
- Raw prompts and credentials (`api_key`, `client_secret`, `refresh_token`) are strictly excluded from log events.
- Request correlation context (`workflow_id`, `task_id`) is supported via context variables.

### 6. Integration Test Safety
- Live external integration tests in `tests/integration/` are gated behind `VIDEOGEN_RUN_EXTERNAL_TESTS=true`.
- Default test runs (`pytest`) execute only against mocked adapters, avoiding paid API quota usage or cloud storage mutations.
