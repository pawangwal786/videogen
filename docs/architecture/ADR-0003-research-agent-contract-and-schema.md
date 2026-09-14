# ADR-0003: Research Agent Contract and ResearchResult Schema

## Status

Accepted

## Context

The Research Agent is the first domain agent in the VideoGen pipeline. It consumes a topic request and produces factual grounding, strategic angles, and a suggested narrative arc for downstream consumption by the Script Agent (Phase 3) and Storyboard Agent (Phase 4).

Previous video generation tools often treat research as an unstructured block of generated text or couple agents directly to cloud storage and specific external tools. A strict architectural boundary and validated domain contract are required.

## Decisions

### 1. Pure Agent Boundary (Single Responsibility)
- [ResearchAgent](file:///e:/ML%20Projects/videogen/app/agents/research/agent.py#L48) implements the [Agent](file:///e:/ML%20Projects/videogen/app/agents/base.py#L4) contract:
  ```text
  Agent[ResearchRequest, ResearchResult]
  ```
- The agent does **not** take an [ArtifactStorage](file:///e:/ML%20Projects/videogen/app/storage/base.py#L8) dependency and does not directly mutate cloud storage.
- The Orchestrator retains full ownership of persistence: it executes the agent, serializes `ResearchResult.model_dump_json(indent=2)`, uploads the resulting `research.json` artifact to `ArtifactStorage`, and tracks the resulting [ArtifactRef](file:///e:/ML%20Projects/videogen/app/artifacts/models.py#L7).

### 2. Structured Factual Grounding & Verification State
- Factual claims are modeled in [ResearchFact](file:///e:/ML%20Projects/videogen/app/agents/research/models.py#L20) with structured citations in [ResearchSource](file:///e:/ML%20Projects/videogen/app/agents/research/models.py#L8) (`title`, `url`, `publisher`, `published_at`).
- Each claim has a `verification_status` (`"source_backed"` or `"needs_verification"`). The agent prompt strictly instructs the LLM not to fabricate citations or URLs; unsupported claims must be flagged as `"needs_verification"` for future automated verification agents.

### 3. Sequential Narrative Arc & Aggregate Duration Validation
- Narrative beats are modeled in [NarrativeBeat](file:///e:/ML%20Projects/videogen/app/agents/research/models.py#L36) with an explicit `sequence` (1, 2, 3...) and `suggested_duration_ratio`.
- [ResearchResult](file:///e:/ML%20Projects/videogen/app/agents/research/models.py#L65) enforces a model validator verifying:
  1. Sequences are contiguous positive integers starting at 1.
  2. The sum of all duration ratios across the narrative arc equals approximately 1.0 (`0.98 <= sum <= 1.02`).

### 4. Zero Blind Trust in Model JSON
- LLM output is parsed through [extract_json_payload](file:///e:/ML%20Projects/videogen/app/agents/research/agent.py#L25) which handles markdown code fences, preambles, and raw text.
- Malformed JSON raises [ResearchResponseError](file:///e:/ML%20Projects/videogen/app/agents/research/errors.py#L14).
- Pydantic schema validation failures raise [ResearchValidationError](file:///e:/ML%20Projects/videogen/app/agents/research/errors.py#L18).
- Model connection/timeout/rate-limit exceptions (`ModelError`) are logged and propagated without mutation.
