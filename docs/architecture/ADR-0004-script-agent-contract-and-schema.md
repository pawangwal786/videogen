# ADR-0004: Script Agent Contract and Schema

## Status

Accepted

## Context

The Script Agent is the second domain agent in the VideoGen pipeline, establishing the first typed agent-to-agent pipeline:

```text
ResearchResult
      │
      ▼
ScriptRequest
      │
      ▼
ScriptAgent
      │
      ▼
ScriptResult
      │
      ▼
StoryboardAgent (Phase 4)
```

Direct LLM outputs frequently suffer from hallucinated metadata, inaccurate word counts, mismatched timings, and lost lineage when passed downstream. This ADR formalizes the architectural boundaries, deterministic derivation rules, and lineage invariants that govern script generation.

## Decisions

### 1. Pure Agent Boundary (In-Memory Transformation)
- [ScriptAgent](file:///e:/ML%20Projects/videogen/app/agents/script/agent.py) implements the [Agent](file:///e:/ML%20Projects/videogen/app/agents/base.py) contract:
  ```text
  Agent[ScriptRequest, ScriptResult]
  ```
- The agent does **not** take a storage or external pipeline dependency. It runs purely in-memory and returns an immutable [ScriptResult](file:///e:/ML%20Projects/videogen/app/agents/script/models.py).
- Downstream persistence (`script.json`) and storage uploading are owned exclusively by the Orchestrator.

### 2. Duration Inheritance Without Duplication
- `target_duration_seconds` is held on [ResearchResult](file:///e:/ML%20Projects/videogen/app/agents/research/models.py) where the original workflow specification is established.
- [ScriptRequest](file:///e:/ML%20Projects/videogen/app/agents/script/models.py) does not define an independent `target_duration_seconds` parameter, eliminating the possibility of conflicting durations across pipeline stages. It exposes a read-only property delegating to `research.target_duration_seconds`.

### 3. Workflow Lineage Invariant
- [ScriptRequest](file:///e:/ML%20Projects/videogen/app/agents/script/models.py) enforces that its `workflow_id` matches `research.workflow_id`.
- If `workflow_id` is omitted in `ScriptRequest`, it is automatically populated from `research.workflow_id`.
- If an explicit conflicting `workflow_id` is provided, model validation rejects it immediately.

### 4. Strict Scene-to-Narrative Beat Lineage & Complete Coverage
- Every [ScriptScene](file:///e:/ML%20Projects/videogen/app/agents/script/models.py) must reference an existing beat sequence in `research.suggested_narrative_arc` via `narrative_beat_sequence`.
- The scene's `beat_type` must match the referenced research beat's `beat_type`.
- Every narrative beat in the research must be represented by at least one scene; multiple scenes may expand a single beat (`scenes >= beats`).
- Scenes must be 1-indexed and strictly sequential (1, 2, 3, ...).
- This ensures full traceability and guarantees no research narrative beats are omitted.

### 5. Deterministic Derived Metadata (Never Trust LLM for Math)
The LLM is prompted only for creative elements: `title`, `scenes` (`narration`, `visual_direction`, `target_keywords`). All mathematical and aggregate values are calculated in application code:
- **Scene Duration**: `round((len(narration.split()) / pacing_wpm) * 60.0, 2)`
- **Total Word Count**: `sum(len(scene.narration.split()) for scene in scenes)`
- **Full Narration**: `" ".join(scene.narration.strip() for scene in scenes)`
- **Estimated Duration**: `round((total_word_count / pacing_wpm) * 60.0, 2)`

### 6. Pacing & Duration Tolerance Guard
- [ScriptResult](file:///e:/ML%20Projects/videogen/app/agents/script/models.py) validates that `estimated_duration_seconds` is within `target_duration_seconds ± 15%` using authoritative constant `DURATION_TOLERANCE_RATIO = 0.15`.
- Scripts with pacing deviating beyond this threshold raise [ScriptValidationError](file:///e:/ML%20Projects/videogen/app/agents/script/errors.py).

### 7. Creative Retrieval Keywords
- `target_keywords` on each scene are enforced to 2–5 keywords (`min_length=2, max_length=5`) and explicitly treated as creative search hints for subsequent B-roll and video generation prompts, decoupled from factual citations (`ResearchSource`).

### 8. Error Hierarchy & Infrastructure Boundary
- JSON syntax or structure errors raise [ScriptResponseError](file:///e:/ML%20Projects/videogen/app/agents/script/errors.py).
- Schema, lineage, or duration tolerance violations raise [ScriptValidationError](file:///e:/ML%20Projects/videogen/app/agents/script/errors.py).
- Infrastructure failures (`ModelError`) are preserved and propagated without masking.
