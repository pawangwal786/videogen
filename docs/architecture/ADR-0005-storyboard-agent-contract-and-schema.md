# ADR-0005: Storyboard Agent Contract, Shot Schema, and Lineage Invariants

## Status

Accepted

## Context

Phase 4 establishes the Storyboard Agent in the VideoGen pipeline. Positioned directly between the Script Agent (Phase 3) and the downstream Video Generation Layer (Phase 5), the Storyboard Agent translates timed narrative scenes into discrete, visually concrete shot definitions with model-ready video generation prompts:

```text
ResearchResult
      │
      ▼
 ScriptResult
      │
      ▼
StoryboardRequest
      │
      ▼
StoryboardAgent (Phase 4)
      │
      ▼
StoryboardResult
      │
      ▼
 VideoModel / Veo (Phase 5)
```

Direct text-to-video generation without a structured storyboard phase leads to visual discontinuity, arbitrary shot pacing, inconsistent framing, and hallucinated visual styles. Furthermore, video models require concrete visual descriptions (composition, lighting, physical motion, optics) rather than narrative abstractions or voiceover prose.

## Decisions

### 1. Pure Agent Boundary (In-Memory Transformation)
- `StoryboardAgent` implements the standard `Agent` contract:
  ```text
  Agent[StoryboardRequest, StoryboardResult]
  ```
- The agent does **not** take a storage or external pipeline dependency. It runs purely in-memory and returns an immutable `StoryboardResult`.
- The agent is decoupled from specific video model implementations (such as Google Veo, Runway, or open-source diffusion models). It produces a provider-neutral storyboard contract whose `video_prompt` is suitable for advanced video diffusion models.
- The actual `VideoModel` abstraction and concrete `VeoAdapter` remain strictly Phase 5 concerns.

### 2. Workflow Lineage and Duration Inheritance
- `StoryboardRequest` enforces that its `workflow_id` matches `script.workflow_id` (auto-populated if omitted).
- `target_duration_seconds` is inherited directly from `script.target_duration_seconds`, maintaining a single source of truth across the pipeline.
- Every `StoryboardShot` references a valid `scene_number` in `script.scenes`.
- **Complete Scene Coverage Invariant**: Every scene in `script.scenes` must be represented by at least one storyboard shot (`shots >= scenes`). Multiple shots may break down a single scene (e.g., establishing shot followed by close-up).

### 3. Centralized Duration Tolerance (Non-Brittle Decimal Arithmetic)
To avoid brittle validation from floating-point rounding in LLM generation, duration validation employs two layers of tolerance:
1. **Total Duration Tolerance**: The cumulative storyboard duration `estimated_duration_seconds` (sum of all shot durations) must fall within `target_duration_seconds ± 15%` (`DURATION_TOLERANCE_RATIO = 0.15`).
2. **Per-Scene Duration Tolerance**: For each scene $s$, the sum of its constituent shot durations must align with `s.estimated_duration_seconds` within a defined margin (`SCENE_DURATION_TOLERANCE_RATIO = 0.25` or 1.5 seconds minimum slack).

### 4. Structured Shot Framing and Camera Movement
To ensure cinematic rigor and prevent arbitrary prompt phrasing, shot composition uses standardized vocabulary:
- **`ShotFraming`**: `extreme_close_up`, `close_up`, `medium_close_up`, `medium_shot`, `cowboy_shot`, `wide_shot`, `extreme_wide_shot`.
- **`CameraMovement`**: `static`, `pan_left`, `pan_right`, `tilt_up`, `tilt_down`, `zoom_in`, `zoom_out`, `tracking`, `drone_aerial`, `orbit`.

### 5. Video Prompt Optimization
Each shot includes a dedicated `video_prompt` and optional `negative_prompt`:
- `video_prompt`: Detailed visual prompt specifying subject appearance, camera framing, lighting, physical motion, environment, and visual mood. Narrative voiceover and conversational instructions are strictly separated.
- `negative_prompt`: Unwanted artifacts to prevent (e.g. text overlays, watermarks, blur, distorted anatomy).

### 6. Error Hierarchy & Infrastructure Boundary
- Parsing or JSON formatting errors raise `StoryboardResponseError`.
- Lineage, scene coverage, or duration tolerance violations raise `StoryboardValidationError`.
- Infrastructure failures (`ModelError`) are preserved and propagated without masking.

## Consequences

- **Separation of Concerns**: Scriptwriting focuses on voiceover pacing and narrative structure; storyboarding focuses on cinematic staging and generative optics.
- **Provider Portability**: Downstream video generation models can consume `StoryboardResult` without modifying prompt generation logic.
- **Deterministic Pipeline State**: Every shot is strictly linked to a script scene and narrative beat, enabling granular retries of single shots if video generation fails.
