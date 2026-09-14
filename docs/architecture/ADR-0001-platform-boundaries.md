# ADR-0001: VideoGen Platform Boundaries and MoneyPrinterTurbo Integration

## Status

Accepted

## Context

VideoGen is an agentic AI video generation platform designed for automated, robust video generation workflows. MoneyPrinterTurbo (MPT) is an existing open-source project providing video generation capabilities, including text-to-speech, subtitle generation, video material harvesting, and FFmpeg assembly.

Directly coupling VideoGen to MPT's monolithic structure or adopting its WebUI/orchestration logic would introduce architectural fragility, tight coupling, and maintenance overhead. Conversely, re-implementing media processing, FFmpeg operations, and TTS integrations from scratch would be redundant and wasteful.

A clear architectural boundary must be established from Phase 1 to govern responsibilities between VideoGen and MoneyPrinterTurbo.

## Decision

We establish a strict boundary between VideoGen core and external media generation engines:

### VideoGen Responsibilities (Owns)
- **Orchestration**: Workflow scheduling, DAG dependency management, retry and recovery mechanisms.
- **Workflow State**: Domain state machine, task lifecycles, and execution tracking.
- **Task Execution**: Discrete task dispatching across specialized agents.
- **Agent Contracts**: Abstract interfaces for research, scriptwriting, storyboard, and quality review agents.
- **Model Routing**: Dynamic routing policies between LLM providers (e.g., Gemini as primary, OpenRouter as fallback).
- **Artifact Metadata**: Tracking generated assets, mime types, versions, and lineage via domain models (`ArtifactRef`).
- **Google Drive Storage Abstraction**: Clean storage protocols separating local file handling from persistent cloud artifact storage.

### MoneyPrinterTurbo Responsibilities (Owns / Reuses)
- **Media Processing**: Video clip cutting, aspect ratio conversion, and visual filters.
- **FFmpeg Integration**: Low-level video rendering, encoding, and compositing pipelines.
- **TTS Integrations**: Voice synthesis providers where appropriate.
- **Subtitle Processing**: Subtitle generation, alignment, and styling.
- **Video Assembly Utilities**: Compositing audio, video clips, and subtitles into final video containers.

### Architectural Invariants (What VideoGen Does NOT Do)
- **VideoGen does NOT duplicate MPT implementations unnecessarily**: Existing media utilities and assembly routines are wrapped, not re-authored.
- **VideoGen does NOT import arbitrary MPT internals throughout the application**: All interactions with MPT must pass through explicit adapter interfaces (`app.mpt.adapter`).
- **VideoGen does NOT use MPT's WebUI as its orchestration layer**: Workflow state, execution, and user interfaces are decoupled from MPT's internal server/web UI.

## Consequences

- **Isolation**: VideoGen's core orchestration and agent logic remain clean, independently testable, and provider-agnostic.
- **Flexibility**: The media rendering engine can be swapped, upgraded, or replaced without modifying agent contracts or workflow definitions.
- **Evolution**: Phase 1 establishes contracts and protocols; concrete MPT adapters will be introduced deliberately in Phase 6 without premature integration.
