"""Phase 9 pipeline stages, contracts, DTOs, and capability providers."""

from app.stages.artifact_errors import (
    ArtifactAlreadyExistsError,
    ArtifactUnavailableError,
    FingerprintConflictError,
    InvalidArtifactLifecycleError,
    SecurityViolationError,
)
from app.stages.artifact_store import ArtifactStore
from app.stages.base import StageHandler
from app.stages.context import StageContext
from app.stages.dto import (
    ArtifactKind,
    AssemblyResult,
    AudioClipArtifact,
    AudioStageResult,
    MediaSynthesisResult,
    PipelineStage,
    PublishResult,
    QCMetrics,
    QCResult,
    ResearchResult,
    ScriptResult,
    StoryboardResult,
    VideoClipArtifact,
)
from app.stages.fingerprint import (
    RESEARCH_STAGE_VERSION,
    SCRIPT_STAGE_VERSION,
    StageFingerprintCalculator,
)
from app.stages.providers import (
    StorageProvider,
    StorageResult,
    StorageUploadRequest,
    TextGenerationProvider,
    TextGenerationRequest,
    TextGenerationResult,
    TextToSpeechProvider,
    TTSRequest,
    TTSResult,
    VideoGenerationProvider,
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoReconciliationResult,
    VideoSubmission,
)
from app.stages.research import ResearchJobPayload, ResearchStageHandler
from app.stages.runtime import StageExecutionResult, StageRuntime, create_stage_context
from app.stages.script import ScriptJobPayload, ScriptStageHandler
from app.stages.stage_artifact_repository import StageArtifactRepository

__all__ = [
    "RESEARCH_STAGE_VERSION",
    "SCRIPT_STAGE_VERSION",
    "ArtifactAlreadyExistsError",
    "ArtifactKind",
    "ArtifactStore",
    "ArtifactUnavailableError",
    "AssemblyResult",
    "AudioClipArtifact",
    "AudioStageResult",
    "FingerprintConflictError",
    "InvalidArtifactLifecycleError",
    "MediaSynthesisResult",
    "PipelineStage",
    "PublishResult",
    "QCMetrics",
    "QCResult",
    "ResearchJobPayload",
    "ResearchResult",
    "ResearchStageHandler",
    "ScriptJobPayload",
    "ScriptResult",
    "ScriptStageHandler",
    "SecurityViolationError",
    "StageArtifactRepository",
    "StageContext",
    "StageExecutionResult",
    "StageFingerprintCalculator",
    "StageHandler",
    "StageRuntime",
    "StorageProvider",
    "StorageResult",
    "StorageUploadRequest",
    "StoryboardResult",
    "TTSRequest",
    "TTSResult",
    "TextGenerationProvider",
    "TextGenerationRequest",
    "TextGenerationResult",
    "TextToSpeechProvider",
    "VideoClipArtifact",
    "VideoGenerationProvider",
    "VideoGenerationRequest",
    "VideoGenerationResult",
    "VideoReconciliationResult",
    "VideoSubmission",
    "create_stage_context",
]
