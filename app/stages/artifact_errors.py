"""Artifact infrastructure error types.

These errors represent distinct failure domains in the artifact storage
and lifecycle management layer:

- SecurityViolationError: filesystem confinement boundary violations
- ArtifactAlreadyExistsError: immutability invariant violations
- InvalidArtifactLifecycleError: lifecycle state machine violations
- FingerprintConflictError: concurrent idempotency claim conflicts
- ArtifactUnavailableError: pipeline invariant violations when a
  referenced artifact is not in AVAILABLE status
"""


class SecurityViolationError(Exception):
    """Path traversal, confinement failure, or invalid path component.

    Raised when an artifact path resolves outside the configured artifact
    root, or when a path component (extension, stage) fails validation.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ArtifactAlreadyExistsError(Exception):
    """Write attempted to an existing immutable artifact path.

    Once an artifact file is written, it is permanently immutable and
    cannot be overwritten. A new artifact_id must be used instead.
    """

    def __init__(self, path: str) -> None:
        super().__init__(f"Artifact already exists at path: {path}")
        self.path = path


class InvalidArtifactLifecycleError(Exception):
    """Disallowed artifact lifecycle state transition.

    The artifact lifecycle is monotonic:
        PENDING → AVAILABLE
        PENDING → FAILED
        AVAILABLE → SUPERSEDED

    Any other transition is structurally invalid.
    """

    def __init__(self, artifact_id: str, current_status: str, target_status: str) -> None:
        super().__init__(
            f"Invalid artifact lifecycle transition for '{artifact_id}': "
            f"'{current_status}' → '{target_status}'"
        )
        self.artifact_id = artifact_id
        self.current_status = current_status
        self.target_status = target_status


class FingerprintConflictError(Exception):
    """Concurrent fingerprint claim violation.

    Another worker has already claimed this (workflow_id, stage,
    stage_fingerprint) combination via a PENDING or AVAILABLE artifact.
    The partial unique index prevents duplicate reservations.
    """

    def __init__(self, workflow_id: str, stage: str, stage_fingerprint: str) -> None:
        super().__init__(
            f"Fingerprint conflict for workflow '{workflow_id}', "
            f"stage '{stage}', fingerprint '{stage_fingerprint[:16]}...'"
        )
        self.workflow_id = workflow_id
        self.stage = stage
        self.stage_fingerprint = stage_fingerprint


class ArtifactUnavailableError(Exception):
    """Pipeline invariant violation: referenced artifact is not AVAILABLE.

    Raised by get_available() when an upstream stage result manifest
    references an artifact that does not exist or is not in AVAILABLE
    status. This is distinct from find_by_fingerprint() returning None
    (which is a normal advisory result).
    """

    def __init__(self, artifact_id: str, reason: str) -> None:
        super().__init__(f"Artifact '{artifact_id}' is unavailable: {reason}")
        self.artifact_id = artifact_id
        self.reason = reason


__all__ = [
    "ArtifactAlreadyExistsError",
    "ArtifactUnavailableError",
    "FingerprintConflictError",
    "InvalidArtifactLifecycleError",
    "SecurityViolationError",
]
