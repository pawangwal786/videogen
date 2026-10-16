"""Deterministic stage idempotency fingerprint calculator.

Implements the stage-level idempotency policy: Job Retry != Stage Recomputation.
Computes canonical SHA-256 fingerprint hashes from input artifacts and configuration.
"""

import hashlib

RESEARCH_STAGE_VERSION: str = "v1"
SCRIPT_STAGE_VERSION: str = "v1"


def compute_component_hash(*components: str | float | bytes) -> str:
    """Compute deterministic SHA-256 digest from canonicalized components."""
    hasher = hashlib.sha256()
    for item in components:
        if isinstance(item, bytes):
            hasher.update(item)
        else:
            hasher.update(str(item).strip().encode("utf-8"))
        hasher.update(b"\x00")  # Delimiter prevents component boundary collision
    return hasher.hexdigest()


class StageFingerprintCalculator:
    """Calculates deterministic idempotency fingerprints for each pipeline stage."""

    @staticmethod
    def for_research(
        topic: str,
        target_duration_seconds: int,
        model_name: str = "default",
        target_audience: str = "General audience",
        research_depth: str = "standard",
        version: str = RESEARCH_STAGE_VERSION,
    ) -> str:
        """Calculate fingerprint for the RESEARCH stage."""
        return compute_component_hash(
            "stage:research",
            version.strip(),
            topic.strip(),
            target_duration_seconds,
            model_name.strip(),
            target_audience.strip(),
            research_depth.strip().lower(),
        )

    @staticmethod
    def for_script(
        research_artifact_checksum: str,
        pacing_wpm: int = 140,
        target_duration_seconds: int | None = None,
        tone: str = "engaging, informative, and cinematic",
        call_to_action: str | None = None,
        version: str = SCRIPT_STAGE_VERSION,
    ) -> str:
        """Calculate fingerprint for the SCRIPT stage.

        Note: target_duration_seconds in ScriptJobPayload is an assertion constraint
        checked against the upstream ResearchResult. The duration is authoritatively
        determined by the research artifact and represented by research_artifact_checksum.
        Therefore, duration assertion does not vary script fingerprint identity, ensuring
        equivalent requests (e.g. omitted vs matching duration assertion) hash identically.
        """
        return compute_component_hash(
            "stage:script",
            version.strip(),
            research_artifact_checksum.strip().lower(),
            pacing_wpm,
            tone.strip(),
            call_to_action.strip() if call_to_action else "",
        )

    @staticmethod
    def for_storyboard(
        script_artifact_checksum: str,
        visual_style: str,
        aspect_ratio: str,
    ) -> str:
        """Calculate fingerprint for the STORYBOARD stage."""
        return compute_component_hash(
            "stage:storyboard",
            script_artifact_checksum,
            visual_style,
            aspect_ratio,
        )

    @staticmethod
    def for_media_synthesis(
        storyboard_scene_checksum: str,
        aspect_ratio: str,
        resolution: str,
    ) -> str:
        """Calculate fingerprint for a MEDIA_SYNTHESIS clip generation."""
        return compute_component_hash(
            "stage:media_synthesis",
            storyboard_scene_checksum,
            aspect_ratio,
            resolution,
        )

    @staticmethod
    def for_audio(
        narration_checksum: str,
        voice_id: str,
        pacing_wpm: int,
    ) -> str:
        """Calculate fingerprint for the AUDIO stage."""
        return compute_component_hash(
            "stage:audio",
            narration_checksum,
            voice_id,
            pacing_wpm,
        )

    @staticmethod
    def for_assembly(
        ordered_video_checksums: list[str],
        audio_checksum: str,
        ffmpeg_profile: str,
    ) -> str:
        """Calculate fingerprint for the ASSEMBLY stage."""
        hasher = hashlib.sha256()
        hasher.update(b"stage:assembly\x00")
        for checksum in ordered_video_checksums:
            hasher.update(checksum.strip().encode("utf-8"))
            hasher.update(b"\x00")
        hasher.update(audio_checksum.strip().encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(ffmpeg_profile.strip().encode("utf-8"))
        hasher.update(b"\x00")
        return hasher.hexdigest()

    @staticmethod
    def for_qc(
        master_video_checksum: str,
        qc_thresholds_version: str,
    ) -> str:
        """Calculate fingerprint for the QC stage."""
        return compute_component_hash(
            "stage:qc",
            master_video_checksum,
            qc_thresholds_version,
        )

    @staticmethod
    def for_publish(
        qc_master_video_checksum: str,
        destination_root_folder_id: str,
    ) -> str:
        """Calculate fingerprint for the PUBLISH stage."""
        return compute_component_hash(
            "stage:publish",
            qc_master_video_checksum,
            destination_root_folder_id,
        )


__all__ = ["StageFingerprintCalculator", "compute_component_hash"]
