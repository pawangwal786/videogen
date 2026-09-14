import json
import re
import time
from typing import Any

from pydantic import ValidationError

from app.agents.base import Agent
from app.agents.script.errors import (
    ScriptResponseError,
    ScriptValidationError,
)
from app.agents.script.models import (
    ScriptRequest,
    ScriptResult,
    ScriptScene,
    validate_scenes_against_research,
)
from app.agents.script.prompts import (
    SCRIPT_SYSTEM_PROMPT,
    build_script_user_prompt,
)
from app.logging import get_logger
from app.models.base import TextModel
from app.models.errors import ModelError

logger = get_logger(__name__)


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Extract and parse a JSON dictionary from an LLM response.

    Strips markdown code blocks, preambles, and postambles if present.
    """
    clean_text = raw_text.strip()

    # If wrapped in markdown fences e.g. ```json ... ``` or ``` ... ```
    fence_pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    fence_match = re.search(fence_pattern, clean_text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        clean_text = fence_match.group(1).strip()
    elif "{" in clean_text and "}" in clean_text:
        start = clean_text.find("{")
        end = clean_text.rfind("}")
        clean_text = clean_text[start : end + 1].strip()

    try:
        parsed = json.loads(clean_text)
        if not isinstance(parsed, dict):
            raise ScriptResponseError(
                f"Expected JSON object from model, but got {type(parsed).__name__}."
            )
        return parsed
    except json.JSONDecodeError as e:
        preview = (raw_text[:300] + "...") if len(raw_text) > 300 else raw_text
        raise ScriptResponseError(
            f"Failed to decode JSON from model response: {e}. Output preview: {preview!r}",
            cause=e,
        ) from e


class ScriptAgent(Agent[ScriptRequest, ScriptResult]):
    """Agent that generates a scene-by-scene script from research and returns a validated ScriptResult."""

    name: str = "script"

    def __init__(
        self,
        model: TextModel,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self._model = model
        self.system_prompt = system_prompt or SCRIPT_SYSTEM_PROMPT

    async def run(self, input: ScriptRequest) -> ScriptResult:
        """Execute the script agent against the configured TextModel.

        Validates narrative beat lineage, computes derived durations and full narration,
        enforces pacing tolerances, and returns an immutable ScriptResult.
        """
        start_time = time.perf_counter()
        user_prompt = build_script_user_prompt(input)

        logger.info(
            "agent.script.start",
            workflow_id=str(input.workflow_id),
            topic=input.research.topic,
            target_duration=input.target_duration_seconds,
            pacing_wpm=input.pacing_wpm,
        )

        try:
            raw_response = await self._model.generate(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
            )
        except ModelError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "agent.script.model_error",
                workflow_id=str(input.workflow_id),
                topic=input.research.topic,
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise

        # 1. Parse JSON payload safely
        payload = extract_json_payload(raw_response)

        raw_scenes = payload.get("scenes")
        if not isinstance(raw_scenes, list) or not raw_scenes:
            raise ScriptValidationError("Model response must contain a non-empty 'scenes' list.")

        title = payload.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ScriptValidationError("Model response must contain a non-empty 'title' string.")

        # 2. Derive scene durations and build ScriptScene models
        validated_scenes: list[ScriptScene] = []
        for idx, raw_scene in enumerate(raw_scenes):
            if not isinstance(raw_scene, dict):
                raise ScriptValidationError(
                    f"Scene at index {idx} must be a JSON object, got {type(raw_scene).__name__}."
                )

            narration = raw_scene.get("narration", "")
            scene_words = len(narration.split())
            derived_scene_duration = round((scene_words / input.pacing_wpm) * 60.0, 2)

            scene_dict = dict(raw_scene)
            scene_dict["estimated_duration_seconds"] = derived_scene_duration

            try:
                scene_model = ScriptScene.model_validate(scene_dict)
                validated_scenes.append(scene_model)
            except ValidationError as e:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                logger.error(
                    "agent.script.scene_validation_failed",
                    workflow_id=str(input.workflow_id),
                    scene_index=idx,
                    validation_errors=e.errors(),
                    duration_ms=round(duration_ms, 2),
                )
                raise ScriptValidationError(
                    f"Scene at index {idx} failed validation: {e}",
                    cause=e,
                ) from e

        # 3. Validate strict lineage against ResearchResult narrative arc
        try:
            validate_scenes_against_research(validated_scenes, input.research)
        except ValueError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "agent.script.lineage_validation_failed",
                workflow_id=str(input.workflow_id),
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise ScriptValidationError(
                f"Scene lineage validation failed: {e}",
                cause=e,
            ) from e

        # 4. Compute derived script metrics
        full_narration = " ".join(s.narration.strip() for s in validated_scenes)
        total_word_count = sum(len(s.narration.split()) for s in validated_scenes)
        estimated_duration_seconds = round((total_word_count / input.pacing_wpm) * 60.0, 2)

        # 5. Build and validate ScriptResult
        result_payload = {
            "workflow_id": input.workflow_id,
            "title": title.strip(),
            "target_duration_seconds": input.target_duration_seconds,
            "estimated_duration_seconds": estimated_duration_seconds,
            "total_word_count": total_word_count,
            "scenes": validated_scenes,
            "full_narration": full_narration,
            "pacing_wpm": input.pacing_wpm,
        }

        try:
            result = ScriptResult.model_validate(result_payload)
        except ValidationError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "agent.script.result_validation_failed",
                workflow_id=str(input.workflow_id),
                validation_errors=e.errors(),
                duration_ms=round(duration_ms, 2),
            )
            raise ScriptValidationError(
                f"ScriptResult validation failed: {e}",
                cause=e,
            ) from e

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "agent.script.success",
            workflow_id=str(input.workflow_id),
            title=result.title,
            scenes_count=len(result.scenes),
            total_words=result.total_word_count,
            estimated_duration=result.estimated_duration_seconds,
            duration_ms=round(duration_ms, 2),
        )
        return result
