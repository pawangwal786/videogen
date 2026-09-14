import json
import re
import time
from typing import Any

from pydantic import ValidationError

from app.agents.base import Agent
from app.agents.storyboard.errors import (
    StoryboardResponseError,
    StoryboardValidationError,
)
from app.agents.storyboard.models import (
    StoryboardRequest,
    StoryboardResult,
    StoryboardShot,
    validate_shots_against_script,
)
from app.agents.storyboard.prompts import (
    STORYBOARD_SYSTEM_PROMPT,
    build_storyboard_user_prompt,
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
            raise StoryboardResponseError(
                f"Expected JSON object from model, but got {type(parsed).__name__}."
            )
        return parsed
    except json.JSONDecodeError as e:
        preview = (raw_text[:300] + "...") if len(raw_text) > 300 else raw_text
        raise StoryboardResponseError(
            f"Failed to decode JSON from model response: {e}. Output preview: {preview!r}",
            cause=e,
        ) from e


class StoryboardAgent(Agent[StoryboardRequest, StoryboardResult]):
    """Agent that translates a ScriptResult into a shot-by-shot StoryboardResult."""

    name: str = "storyboard"

    def __init__(
        self,
        model: TextModel,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self._model = model
        self.system_prompt = system_prompt or STORYBOARD_SYSTEM_PROMPT

    async def run(self, input: StoryboardRequest) -> StoryboardResult:
        """Execute the storyboard agent against the configured TextModel.

        Validates shot-to-scene lineage, complete scene coverage, duration tolerance,
        and returns an immutable StoryboardResult.
        """
        start_time = time.perf_counter()
        user_prompt = build_storyboard_user_prompt(input)

        logger.info(
            "agent.storyboard.start",
            workflow_id=str(input.workflow_id),
            title=input.script.title,
            target_duration=input.target_duration_seconds,
            aspect_ratio=input.aspect_ratio,
            scenes_count=len(input.script.scenes),
        )

        try:
            raw_response = await self._model.generate(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
            )
        except ModelError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "agent.storyboard.model_error",
                workflow_id=str(input.workflow_id),
                title=input.script.title,
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise

        # 1. Parse JSON payload safely
        payload = extract_json_payload(raw_response)

        raw_shots = payload.get("shots")
        if not isinstance(raw_shots, list) or not raw_shots:
            raise StoryboardValidationError("Model response must contain a non-empty 'shots' list.")

        # 2. Parse and validate each StoryboardShot model
        validated_shots: list[StoryboardShot] = []
        for idx, raw_shot in enumerate(raw_shots):
            if not isinstance(raw_shot, dict):
                raise StoryboardValidationError(
                    f"Shot at index {idx} must be a JSON object, got {type(raw_shot).__name__}."
                )

            try:
                shot_model = StoryboardShot.model_validate(raw_shot)
                validated_shots.append(shot_model)
            except ValidationError as e:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                logger.error(
                    "agent.storyboard.shot_validation_failed",
                    workflow_id=str(input.workflow_id),
                    shot_index=idx,
                    validation_errors=e.errors(),
                    duration_ms=round(duration_ms, 2),
                )
                raise StoryboardValidationError(
                    f"Shot at index {idx} failed validation: {e}",
                    cause=e,
                ) from e

        # 3. Validate shot-to-scene lineage, complete scene coverage, and per-scene duration tolerance
        try:
            validate_shots_against_script(validated_shots, input.script)
        except ValueError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "agent.storyboard.lineage_validation_failed",
                workflow_id=str(input.workflow_id),
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise StoryboardValidationError(
                f"Storyboard lineage validation failed: {e}",
                cause=e,
            ) from e

        # 4. Compute aggregate metrics
        total_shots = len(validated_shots)
        estimated_duration_seconds = round(
            sum(shot.estimated_duration_seconds for shot in validated_shots), 2
        )

        # 5. Build and validate StoryboardResult
        result_payload = {
            "workflow_id": input.workflow_id,
            "title": input.script.title,
            "aspect_ratio": input.aspect_ratio,
            "visual_style": input.visual_style,
            "target_duration_seconds": input.target_duration_seconds,
            "estimated_duration_seconds": estimated_duration_seconds,
            "total_shots": total_shots,
            "shots": validated_shots,
        }

        try:
            result = StoryboardResult.model_validate(result_payload)
        except ValidationError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "agent.storyboard.result_validation_failed",
                workflow_id=str(input.workflow_id),
                validation_errors=e.errors(),
                duration_ms=round(duration_ms, 2),
            )
            raise StoryboardValidationError(
                f"StoryboardResult validation failed: {e}",
                cause=e,
            ) from e

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "agent.storyboard.success",
            workflow_id=str(input.workflow_id),
            title=result.title,
            total_shots=result.total_shots,
            estimated_duration=result.estimated_duration_seconds,
            duration_ms=round(duration_ms, 2),
        )
        return result
