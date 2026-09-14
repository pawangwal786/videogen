import json
import re
import time
from typing import Any

from pydantic import ValidationError

from app.agents.base import Agent
from app.agents.research.errors import (
    ResearchResponseError,
    ResearchValidationError,
)
from app.agents.research.models import ResearchRequest, ResearchResult
from app.agents.research.prompts import (
    RESEARCH_SYSTEM_PROMPT,
    build_research_user_prompt,
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
        # Fallback: extract substring between first '{' and last '}'
        start = clean_text.find("{")
        end = clean_text.rfind("}")
        clean_text = clean_text[start : end + 1].strip()

    try:
        parsed = json.loads(clean_text)
        if not isinstance(parsed, dict):
            raise ResearchResponseError(
                f"Expected JSON object from model, but got {type(parsed).__name__}."
            )
        return parsed
    except json.JSONDecodeError as e:
        preview = (raw_text[:300] + "...") if len(raw_text) > 300 else raw_text
        raise ResearchResponseError(
            f"Failed to decode JSON from model response: {e}. Output preview: {preview!r}",
            cause=e,
        ) from e


class ResearchAgent(Agent[ResearchRequest, ResearchResult]):
    """Agent that performs structured topic research and returns a validated ResearchResult."""

    name: str = "research"

    def __init__(
        self,
        model: TextModel,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self._model = model
        self.system_prompt = system_prompt or RESEARCH_SYSTEM_PROMPT

    async def run(self, input: ResearchRequest) -> ResearchResult:
        """Execute the research agent against the configured TextModel.

        Validates the input parameters, prompts the model, and validates the
        resulting JSON payload into an immutable ResearchResult domain object.
        """
        start_time = time.perf_counter()
        user_prompt = build_research_user_prompt(input)

        logger.info(
            "agent.research.start",
            workflow_id=str(input.workflow_id),
            topic=input.topic,
            research_depth=input.research_depth,
        )

        try:
            raw_response = await self._model.generate(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
            )
        except ModelError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "agent.research.model_error",
                workflow_id=str(input.workflow_id),
                topic=input.topic,
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise

        # 1. Parse JSON payload safely
        payload = extract_json_payload(raw_response)

        # 2. Inject workflow context
        payload["workflow_id"] = input.workflow_id
        if "research_depth" not in payload:
            payload["research_depth"] = input.research_depth
        if "topic" not in payload:
            payload["topic"] = input.topic
        payload["target_duration_seconds"] = input.target_duration_seconds

        # 3. Validate against Pydantic schema
        try:
            result = ResearchResult.model_validate(payload)
        except ValidationError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "agent.research.validation_failed",
                workflow_id=str(input.workflow_id),
                topic=input.topic,
                validation_errors=e.errors(),
                duration_ms=round(duration_ms, 2),
            )
            raise ResearchValidationError(
                f"ResearchResult schema validation failed: {e}",
                cause=e,
            ) from e

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "agent.research.success",
            workflow_id=str(input.workflow_id),
            topic=input.topic,
            facts_count=len(result.facts),
            narrative_beats_count=len(result.suggested_narrative_arc),
            duration_ms=round(duration_ms, 2),
        )
        return result
