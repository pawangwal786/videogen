import os

import pytest

from app.config.settings import Settings
from app.models.gemini import GeminiTextModel
from app.models.openrouter import OpenRouterTextModel

RUN_EXTERNAL = os.getenv("VIDEOGEN_RUN_EXTERNAL_TESTS", "false").lower() in (
    "true",
    "1",
)


@pytest.mark.skipif(
    not RUN_EXTERNAL,
    reason="External integration tests are disabled by default. Set VIDEOGEN_RUN_EXTERNAL_TESTS=true to enable.",
)
@pytest.mark.asyncio
async def test_live_gemini_provider():
    settings = Settings()
    api_key = settings.require_gemini()
    model = GeminiTextModel(api_key=api_key, model_name=settings.gemini_model)
    result = await model.generate("Say 'Hello, VideoGen!' and nothing else.")
    assert "Hello, VideoGen!" in result


@pytest.mark.skipif(
    not RUN_EXTERNAL,
    reason="External integration tests are disabled by default. Set VIDEOGEN_RUN_EXTERNAL_TESTS=true to enable.",
)
@pytest.mark.asyncio
async def test_live_openrouter_provider():
    settings = Settings()
    api_key, base_url, model_name = settings.require_openrouter()
    model = OpenRouterTextModel(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
    )
    result = await model.generate("Say 'Hello, VideoGen!' and nothing else.")
    assert "Hello, VideoGen!" in result
