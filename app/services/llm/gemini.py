import logging
from typing import AsyncIterator, Optional

from google import genai
from google.genai import types

from app.core.config import settings
from app.schemas.llm import ChatMessage, LlmResponse
from app.services.llm.base import BaseLlmClient

logger = logging.getLogger(__name__)


def _to_gemini_contents(messages: list[ChatMessage]) -> list[types.Content]:
    """Convert our ChatMessage list to Gemini Content objects.

    Gemini uses 'user' and 'model' roles (not 'assistant').
    System messages are handled separately via system_instruction.
    """
    contents: list[types.Content] = []
    for msg in messages:
        if msg.role == "system":
            continue  # handled via system_instruction config
        role = "model" if msg.role == "assistant" else "user"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg.content)])
        )
    return contents


class GeminiLlmClient(BaseLlmClient):
    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)

    async def generate(
        self,
        messages: list[ChatMessage],
        system_prompt: Optional[str] = None,
    ) -> LlmResponse:
        contents = _to_gemini_contents(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
        )

        response = await self._client.aio.models.generate_content(
            model=settings.gemini_llm_model,
            contents=contents,
            config=config,
        )

        return LlmResponse(
            content=response.text or "",
            model=settings.gemini_llm_model,
            prompt_tokens=getattr(response.usage_metadata, "prompt_token_count", None),
            completion_tokens=getattr(
                response.usage_metadata, "candidates_token_count", None
            ),
        )

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        contents = _to_gemini_contents(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
        )

        async for chunk in await self._client.aio.models.generate_content_stream(
            model=settings.gemini_llm_model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

    async def close(self) -> None:
        self._client.close()
