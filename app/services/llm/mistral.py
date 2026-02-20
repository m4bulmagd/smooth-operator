import logging
from typing import AsyncIterator, Optional

from mistralai import Mistral

from app.core.config import settings
from app.schemas.llm import ChatMessage, LlmResponse
from app.services.llm.base import BaseLlmClient

logger = logging.getLogger(__name__)


def _to_mistral_messages(
    messages: list[ChatMessage],
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """Convert ChatMessages to Mistral's message format.

    Prepend system prompt as first message if provided.
    """
    result: list[dict] = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})

    for msg in messages:
        if msg.role == "system":
            continue  # skip — we inject system_prompt above
        result.append({"role": msg.role, "content": msg.content})
    return result


class MistralLlmClient(BaseLlmClient):
    def __init__(self) -> None:
        self._client = Mistral(api_key=settings.mistral_api_key)

    async def generate(
        self,
        messages: list[ChatMessage],
        system_prompt: Optional[str] = None,
    ) -> LlmResponse:
        mistral_msgs = _to_mistral_messages(messages, system_prompt)

        response = await self._client.chat.complete_async(
            model=settings.mistral_llm_model,
            messages=mistral_msgs,
        )

        content = ""
        if response and response.choices:
            content = response.choices[0].message.content or ""

        return LlmResponse(
            content=content,
            model=settings.mistral_llm_model,
            prompt_tokens=(
                getattr(response.usage, "prompt_tokens", None) if response else None
            ),
            completion_tokens=(
                getattr(response.usage, "completion_tokens", None) if response else None
            ),
        )

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        mistral_msgs = _to_mistral_messages(messages, system_prompt)

        response = await self._client.chat.stream_async(
            model=settings.mistral_llm_model,
            messages=mistral_msgs,
        )

        async for event in response:
            if event.data and event.data.choices:
                delta = event.data.choices[0].delta
                if delta and delta.content:
                    yield delta.content

    async def close(self) -> None:
        pass  # Mistral SDK manages its own connection lifecycle
