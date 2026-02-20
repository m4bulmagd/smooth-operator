import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from app.schemas.llm import ChatMessage, LlmResponse

logger = logging.getLogger(__name__)


class BaseLlmClient(ABC):
    """Abstract base for LLM providers — mirrors the STT pattern."""

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        system_prompt: Optional[str] = None,
    ) -> LlmResponse:
        """Generate a complete response from the conversation messages."""
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[ChatMessage],
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens. Used by TTS in Chapter 3."""
        ...
        # Make it a valid async generator for type-checking
        yield ""  # pragma: no cover

    async def close(self) -> None:
        """Override to release provider-specific resources."""
        pass
