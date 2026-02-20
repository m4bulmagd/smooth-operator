import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseKnowledgeProvider(ABC):
    """Abstract base for knowledge providers.

    Extensible for future RAG / vector-search implementations.
    """

    @abstractmethod
    async def load(self) -> None:
        """Load or refresh the knowledge data (no-op if already loaded)."""
        ...

    @abstractmethod
    async def reload(self) -> None:
        """Force re-load of knowledge data, even if already loaded."""
        ...

    @abstractmethod
    async def get_context(self) -> str:
        """Return knowledge formatted for injection into an LLM prompt."""
        ...
