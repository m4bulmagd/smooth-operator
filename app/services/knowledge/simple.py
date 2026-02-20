import asyncio
import logging
from pathlib import Path

import yaml

from app.core.config import settings
from app.schemas.knowledge import KnowledgeData
from app.services.knowledge.base import BaseKnowledgeProvider

logger = logging.getLogger(__name__)


class SimpleKnowledgeProvider(BaseKnowledgeProvider):
    """Loads knowledge from a YAML file and formats it for LLM prompts."""

    def __init__(self) -> None:
        self._data: KnowledgeData = KnowledgeData()
        self._file_path = Path(settings.knowledge_file_path)
        self._loaded = False

    def _load_sync(self) -> None:
        """Synchronous file I/O — called via asyncio.to_thread()."""
        if not self._file_path.exists():
            logger.warning("Knowledge file not found: %s", self._file_path)
            return

        with open(self._file_path, "r") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            logger.warning(
                "Knowledge file is not a valid YAML dict: %s", self._file_path
            )
            return

        self._data = KnowledgeData(entries=raw)
        logger.info(
            "Loaded knowledge: %d categories from %s",
            len(self._data.entries),
            self._file_path,
        )

    async def load(self) -> None:
        """Load knowledge data if not already loaded."""
        if self._loaded:
            return
        await asyncio.to_thread(self._load_sync)
        self._loaded = True

    async def reload(self) -> None:
        """Force re-load of knowledge data."""
        self._loaded = False
        await asyncio.to_thread(self._load_sync)
        self._loaded = True

    async def get_context(self) -> str:
        return self._data.format_for_prompt()
