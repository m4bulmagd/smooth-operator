import logging
from typing import AsyncIterator

from app.core.config import settings
from app.services.conversation.manager import ConversationManager
from app.services.knowledge.base import BaseKnowledgeProvider
from app.services.knowledge.simple import SimpleKnowledgeProvider
from app.services.llm.base import BaseLlmClient
from app.services.llm.factory import LlmFactory
from app.services.orchestrator.decision import build_system_prompt, should_respond

logger = logging.getLogger(__name__)


class ConversationAgent:
    """Main orchestrator — ties STT transcripts to LLM responses.

    Lifecycle per call:
        agent.start(call_sid)        — initialise state
        agent.process_transcript()   — called for each STT event
        agent.stop(call_sid)         — remove conversation state for a call

    When the agent itself is no longer needed:
        agent.shutdown()             — release underlying resources (LLM client)
    """

    def __init__(
        self,
        llm: BaseLlmClient | None = None,
        knowledge: BaseKnowledgeProvider | None = None,
    ) -> None:
        self._llm = llm or LlmFactory.get_client()
        self._knowledge = knowledge or SimpleKnowledgeProvider()
        self._conversations = ConversationManager(
            max_messages=settings.max_conversation_history,
        )
        self._system_prompts: dict[str, str] = {}

    async def start(self, call_sid: str) -> None:
        """Initialise conversation for a new call."""
        self._conversations.get_or_create(call_sid)

        # Load knowledge (idempotent — provider tracks its own loaded state)
        await self._knowledge.load()

        # Cache the system prompt for this call
        knowledge_context = await self._knowledge.get_context()
        self._system_prompts[call_sid] = build_system_prompt(knowledge_context)

        logger.info("[%s] Conversation agent started", call_sid)

    async def process_transcript(
        self, call_sid: str, kind: str, text: str
    ) -> AsyncIterator[str] | None:
        """Process an incoming STT transcript.

        Returns an async iterator of LLM response chunks, or None.
        """
        if not should_respond(kind):
            return None

        if not text.strip():
            return None

        # Warn if start() was never called for this call_sid
        if call_sid not in self._system_prompts:
            logger.warning(
                "[%s] process_transcript called before start() — "
                "knowledge context may be missing",
                call_sid,
            )

        # Add user message to conversation history
        self._conversations.add_user_message(call_sid, text)

        # Generate and return the response implicitly (as an async generator)
        return self._generate_response(call_sid)

    async def _generate_response(self, call_sid: str) -> AsyncIterator[str]:
        """Assemble context and call the LLM, streaming output."""
        messages = self._conversations.get_messages(call_sid)
        system_prompt = self._system_prompts.get(call_sid, build_system_prompt(""))

        full_response = ""
        try:
            async for chunk in self._llm.generate_stream(
                messages=messages,
                system_prompt=system_prompt,
            ):
                full_response += chunk
                yield chunk
        except Exception:
            logger.exception("[%s] LLM generation failed", call_sid)
            fallback = "I'm sorry, I'm having trouble processing that right now."
            full_response += fallback
            yield fallback

        # Store the assistant's reply in conversation history
        self._conversations.add_assistant_message(call_sid, full_response)
        logger.info("[%s] ASSISTANT completion stored", call_sid)

    async def stop(self, call_sid: str) -> None:
        """Remove conversation state for a single call."""
        self._conversations.remove(call_sid)
        self._system_prompts.pop(call_sid, None)
        logger.info("[%s] Conversation stopped", call_sid)

    async def shutdown(self) -> None:
        """Release underlying resources (LLM client, etc.)."""
        await self._llm.close()
        logger.info("Conversation agent shut down")
