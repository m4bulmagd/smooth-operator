import logging

from app.schemas.llm import ChatMessage
from app.services.conversation.memory import MessageBuffer

logger = logging.getLogger(__name__)

_WARN_CONVERSATION_COUNT = 100


class ConversationManager:
    """Manages per-call conversation state — message history keyed by call_sid."""

    def __init__(self, max_messages: int = 20) -> None:
        self._max_messages = max_messages
        self._conversations: dict[str, MessageBuffer] = {}

    def get_or_create(self, call_sid: str) -> MessageBuffer:
        if call_sid not in self._conversations:
            self._conversations[call_sid] = MessageBuffer(
                max_messages=self._max_messages
            )
            logger.info("[%s] New conversation created", call_sid)

            if len(self._conversations) >= _WARN_CONVERSATION_COUNT:
                logger.warning(
                    "Active conversations (%d) exceeded threshold (%d) — "
                    "possible memory leak if conversations are not cleaned up",
                    len(self._conversations),
                    _WARN_CONVERSATION_COUNT,
                )

        return self._conversations[call_sid]

    def add_user_message(self, call_sid: str, text: str) -> None:
        buf = self.get_or_create(call_sid)
        buf.append(ChatMessage(role="user", content=text))

    def add_assistant_message(self, call_sid: str, text: str) -> None:
        buf = self.get_or_create(call_sid)
        buf.append(ChatMessage(role="assistant", content=text))

    def get_messages(self, call_sid: str) -> list[ChatMessage]:
        buf = self.get_or_create(call_sid)
        return buf.get_all()

    def remove(self, call_sid: str) -> None:
        if call_sid in self._conversations:
            del self._conversations[call_sid]
            logger.info("[%s] Conversation removed", call_sid)
