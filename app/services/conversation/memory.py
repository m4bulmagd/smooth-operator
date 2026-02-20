from app.schemas.llm import ChatMessage


class MessageBuffer:
    """Fixed-size message buffer that drops oldest messages when over capacity."""

    def __init__(self, max_messages: int = 20) -> None:
        self._max = max_messages
        self._messages: list[ChatMessage] = []

    def append(self, msg: ChatMessage) -> None:
        self._messages.append(msg)
        self._trim()

    def get_all(self) -> list[ChatMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def _trim(self) -> None:
        """Keep system messages at front, drop oldest non-system messages."""
        if len(self._messages) <= self._max:
            return

        system = [m for m in self._messages if m.role == "system"]
        non_system = [m for m in self._messages if m.role != "system"]

        keep_count = max(0, self._max - len(system))
        self._messages = (
            system + non_system[-keep_count:] if keep_count else list(system)
        )
