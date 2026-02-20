import pytest
from unittest.mock import patch

from app.schemas.llm import ChatMessage
from app.services.conversation.memory import MessageBuffer
from app.services.conversation.manager import ConversationManager


class TestMessageBuffer:
    def test_append_and_get(self):
        buf = MessageBuffer(max_messages=5)
        buf.append(ChatMessage(role="user", content="hello"))
        buf.append(ChatMessage(role="assistant", content="hi"))

        msgs = buf.get_all()
        assert len(msgs) == 2
        assert msgs[0].content == "hello"
        assert msgs[1].content == "hi"

    def test_trim_drops_oldest(self):
        buf = MessageBuffer(max_messages=3)
        for i in range(5):
            buf.append(ChatMessage(role="user", content=f"msg-{i}"))

        msgs = buf.get_all()
        assert len(msgs) == 3
        assert msgs[0].content == "msg-2"
        assert msgs[2].content == "msg-4"

    def test_trim_preserves_system_messages(self):
        buf = MessageBuffer(max_messages=3)
        buf.append(ChatMessage(role="system", content="system prompt"))
        for i in range(5):
            buf.append(ChatMessage(role="user", content=f"msg-{i}"))

        msgs = buf.get_all()
        assert len(msgs) == 3
        assert msgs[0].role == "system"
        assert msgs[0].content == "system prompt"
        # The remaining 2 slots should be the newest non-system messages
        assert msgs[1].content == "msg-3"
        assert msgs[2].content == "msg-4"

    def test_clear(self):
        buf = MessageBuffer(max_messages=5)
        buf.append(ChatMessage(role="user", content="hello"))
        buf.clear()
        assert len(buf) == 0

    def test_len(self):
        buf = MessageBuffer(max_messages=5)
        assert len(buf) == 0
        buf.append(ChatMessage(role="user", content="hello"))
        assert len(buf) == 1

    def test_system_messages_exceed_max(self):
        """Issue 12A: When system messages alone exceed max_messages,
        only system messages remain and no crash occurs."""
        buf = MessageBuffer(max_messages=2)
        buf.append(ChatMessage(role="system", content="sys-1"))
        buf.append(ChatMessage(role="system", content="sys-2"))
        buf.append(ChatMessage(role="system", content="sys-3"))
        buf.append(ChatMessage(role="user", content="user-msg"))

        msgs = buf.get_all()
        # All 3 system messages kept (they exceed max), no non-system
        assert all(m.role == "system" for m in msgs)
        assert len(msgs) == 3

    def test_interleaved_ordering_after_trim(self):
        """Issue 12A: System messages are moved to front, newest non-system
        messages are kept, and relative order is preserved."""
        buf = MessageBuffer(max_messages=4)
        buf.append(ChatMessage(role="system", content="sys"))
        buf.append(ChatMessage(role="user", content="u1"))
        buf.append(ChatMessage(role="assistant", content="a1"))
        buf.append(ChatMessage(role="user", content="u2"))
        buf.append(ChatMessage(role="assistant", content="a2"))
        buf.append(ChatMessage(role="user", content="u3"))

        msgs = buf.get_all()
        assert len(msgs) == 4
        assert msgs[0].role == "system"
        assert msgs[0].content == "sys"
        # Trim happens on each append, so after 6 messages with max=4:
        # After 5th: [sys, a1, u2, a2] → After 6th: [sys, u2, a2, u3]
        assert msgs[1].content == "u2"
        assert msgs[2].content == "a2"
        assert msgs[3].content == "u3"


class TestConversationManager:
    def test_get_or_create_returns_buffer(self):
        mgr = ConversationManager(max_messages=20)
        buf = mgr.get_or_create("CA_TEST")
        assert isinstance(buf, MessageBuffer)

    def test_add_and_get_messages(self):
        mgr = ConversationManager(max_messages=20)
        mgr.add_user_message("CA_TEST", "hello")
        mgr.add_assistant_message("CA_TEST", "hi there")

        msgs = mgr.get_messages("CA_TEST")
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"

    def test_remove_cleans_up(self):
        mgr = ConversationManager(max_messages=20)
        mgr.add_user_message("CA_TEST", "hello")
        mgr.remove("CA_TEST")

        # After remove, get_messages should return empty (new buffer created)
        msgs = mgr.get_messages("CA_TEST")
        assert len(msgs) == 0

    def test_separate_conversations(self):
        mgr = ConversationManager(max_messages=20)
        mgr.add_user_message("CALL_1", "hello from call 1")
        mgr.add_user_message("CALL_2", "hello from call 2")

        msgs_1 = mgr.get_messages("CALL_1")
        msgs_2 = mgr.get_messages("CALL_2")
        assert len(msgs_1) == 1
        assert len(msgs_2) == 1
        assert msgs_1[0].content == "hello from call 1"
        assert msgs_2[0].content == "hello from call 2"
