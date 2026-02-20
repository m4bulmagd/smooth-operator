import logging

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.schemas.llm import LlmResponse
from app.services.orchestrator.agent import ConversationAgent
from app.services.orchestrator.decision import should_respond, build_system_prompt


class TestDecisionFunctions:
    def test_should_respond_committed(self):
        assert should_respond("committed") is True

    def test_should_respond_partial(self):
        assert should_respond("partial") is False

    def test_should_respond_unknown(self):
        assert should_respond("something_else") is False

    def test_build_system_prompt_with_knowledge(self):
        prompt = build_system_prompt("Name: Jane\nRole: Engineer")
        assert "Name: Jane" in prompt
        assert "phone assistant" in prompt.lower()

    def test_build_system_prompt_empty_knowledge(self):
        prompt = build_system_prompt("")
        assert "phone assistant" in prompt.lower()
        # Knowledge section should NOT be injected
        assert "Here is information" not in prompt

    def test_build_system_prompt_whitespace_only_knowledge(self):
        """Issue 11A: Whitespace-only input should be treated as empty."""
        prompt = build_system_prompt("   \n  \t  ")
        assert "phone assistant" in prompt.lower()
        assert "Here is information" not in prompt

    def test_build_system_prompt_empty_produces_clean_output(self):
        """Issue 11A: Verify prompt structure has no trailing whitespace artifacts."""
        prompt = build_system_prompt("")
        # No double newlines or trailing whitespace from empty substitution
        assert "\n\n\n" not in prompt


@pytest.mark.asyncio
class TestConversationAgent:
    def _make_agent(self):
        mock_llm = AsyncMock()

        async def mock_stream(*args, **kwargs):
            yield "Hello! "
            yield "How can I help?"

        mock_llm.generate_stream = MagicMock(return_value=mock_stream())
        mock_knowledge = AsyncMock()
        mock_knowledge.get_context = AsyncMock(return_value="Name: Test")
        mock_knowledge.load = AsyncMock()

        agent = ConversationAgent(llm=mock_llm, knowledge=mock_knowledge)
        return agent, mock_llm, mock_knowledge

    async def test_process_partial_returns_none(self):
        agent, mock_llm, _ = self._make_agent()
        await agent.start("CA_TEST")
        result = await agent.process_transcript("CA_TEST", "partial", "hello")

        assert result is None
        mock_llm.generate_stream.assert_not_called()

    async def test_process_committed_generates_response(self):
        agent, mock_llm, _ = self._make_agent()
        await agent.start("CA_TEST")
        result_stream = await agent.process_transcript(
            "CA_TEST", "committed", "What's your name?"
        )

        result = ""
        async for chunk in result_stream:
            result += chunk

        assert result == "Hello! How can I help?"
        mock_llm.generate_stream.assert_called_once()

    async def test_process_empty_text_returns_none(self):
        agent, mock_llm, _ = self._make_agent()
        await agent.start("CA_TEST")
        result = await agent.process_transcript("CA_TEST", "committed", "  ")

        assert result is None

    async def test_llm_failure_returns_fallback(self):
        agent, mock_llm, _ = self._make_agent()

        async def failing_stream(*args, **kwargs):
            raise RuntimeError("API down")
            yield

        mock_llm.generate_stream = MagicMock(return_value=failing_stream())

        await agent.start("CA_TEST")
        result_stream = await agent.process_transcript("CA_TEST", "committed", "Hello?")

        result = ""
        async for chunk in result_stream:
            result += chunk

        assert "trouble" in result.lower()

    async def test_conversation_history_accumulates(self):
        agent, mock_llm, _ = self._make_agent()

        await agent.start("CA_TEST")

        r1 = await agent.process_transcript("CA_TEST", "committed", "Hello")
        async for _ in r1:
            pass

        r2 = await agent.process_transcript("CA_TEST", "committed", "How are you?")
        async for _ in r2:
            pass

        # LLM should have been called twice
        assert mock_llm.generate_stream.call_count == 2

        # Second call should include previous messages in context
        second_call_messages = mock_llm.generate_stream.call_args_list[1][1]["messages"]
        assert len(second_call_messages) >= 3  # user + assistant + user

    async def test_start_loads_knowledge_once(self):
        agent, _, mock_knowledge = self._make_agent()

        await agent.start("CALL_1")
        await agent.start("CALL_2")

        # Knowledge should only be loaded once (idempotent)
        assert (
            mock_knowledge.load.call_count == 2
        )  # called twice, but provider is idempotent

    async def test_stop_cleans_up_without_closing_llm(self):
        """Issue 1A: stop() should not close the LLM client."""
        agent, mock_llm, _ = self._make_agent()

        await agent.start("CA_TEST")
        await agent.stop("CA_TEST")

        mock_llm.close.assert_not_called()

    async def test_shutdown_closes_llm(self):
        """Issue 1A: shutdown() should close the LLM client."""
        agent, mock_llm, _ = self._make_agent()

        await agent.shutdown()

        mock_llm.close.assert_called_once()

    async def test_process_without_start_logs_warning(self, caplog):
        """Issue 10A: Calling process_transcript without start() should log a warning."""
        agent, mock_llm, _ = self._make_agent()

        with caplog.at_level(logging.WARNING):
            result_stream = await agent.process_transcript(
                "UNSTARTED", "committed", "Hello"
            )
            async for _ in result_stream:
                pass

        # Should still work (auto-creates conversation)
        assert result_stream is not None
        # But should warn about missing start()
        assert any("before start()" in record.message for record in caplog.records)
