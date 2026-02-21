import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from app.services.orchestrator.session import CallSession


@pytest.fixture
def mock_deps():
    with (
        patch("app.services.orchestrator.session.ConversationAgent") as mock_agent_cls,
        patch("app.services.orchestrator.session.SttFactory") as mock_stt_factory,
    ):

        mock_stt = mock_stt_factory.get_client.return_value
        mock_stt.run_receive_loop = AsyncMock(return_value=None)
        mock_stt.send_audio_base64 = AsyncMock()
        mock_stt.close = AsyncMock()

        mock_agent = mock_agent_cls.return_value
        mock_agent.start = AsyncMock()
        mock_agent.stop = AsyncMock()
        mock_agent.shutdown = AsyncMock()
        mock_agent.process_transcript = AsyncMock()

        yield mock_stt_factory, mock_agent_cls


@pytest.mark.asyncio
class TestCallSession:
    async def test_session_start_initializes_components(self, mock_deps):
        mock_stt_factory, mock_agent_cls = mock_deps
        mock_stt = mock_stt_factory.get_client.return_value
        mock_agent = mock_agent_cls.return_value

        session = CallSession("sid_123")
        await session.start()

        mock_stt_factory.get_client.assert_called_once()
        mock_stt.set_on_transcript.assert_called_once_with(session._on_transcript)
        assert session.recv_task is not None
        mock_agent.start.assert_awaited_once_with("sid_123")

        await session.shutdown()

    async def test_session_process_audio_sends_to_stt(self, mock_deps):
        mock_stt_factory, mock_agent_cls = mock_deps
        mock_stt = mock_stt_factory.get_client.return_value

        session = CallSession("sid_123")
        await session.start()

        await session.process_audio("dGVzdA==")
        mock_stt.send_audio_base64.assert_awaited_once_with(
            "dGVzdA==", sample_rate=8000
        )

        await session.shutdown()

    async def test_session_shutdown_cleans_up_resources(self, mock_deps):
        mock_stt_factory, mock_agent_cls = mock_deps
        mock_stt = mock_stt_factory.get_client.return_value
        mock_agent = mock_agent_cls.return_value

        session = CallSession("sid_123")
        await session.start()

        await session.shutdown()

        mock_agent.stop.assert_awaited_once_with("sid_123")
        mock_agent.shutdown.assert_awaited_once()
        mock_stt.close.assert_awaited_once()

        assert session.recv_task.done() or session.recv_task.cancelled()

    async def test_session_on_transcript_invokes_agent(self, mock_deps):
        mock_stt_factory, mock_agent_cls = mock_deps
        mock_agent = mock_agent_cls.return_value

        async def mock_agent_iter():
            yield "streamed "
            yield "response"

        mock_agent.process_transcript = AsyncMock(return_value=mock_agent_iter())

        session = CallSession("sid_123")
        await session.start()

        # Simulate STT sending a committed chunk back up
        await session._on_transcript("committed", "hello")

        mock_agent.process_transcript.assert_awaited_once_with(
            call_sid="sid_123",
            kind="committed",
            text="hello",
        )

    @patch("app.services.orchestrator.session.logger")
    async def test_session_shutdown_hung_recv_task(self, mock_logger, mock_deps):
        mock_stt_factory, mock_agent_cls = mock_deps
        session = CallSession("sid_123")
        await session.start()

        # Simulate waiting for the task timing out
        with patch(
            "app.services.orchestrator.session.asyncio.wait_for",
            side_effect=asyncio.TimeoutError,
        ):
            await session.shutdown()

        mock_logger.warning.assert_called_with(
            "[%s] STT recv_task did not finish in time, cancelling", "sid_123"
        )
