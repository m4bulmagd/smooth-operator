import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch

from fastapi import WebSocketDisconnect
from app.services.twilio.stream import handle_twilio_stream


class FakeWebSocket:
    """Minimal mock of FastAPI WebSocket for testing handle_twilio_stream."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)
        self._index = 0
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive_text(self) -> str:
        if self._index >= len(self._messages):
            raise WebSocketDisconnect(code=1000)
        msg = self._messages[self._index]
        self._index += 1
        return msg


def _event(event_type: str, **kwargs) -> str:
    payload: dict = {"event": event_type}
    if event_type == "connected":
        payload["connected"] = kwargs.get("connected", {})
    elif event_type == "start":
        payload["start"] = kwargs.get("start", {"callSid": "CA_TEST"})
    elif event_type == "media":
        payload["media"] = kwargs.get("media", {"payload": "dGVzdA=="})
    elif event_type == "stop":
        payload["stop"] = kwargs.get("stop", {})
    return json.dumps(payload)


@pytest.mark.asyncio
class TestHandleTwilioStream:
    @patch("app.services.twilio.stream.CallSession")
    async def test_normal_flow(self, mock_session_cls):
        """connected → start → media → stop should work without errors."""
        mock_session = mock_session_cls.return_value
        mock_session.start = AsyncMock()
        mock_session.process_audio = AsyncMock()
        mock_session.shutdown = AsyncMock()

        messages = [
            _event("connected"),
            _event("start"),
            _event("media"),
            _event("media"),
            _event("stop"),
        ]
        ws = FakeWebSocket(messages)
        await handle_twilio_stream(ws)

        assert ws.accepted
        mock_session_cls.assert_called_once_with("CA_TEST")
        mock_session.start.assert_awaited_once()
        assert mock_session.process_audio.call_count == 2
        mock_session.shutdown.assert_awaited_once()

    @patch("app.services.twilio.stream.CallSession")
    async def test_media_before_start_is_ignored(self, mock_session_cls):
        """Media events before start should be silently ignored."""
        mock_session = mock_session_cls.return_value
        mock_session.start = AsyncMock()
        mock_session.process_audio = AsyncMock()
        mock_session.shutdown = AsyncMock()

        messages = [
            _event("connected"),
            _event("media"),  # before start — should be ignored
            _event("start"),
            _event("stop"),
        ]
        ws = FakeWebSocket(messages)
        await handle_twilio_stream(ws)

        mock_session.process_audio.assert_not_called()
        mock_session.shutdown.assert_awaited_once()

    @patch("app.services.twilio.stream.CallSession")
    async def test_disconnect_triggers_cleanup(self, mock_session_cls):
        """WebSocketDisconnect should trigger proper cleanup."""
        mock_session = mock_session_cls.return_value
        mock_session.start = AsyncMock()
        mock_session.shutdown = AsyncMock()

        messages = [
            _event("connected"),
            _event("start"),
            # no stop — FakeWebSocket raises WebSocketDisconnect
        ]
        ws = FakeWebSocket(messages)
        await handle_twilio_stream(ws)

        mock_session.shutdown.assert_awaited_once()

    @patch("app.services.twilio.stream.logger")
    async def test_malformed_json_does_not_crash_silently(self, mock_logger):
        """Bad JSON should be caught by the exception handler."""
        messages = [
            _event("connected"),
            "not valid json at all",
        ]
        ws = FakeWebSocket(messages)

        # Should not raise — the generic Exception handler catches it
        await handle_twilio_stream(ws)
        assert ws.accepted

        mock_logger.exception.assert_called_once_with(
            "[%s] ERROR in Twilio stream", "unknown"
        )
