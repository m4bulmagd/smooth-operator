import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

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
    @patch("app.services.twilio.stream.SttFactory")
    async def test_normal_flow(self, mock_factory):
        """connected → start → media → stop should work without errors."""
        mock_stt = AsyncMock()
        mock_stt.run_receive_loop = AsyncMock(return_value=None)
        mock_factory.get_client.return_value = mock_stt

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
        mock_factory.get_client.assert_called_once()
        mock_stt.set_on_transcript.assert_called_once()
        assert mock_stt.send_audio_base64.call_count == 2
        mock_stt.close.assert_awaited_once()

    @patch("app.services.twilio.stream.SttFactory")
    async def test_media_before_start_is_ignored(self, mock_factory):
        """Media events before start should be silently ignored (stt is None)."""
        mock_stt = AsyncMock()
        mock_stt.run_receive_loop = AsyncMock(return_value=None)
        mock_factory.get_client.return_value = mock_stt

        messages = [
            _event("connected"),
            _event("media"),  # before start — should be ignored
            _event("start"),
            _event("stop"),
        ]
        ws = FakeWebSocket(messages)
        await handle_twilio_stream(ws)

        # send_audio_base64 should NOT have been called for the pre-start media
        mock_stt.send_audio_base64.assert_not_called()

    @patch("app.services.twilio.stream.SttFactory")
    async def test_disconnect_triggers_cleanup(self, mock_factory):
        """WebSocketDisconnect should trigger proper cleanup."""
        mock_stt = AsyncMock()
        mock_stt.run_receive_loop = AsyncMock(return_value=None)
        mock_factory.get_client.return_value = mock_stt

        messages = [
            _event("connected"),
            _event("start"),
            # no stop — FakeWebSocket raises WebSocketDisconnect
        ]
        ws = FakeWebSocket(messages)
        await handle_twilio_stream(ws)

        mock_stt.close.assert_awaited_once()

    async def test_malformed_json_does_not_crash_silently(self):
        """Bad JSON should be caught by the exception handler."""
        messages = [
            _event("connected"),
            "not valid json at all",
        ]
        ws = FakeWebSocket(messages)

        # Should not raise — the generic Exception handler catches it
        await handle_twilio_stream(ws)
        assert ws.accepted
