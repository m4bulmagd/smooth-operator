import asyncio
import base64
import pytest
from unittest.mock import AsyncMock, patch

from mistralai.models import TranscriptionStreamTextDelta
from app.services.stt.elevenlabs import ElevenLabsRealtimeClient
from app.services.stt.mistral import MistralRealtimeClient


@pytest.mark.asyncio
class TestElevenLabsRealtimeClient:
    @patch("app.services.stt.elevenlabs.AsyncElevenLabs")
    async def test_client_stream_lifecycle(self, mock_elevenlabs):
        mock_client = mock_elevenlabs.return_value
        mock_connection = AsyncMock()
        mock_client.speech_to_text.realtime.connect = AsyncMock()
        mock_client.speech_to_text.realtime.connect.return_value = mock_connection

        client = ElevenLabsRealtimeClient()
        cb = AsyncMock()
        client.set_on_transcript(cb)

        task = asyncio.create_task(client.run_receive_loop())
        await asyncio.sleep(0.01)

        # Feed base64 audio directly to trigger the put_nowait success path
        await client.send_audio_base64("dGVzdA==")
        await client.close()

        await task

        mock_client.speech_to_text.realtime.connect.assert_awaited_once()
        mock_connection.send.assert_awaited_with({"audio_base_64": "dGVzdA=="})
        mock_connection.close.assert_awaited_once()


@pytest.mark.asyncio
class TestMistralRealtimeClient:
    @patch("app.services.stt.mistral.Mistral")
    async def test_client_stream_lifecycle_and_heuristic(self, mock_mistral):
        mock_client = mock_mistral.return_value

        async def mock_transcribe(*args, **kwargs):
            yield TranscriptionStreamTextDelta(text="Hello world.")

        mock_client.audio.realtime.transcribe_stream.return_value = mock_transcribe()

        client = MistralRealtimeClient()
        cb = AsyncMock()
        client.set_on_transcript(cb)

        task = asyncio.create_task(client.run_receive_loop())
        await asyncio.sleep(0.01)

        # Mistral decode layer decodes base64 into raw bytes.
        await client.send_audio_base64(base64.b64encode(b"testdata").decode("utf-8"))
        await client.close()

        await task

        # Test the end of sentence heuristic triggers a committed event
        cb.assert_awaited_once_with("committed", "Hello world.")
