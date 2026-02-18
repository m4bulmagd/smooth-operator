import asyncio
import base64
import logging
from typing import AsyncIterator

from elevenlabs.client import AsyncElevenLabs
from elevenlabs import AudioFormat
from elevenlabs.realtime.scribe import CommitStrategy

from app.core.config import settings
from app.services.stt.base import RealtimeSttClient

logger = logging.getLogger(__name__)


class ElevenLabsRealtimeClient(RealtimeSttClient):
    def __init__(self) -> None:
        super().__init__()
        self.client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)

    async def send_audio_base64(self, audio_b64: str, sample_rate: int = 8000) -> None:
        """ElevenLabs SDK expects base64, so skip the decode/re-encode — pass through directly."""
        if self._audio_queue is not None:
            await self._audio_queue.put(audio_b64)

    async def _run_stt_stream(self, audio_stream: AsyncIterator[bytes]) -> None:
        commit_strategy_str = settings.elevenlabs_commit_strategy
        cs = (
            CommitStrategy.VAD
            if commit_strategy_str == "vad"
            else CommitStrategy.MANUAL
        )

        options = {
            "model_id": settings.elevenlabs_model_id,
            "audio_format": AudioFormat.ULAW_8000,
            "sample_rate": 8000,
            "commit_strategy": cs,
        }

        connection = await self.client.speech_to_text.realtime.connect(options)

        def handle_transcript(data: dict, kind: str) -> None:
            text = data.get("text") or data.get("transcript") or ""
            if self._on_transcript and text:
                asyncio.create_task(self._on_transcript(kind, text))

        connection.on(
            "partial_transcript", lambda data: handle_transcript(data, "partial")
        )
        connection.on(
            "committed_transcript",
            lambda data: handle_transcript(data, "committed"),
        )

        try:
            # audio_stream yields raw base64 strings (not bytes) for ElevenLabs
            async for chunk in audio_stream:
                # chunk is already a base64 string since we override send_audio_base64
                audio_b64 = (
                    chunk
                    if isinstance(chunk, str)
                    else base64.b64encode(chunk).decode("utf-8")
                )
                await connection.send({"audio_base_64": audio_b64})
        finally:
            await connection.close()
