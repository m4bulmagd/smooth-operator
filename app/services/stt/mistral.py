import logging
from typing import AsyncIterator

from mistralai import Mistral
from mistralai.models import AudioFormat
from mistralai.models import TranscriptionStreamTextDelta

from app.core.config import settings
from app.services.stt.base import RealtimeSttClient

logger = logging.getLogger(__name__)


class MistralRealtimeClient(RealtimeSttClient):
    def __init__(self) -> None:
        super().__init__()
        self.client = Mistral(api_key=settings.mistral_api_key)
        self._current_sentence = ""

    async def _run_stt_stream(self, audio_stream: AsyncIterator[bytes]) -> None:
        # Twilio sends 8000Hz mulaw
        audio_format = AudioFormat(encoding="pcm_mulaw", sample_rate=8000)
        logger.info("Mistral: Starting stream...")

        async for event in self.client.audio.realtime.transcribe_stream(
            audio_stream=audio_stream,
            model=settings.mistral_model,
            audio_format=audio_format,
        ):
            if isinstance(event, TranscriptionStreamTextDelta):
                delta = event.text
                self._current_sentence += delta

                # Heuristic for commit: end of sentence punctuation
                if any(
                    self._current_sentence.strip().endswith(p) for p in [".", "?", "!"]
                ):
                    if self._on_transcript:
                        await self._on_transcript("committed", self._current_sentence)
                    self._current_sentence = ""
                else:
                    if self._on_transcript:
                        await self._on_transcript("partial", self._current_sentence)
