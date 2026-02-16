import asyncio
import base64
from typing import Optional

from mistralai import Mistral
from mistralai.models import AudioFormat
from mistralai.models import TranscriptionStreamTextDelta

from app.core.config import settings
from app.services.stt.base import RealtimeSttClient


class MistralRealtimeClient(RealtimeSttClient):
    def __init__(self) -> None:
        super().__init__()
        self.client = Mistral(api_key=settings.mistral_api_key)
        self._audio_queue: Optional[asyncio.Queue] = None
        self._current_sentence = ""

    async def send_audio_base64(self, audio_b64: str, sample_rate: int = 8000) -> None:
        if self._audio_queue:
            # Twilio sends base64 encoded audio
            await self._audio_queue.put(base64.b64decode(audio_b64))

    async def run_receive_loop(self) -> None:
        self._audio_queue = asyncio.Queue()

        async def audio_generator():
            while True:
                chunk = await self._audio_queue.get()
                if chunk is None:
                    break
                yield chunk

        # Twilio sends 8000Hz mulaw
        audio_format = AudioFormat(encoding="pcm_mulaw", sample_rate=8000)
        print("Mistral: Starting stream...")
        try:
            async for event in self.client.audio.realtime.transcribe_stream(
                audio_stream=audio_generator(),
                model=settings.mistral_model,
                audio_format=audio_format,
            ):
                if isinstance(event, TranscriptionStreamTextDelta):
                    delta = event.text
                    self._current_sentence += delta

                    # Heuristic for commit: end of sentence punctuation
                    if any(
                        self._current_sentence.strip().endswith(p)
                        for p in [".", "?", "!"]
                    ):
                        if self._on_transcript:
                            await self._on_transcript(
                                "committed", self._current_sentence
                            )
                        self._current_sentence = ""
                    else:
                        if self._on_transcript:
                            await self._on_transcript("partial", self._current_sentence)

        except Exception as e:
            print(f"Mistral error: {e}")
        finally:
            pass

    async def close(self) -> None:
        if hasattr(self, "_audio_queue") and self._audio_queue:
            await self._audio_queue.put(None)
