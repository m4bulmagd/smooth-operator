import asyncio
import base64
from typing import Optional

from elevenlabs.client import AsyncElevenLabs
from elevenlabs import AudioFormat
from elevenlabs.realtime.scribe import CommitStrategy

from app.core.config import settings
from app.services.stt.base import RealtimeSttClient


class ElevenLabsRealtimeClient(RealtimeSttClient):
    def __init__(self) -> None:
        super().__init__()
        self.client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)
        self._socket = None
        self._receive_task: Optional[asyncio.Task] = None

    async def send_audio_base64(self, audio_b64: str, sample_rate: int = 8000) -> None:
        # We need to send audio chunks to the generator
        # This is tricky because the SDK expects an iterator of chunks
        # We might need to use an asyncio.Queue to bridge the gap
        if hasattr(self, "_audio_queue"):
            await self._audio_queue.put(base64.b64decode(audio_b64))

    async def run_receive_loop(self) -> None:
        self._audio_queue = asyncio.Queue()

        # Define handler for transcripts
        def handle_transcript(data: dict, kind: str):
            # The data payload from ElevenLabs usually contains "text"
            # In some versions it might be "transcript".
            # safely get text
            text = data.get("text") or data.get("transcript") or ""
            if self._on_transcript and text:
                asyncio.create_task(self._on_transcript(kind, text))

        try:
            # Connect
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

            # Register handlers - using string literals matching RealtimeEvents enum
            connection.on(
                "partial_transcript", lambda data: handle_transcript(data, "partial")
            )
            connection.on(
                "committed_transcript",
                lambda data: handle_transcript(data, "committed"),
            )

            # Send audio loop
            while True:
                chunk = await self._audio_queue.get()
                if chunk is None:
                    break

                # Convert bytes to base64
                audio_b64 = base64.b64encode(chunk).decode("utf-8")

                await connection.send({"audio_base_64": audio_b64})

            # Close connection when done
            await connection.close()

        except Exception as e:
            print(f"ElevenLabs error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            pass

    async def close(self) -> None:
        if hasattr(self, "_audio_queue"):
            await self._audio_queue.put(None)
