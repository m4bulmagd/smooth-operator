import asyncio
import base64
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

TranscriptCallback = Callable[[str, str], Awaitable[None]]
# callback(kind, text) where kind = "partial" | "committed"

# ~80 seconds of audio at 160ms chunks — bounds memory under backpressure
_DEFAULT_QUEUE_MAXSIZE = 500


class RealtimeSttClient(ABC):
    def __init__(self) -> None:
        self._on_transcript: Optional[TranscriptCallback] = None
        self._audio_queue: Optional[asyncio.Queue] = None

    def set_on_transcript(self, cb: TranscriptCallback) -> None:
        self._on_transcript = cb

    async def send_audio_base64(self, audio_b64: str, sample_rate: int = 8000) -> None:
        """Decode base64 audio and enqueue raw bytes for the STT stream."""
        if self._audio_queue is not None:
            raw = base64.b64decode(audio_b64)
            try:
                self._audio_queue.put_nowait(raw)
            except asyncio.QueueFull:
                logger.warning(
                    "Audio queue full! Dropping oldest frame to prevent latency build-up."
                )
                try:
                    self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._audio_queue.put_nowait(raw)
                except asyncio.QueueFull:
                    pass

    async def run_receive_loop(self) -> None:
        """Create the audio queue, build an async iterator, and delegate to subclass."""
        self._audio_queue = asyncio.Queue(maxsize=_DEFAULT_QUEUE_MAXSIZE)

        async def _audio_iter() -> AsyncIterator[bytes]:
            while True:
                chunk = await self._audio_queue.get()
                if chunk is None:  # sentinel → stop
                    break
                yield chunk

        try:
            await self._run_stt_stream(_audio_iter())
        except Exception:
            logger.exception("STT stream failed")
            raise

    @abstractmethod
    async def _run_stt_stream(self, audio_stream: AsyncIterator[bytes]) -> None:
        """Subclasses implement their provider-specific streaming logic here."""
        ...

    async def close(self) -> None:
        """Send the stop sentinel so the receive loop exits cleanly."""
        if self._audio_queue is not None:
            await self._audio_queue.put(None)
