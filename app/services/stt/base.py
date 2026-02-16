from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional

TranscriptCallback = Callable[[str, str], Awaitable[None]]
# callback(kind, text) where kind = "partial" | "committed"


class RealtimeSttClient(ABC):
    def __init__(self) -> None:
        self._on_transcript: Optional[TranscriptCallback] = None

    def set_on_transcript(self, cb: TranscriptCallback) -> None:
        self._on_transcript = cb

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def send_audio_base64(self, audio_b64: str, sample_rate: int) -> None: ...

    @abstractmethod
    async def run_receive_loop(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...
