import json
from typing import Optional

import websockets

from app.core.config import settings
from app.services.stt.base import RealtimeSttClient


ELEVENLABS_REALTIME_WSS = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"


class ElevenLabsRealtimeClient(RealtimeSttClient):
    def __init__(self) -> None:
        super().__init__()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self) -> None:
        qs = (
            f"?model_id={settings.elevenlabs_model_id}"
            f"&audio_format={settings.elevenlabs_audio_format}"
            f"&commit_strategy={settings.elevenlabs_commit_strategy}"
            f"&include_timestamps={'true' if settings.elevenlabs_include_timestamps else 'false'}"
        )
        headers = {"xi-api-key": settings.elevenlabs_api_key}
        self._ws = await websockets.connect(
            ELEVENLABS_REALTIME_WSS + qs, additional_headers=headers
        )

    async def send_audio_base64(self, audio_b64: str, sample_rate: int = 8000) -> None:
        if not self._ws:
            raise RuntimeError("ElevenLabs client not connected")

        msg = {
            "message_type": "input_audio_chunk",
            "audio_base_64": audio_b64,
            "sample_rate": sample_rate,
            "commit": False,
        }
        await self._ws.send(json.dumps(msg))

    async def run_receive_loop(self) -> None:
        if not self._ws:
            raise RuntimeError("ElevenLabs client not connected")

        async for raw in self._ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mt = data.get("message_type")
            text = data.get("text", "")

            if mt == "partial_transcript" and self._on_transcript:
                await self._on_transcript("partial", text)

            elif mt == "committed_transcript" and self._on_transcript:
                await self._on_transcript("committed", text)

            # optional debugging:
            # else:
            #     print("ELEVEN EVENT:", data)

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None
