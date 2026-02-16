import asyncio
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.schemas.twilio_stream import TwilioWsEvent
from app.services.stt.elevenlabs import ElevenLabsRealtimeClient


async def handle_twilio_stream(ws: WebSocket) -> None:
    await ws.accept()

    call_sid = "unknown"
    stt = None  # type: Optional[ElevenLabsRealtimeClient]
    recv_task: Optional[asyncio.Task] = None

    async def on_transcript(kind: str, text: str) -> None:
        # "kind" is partial/committed
        if text.strip():
            if kind == "partial":
                print(f"\r[{call_sid}] PARTIAL: {text}", end="", flush=True)
            else:
                print(f"\r[{call_sid}] COMMITTED: {text}")

    try:
        while True:
            msg_text = await ws.receive_text()

            event = TwilioWsEvent.model_validate_json(msg_text)

            if event.event == "connected":
                print(f"[{call_sid}] Twilio stream connected")
                continue

            if event.event == "start":
                call_sid = (event.start.callSid if event.start else None) or "unknown"
                print(f"[{call_sid}] Twilio stream started")

                stt = ElevenLabsRealtimeClient()
                stt.set_on_transcript(on_transcript)
                await stt.connect()

                recv_task = asyncio.create_task(stt.run_receive_loop())

            elif event.event == "media":
                if not stt or not event.media:
                    continue

                await stt.send_audio_base64(event.media.payload, sample_rate=8000)

            elif event.event == "stop":
                print(f"[{call_sid}] Twilio stream stopped")
                break

    except WebSocketDisconnect:
        print(f"[{call_sid}] Twilio WS disconnected")
    except Exception as e:
        print(f"[{call_sid}] ERROR: {e}")
    finally:
        if stt:
            await stt.close()
        if recv_task:
            recv_task.cancel()
