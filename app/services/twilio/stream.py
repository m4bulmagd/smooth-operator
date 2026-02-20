import asyncio
import json
import logging
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.schemas.twilio_stream import TwilioWsEvent
from app.services.orchestrator.session import CallSession

logger = logging.getLogger(__name__)


async def handle_twilio_stream(ws: WebSocket) -> None:
    await ws.accept()

    call_sid = "unknown"
    session: Optional[CallSession] = None

    try:
        while True:
            msg_text = await ws.receive_text()
            raw = json.loads(msg_text)
            event_type = raw.get("event")

            # PERFORMANCE FAST PATH: Bypass Pydantic for high-frequency media frames
            if event_type == "media":
                if session and "media" in raw and "payload" in raw["media"]:
                    await session.process_audio(raw["media"]["payload"])
                continue

            event = TwilioWsEvent.model_validate_json(msg_text)

            if event.event == "connected":
                logger.info("[%s] Twilio stream connected", call_sid)
                continue

            if event.event == "start":
                call_sid = (event.start.callSid if event.start else None) or "unknown"
                logger.info("[%s] Twilio stream started", call_sid)

                session = CallSession(call_sid)
                await session.start()

            elif event.event == "stop":
                logger.info("[%s] Twilio stream stopped", call_sid)
                break

    except WebSocketDisconnect:
        logger.info("[%s] Twilio WS disconnected", call_sid)
    except Exception:
        logger.exception("[%s] ERROR in Twilio stream", call_sid)
    finally:
        if session:
            await session.shutdown()
