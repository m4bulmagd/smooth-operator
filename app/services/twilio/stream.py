import asyncio
import logging
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.schemas.twilio_stream import TwilioWsEvent
from app.services.orchestrator.agent import ConversationAgent
from app.services.stt.base import RealtimeSttClient
from app.services.stt.factory import SttFactory

logger = logging.getLogger(__name__)


async def handle_twilio_stream(ws: WebSocket) -> None:
    await ws.accept()

    call_sid = "unknown"
    stt: Optional[RealtimeSttClient] = None
    recv_task: Optional[asyncio.Task] = None
    agent: Optional[ConversationAgent] = None

    async def on_transcript(kind: str, text: str) -> None:
        # "kind" is partial/committed
        if text.strip():
            if kind == "partial":
                print(f"\r[{call_sid}] PARTIAL: {text}", end="", flush=True)
            else:
                print(f"\r[{call_sid}] COMMITTED: {text}")

            # Send to conversation agent for LLM processing
            if agent:
                response = await agent.process_transcript(
                    call_sid=call_sid,
                    kind=kind,
                    text=text,
                )
                if response:
                    print(f"[{call_sid}] ASSISTANT: {response}")

    try:
        while True:
            msg_text = await ws.receive_text()
            event = TwilioWsEvent.model_validate_json(msg_text)

            if event.event == "connected":
                logger.info("[%s] Twilio stream connected", call_sid)
                continue

            if event.event == "start":
                call_sid = (event.start.callSid if event.start else None) or "unknown"
                logger.info("[%s] Twilio stream started", call_sid)

                stt = SttFactory.get_client()
                stt.set_on_transcript(on_transcript)

                recv_task = asyncio.create_task(stt.run_receive_loop())

                # Start the brain
                agent = ConversationAgent()
                await agent.start(call_sid)

            elif event.event == "media":
                if not stt or not event.media:
                    continue

                await stt.send_audio_base64(event.media.payload, sample_rate=8000)

            elif event.event == "stop":
                logger.info("[%s] Twilio stream stopped", call_sid)
                break

    except WebSocketDisconnect:
        logger.info("[%s] Twilio WS disconnected", call_sid)
    except Exception:
        logger.exception("[%s] ERROR in Twilio stream", call_sid)
    finally:
        if agent:
            await agent.stop(call_sid)
            await agent.shutdown()
        if stt:
            await stt.close()
        if recv_task:
            # Wait for the receive loop to process the sentinel and close
            # the upstream STT connection cleanly before cancelling.
            try:
                await asyncio.wait_for(recv_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] STT recv_task did not finish in time, cancelling", call_sid
                )
                recv_task.cancel()
            except Exception:
                logger.exception("[%s] STT recv_task raised during shutdown", call_sid)
