import asyncio
import logging
from typing import Optional

from app.services.orchestrator.agent import ConversationAgent
from app.services.stt.base import RealtimeSttClient
from app.services.stt.factory import SttFactory

logger = logging.getLogger(__name__)


class CallSession:
    """Encapsulates the lifecycle of a single call's audio processing and AI orchestration.
    Decouples Twilio/WebSocket transport from the core AI reasoning logic.
    """

    def __init__(self, call_sid: str) -> None:
        self.call_sid = call_sid
        self.agent: Optional[ConversationAgent] = None
        self.stt: Optional[RealtimeSttClient] = None
        self.recv_task: Optional[asyncio.Task] = None

    def _on_recv_task_done(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                "[%s] STT receive loop crashed unexpectedly: %s",
                self.call_sid,
                e,
                exc_info=True,
            )

    async def start(self) -> None:
        """Start the session: initialize agent and STT."""
        self.stt = SttFactory.get_client()
        self.stt.set_on_transcript(self._on_transcript)

        # Start the receive background loop
        self.recv_task = asyncio.create_task(self.stt.run_receive_loop())
        self.recv_task.add_done_callback(self._on_recv_task_done)

        self.agent = ConversationAgent()
        await self.agent.start(self.call_sid)

    async def process_audio(self, payload: str, sample_rate: int = 8000) -> None:
        """Process incoming audio chunk (base64)."""
        if not self.stt:
            return
        await self.stt.send_audio_base64(payload, sample_rate=sample_rate)

    async def _on_transcript(self, kind: str, text: str) -> None:
        if not text.strip():
            return

        if kind == "partial":
            print(f"\r[{self.call_sid}] PARTIAL: {text}", end="", flush=True)
        else:
            print(f"\r[{self.call_sid}] COMMITTED: {text}")

        if self.agent:
            response_stream = await self.agent.process_transcript(
                call_sid=self.call_sid,
                kind=kind,
                text=text,
            )
            if response_stream:
                print(f"[{self.call_sid}] ASSISTANT: ", end="", flush=True)
                async for chunk in response_stream:
                    print(chunk, end="", flush=True)
                print()  # newline after completing response

    async def shutdown(self) -> None:
        """Cleanly shutdown resources."""
        if self.agent:
            await self.agent.stop(self.call_sid)
            await self.agent.shutdown()
        if self.stt:
            await self.stt.close()

        if self.recv_task:
            try:
                await asyncio.wait_for(self.recv_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] STT recv_task did not finish in time, cancelling",
                    self.call_sid,
                )
                self.recv_task.cancel()
            except Exception:
                logger.exception(
                    "[%s] STT recv_task raised during shutdown", self.call_sid
                )
