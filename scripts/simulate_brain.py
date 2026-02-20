import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from app.core.config import settings
from app.services.orchestrator.agent import ConversationAgent
from app.services.stt.factory import SttFactory
from scripts._audio_utils import stream_wav_to_stt


async def simulate_brain_stream(wav_file_path: str):
    """
    Full STT → Brain pipeline simulation.
    Reads a WAV file, transcribes via STT, sends committed transcripts
    to the conversation agent, and prints LLM responses.
    """

    print(f"Using STT Provider: {settings.stt_provider}")
    print(f"Using LLM Provider: {settings.llm_provider}")

    stt = SttFactory.get_client()
    agent = ConversationAgent()
    call_sid = "SIMULATE_BRAIN"

    last_partial = ""

    async def on_transcript(kind: str, text: str):
        nonlocal last_partial
        if kind == "partial":
            last_partial = text
            print(f"\rPartial: {text}", end="", flush=True)
        else:
            last_partial = ""
            print(f"\rCommitted: {text}")

            # Send to brain
            if agent:
                response_stream = await agent.process_transcript(
                    call_sid=call_sid,
                    kind=kind,
                    text=text,
                )
                if response_stream:
                    print("Assistant: ", end="", flush=True)
                    async for chunk in response_stream:
                        print(chunk, end="", flush=True)
                    print()  # Add a newline when the stream finishes

    stt.set_on_transcript(on_transcript)

    # Start the receive loop
    receive_task = asyncio.create_task(stt.run_receive_loop())
    await agent.start(call_sid)

    try:
        await stream_wav_to_stt(wav_file_path, stt)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error streaming audio: {e}")
    finally:
        await asyncio.sleep(1)  # allow any final STT events

        if last_partial:
            await on_transcript("committed", last_partial)
            await asyncio.sleep(1)  # extra time for LLM responses

        await stt.close()
        await agent.stop(call_sid)
        await agent.shutdown()

        if not receive_task.done():
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
        print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: uv run python -m scripts.simulate_brain <path_to_8k_16bit_mono.wav>"
        )
        print("Example: uv run python -m scripts.simulate_brain scripts/output.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    asyncio.run(simulate_brain_stream(wav_path))
