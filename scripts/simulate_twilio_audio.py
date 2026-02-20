import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from app.core.config import settings
from app.services.stt.factory import SttFactory
from scripts._audio_utils import stream_wav_to_stt


async def simulate_audio_stream(wav_file_path: str):
    """
    Reads a WAV file (must be 8000Hz, 16-bit mono PCM),
    converts it to mu-law (emulating Twilio),
    and sends it to the configured STT provider.
    """

    print(f"Using STT Provider: {settings.stt_provider}")

    stt = SttFactory.get_client()

    async def on_transcript(kind: str, text: str):
        if kind == "partial":
            print(f"\rPartial: {text}", end="", flush=True)
        else:
            print(f"\rCommitted: {text}")

    stt.set_on_transcript(on_transcript)

    # Start the receive loop
    receive_task = asyncio.create_task(stt.run_receive_loop())

    try:
        await stream_wav_to_stt(wav_file_path, stt)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error streaming audio: {e}")
    finally:
        print("\nAudio file finished. Waiting for final processing...")
        await asyncio.sleep(2)
        await stt.close()

        # Cleanup task
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
            "Usage: uv run python -m scripts.simulate_twilio_audio <path_to_8k_16bit_mono.wav>"
        )
        print(
            "Example: uv run python -m scripts.simulate_twilio_audio scripts/sample.wav"
        )
        sys.exit(1)

    wav_path = sys.argv[1]
    asyncio.run(simulate_audio_stream(wav_path))
