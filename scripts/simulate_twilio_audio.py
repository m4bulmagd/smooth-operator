import asyncio
import base64
import sys
import wave

import audioop
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from app.core.config import settings
from app.services.stt.factory import SttFactory


async def simulate_audio_stream(wav_file_path: str):
    """
    Reads a WAV file (must be 8000Hz, 16-bit mono PCM),
    converts it to mu-law (emulating Twilio),
    and sends it to the configured STT provider.
    """

    if not os.path.exists(wav_file_path):
        print(f"Error: File not found: {wav_file_path}")
        return

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
        with wave.open(wav_file_path, "rb") as wf:
            # We enforce 8000Hz 16-bit mono because our simplified script
            # uses audioop.lin2ulaw which expects 16-bit linear PCM
            if (
                wf.getnchannels() != 1
                or wf.getsampwidth() != 2
                or wf.getframerate() != 8000
            ):
                print(
                    "Warning: WAV file should be 8000Hz, 16-bit, mono for correct mu-law conversion."
                )

            # Read in chunks of 160ms (1280 frames = 2560 bytes)
            chunk_duration = 0.16  # seconds
            chunk_frames = int(8000 * chunk_duration)

            print("Starting stream...")
            while True:
                data = wf.readframes(chunk_frames)
                if not data:
                    break

                # Convert 16-bit PCM to mu-law (1 byte per sample)
                # Twilio sends mu-law encoded audio
                ulaw_data = audioop.lin2ulaw(data, 2)

                # Encode to base64 as expected by our service
                b64_data = base64.b64encode(ulaw_data).decode("utf-8")

                await stt.send_audio_base64(b64_data)

                await asyncio.sleep(chunk_duration)

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
            "Usage: python -m scripts.simulate_twilio_audio <path_to_8k_16bit_mono.wav>"
        )
        print("Example: python -m scripts.simulate_twilio_audio scripts/sample.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    asyncio.run(simulate_audio_stream(wav_path))
