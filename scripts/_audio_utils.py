"""Shared audio streaming utilities for simulation scripts."""

import asyncio
import base64
import os
import wave

import audioop

from app.services.stt.base import RealtimeSttClient


async def stream_wav_to_stt(
    wav_file_path: str,
    stt: RealtimeSttClient,
    chunk_duration: float = 0.16,
) -> None:
    """Read a WAV file and stream it to an STT client as base64-encoded mu-law.

    The WAV file should be 8000Hz, 16-bit, mono PCM for correct mu-law conversion.

    Args:
        wav_file_path: Path to the WAV file.
        stt: The STT client to send audio to.
        chunk_duration: Duration of each audio chunk in seconds.
    """
    if not os.path.exists(wav_file_path):
        raise FileNotFoundError(f"WAV file not found: {wav_file_path}")

    with wave.open(wav_file_path, "rb") as wf:
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getframerate() != 8000
        ):
            print(
                "Warning: WAV file should be 8000Hz, 16-bit, mono "
                "for correct mu-law conversion."
            )

        chunk_frames = int(8000 * chunk_duration)

        print("Starting stream...")
        while True:
            data = wf.readframes(chunk_frames)
            if not data:
                break

            ulaw_data = audioop.lin2ulaw(data, 2)
            b64_data = base64.b64encode(ulaw_data).decode("utf-8")

            await stt.send_audio_base64(b64_data)
            await asyncio.sleep(chunk_duration)

        # Send 1.5 seconds of silence to ensure STT VAD triggers a commit at the end of the file
        padding_frames = int(8000 * 1.5)
        silence_pcm = b"\x00" * (padding_frames * 2)
        silence_ulaw = audioop.lin2ulaw(silence_pcm, 2)
        silence_b64 = base64.b64encode(silence_ulaw).decode("utf-8")
        await stt.send_audio_base64(silence_b64)
        await asyncio.sleep(1.5)
