import base64
import struct

import audioop


class TestAudioConversion:
    """Test that the audio conversion logic from the simulation script
    (16-bit PCM → µ-law → base64) produces valid, round-trippable output."""

    def _make_pcm_samples(self, num_frames: int = 160) -> bytes:
        """Create synthetic 16-bit PCM audio (sine-ish pattern)."""
        samples = []
        for i in range(num_frames):
            # Simple sawtooth wave within 16-bit range
            val = (i * 100) % 32767
            samples.append(struct.pack("<h", val))
        return b"".join(samples)

    def test_pcm_to_ulaw_produces_correct_length(self):
        """µ-law is 1 byte per sample vs 2 bytes for 16-bit PCM."""
        pcm_data = self._make_pcm_samples(160)
        assert len(pcm_data) == 320  # 160 frames × 2 bytes

        ulaw_data = audioop.lin2ulaw(pcm_data, 2)
        assert len(ulaw_data) == 160  # 1 byte per sample

    def test_ulaw_to_base64_round_trip(self):
        """base64 encode/decode should be lossless."""
        pcm_data = self._make_pcm_samples(160)
        ulaw_data = audioop.lin2ulaw(pcm_data, 2)

        b64 = base64.b64encode(ulaw_data).decode("utf-8")
        decoded = base64.b64decode(b64)

        assert decoded == ulaw_data

    def test_ulaw_to_pcm_round_trip_approximate(self):
        """µ-law is lossy but should be close to original."""
        pcm_data = self._make_pcm_samples(80)
        ulaw_data = audioop.lin2ulaw(pcm_data, 2)
        recovered = audioop.ulaw2lin(ulaw_data, 2)

        # Unpack both as 16-bit signed integers
        original = struct.unpack(f"<{len(pcm_data) // 2}h", pcm_data)
        restored = struct.unpack(f"<{len(recovered) // 2}h", recovered)

        assert len(original) == len(restored)
        # µ-law quantization error should be small for reasonably large signals
        for orig, rest in zip(original, restored):
            if abs(orig) > 100:  # skip near-zero where relative error is large
                assert (
                    abs(orig - rest) < abs(orig) * 0.15
                ), f"Too much error: original={orig}, restored={rest}"

    def test_empty_pcm(self):
        """Edge case: empty audio should produce empty output."""
        ulaw_data = audioop.lin2ulaw(b"", 2)
        assert ulaw_data == b""
        assert base64.b64encode(ulaw_data) == b""
