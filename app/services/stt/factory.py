from app.core.config import settings
from app.services.stt.base import RealtimeSttClient
from app.services.stt.elevenlabs import ElevenLabsRealtimeClient
from app.services.stt.mistral import MistralRealtimeClient


class SttFactory:
    @staticmethod
    def get_client() -> RealtimeSttClient:
        if settings.stt_provider == "mistral":
            return MistralRealtimeClient()
        return ElevenLabsRealtimeClient()
