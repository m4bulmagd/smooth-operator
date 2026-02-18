from app.core.config import settings
from app.services.stt.base import RealtimeSttClient
from app.services.stt.elevenlabs import ElevenLabsRealtimeClient
from app.services.stt.mistral import MistralRealtimeClient

_PROVIDERS: dict[str, type[RealtimeSttClient]] = {
    "elevenlabs": ElevenLabsRealtimeClient,
    "mistral": MistralRealtimeClient,
}


class SttFactory:
    @staticmethod
    def get_client() -> RealtimeSttClient:
        provider_cls = _PROVIDERS.get(settings.stt_provider)
        if provider_cls is None:
            raise ValueError(
                f"Unknown stt_provider '{settings.stt_provider}'. "
                f"Supported: {', '.join(_PROVIDERS)}"
            )
        return provider_cls()
