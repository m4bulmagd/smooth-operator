from app.core.config import settings
from app.services.llm.base import BaseLlmClient
from app.services.llm.gemini import GeminiLlmClient
from app.services.llm.mistral import MistralLlmClient

_PROVIDERS: dict[str, type[BaseLlmClient]] = {
    "gemini": GeminiLlmClient,
    "mistral": MistralLlmClient,
}


class LlmFactory:
    @staticmethod
    def get_client() -> BaseLlmClient:
        provider_cls = _PROVIDERS.get(settings.llm_provider)
        if provider_cls is None:
            raise ValueError(
                f"Unknown llm_provider '{settings.llm_provider}'. "
                f"Supported: {', '.join(_PROVIDERS)}"
            )
        return provider_cls()
