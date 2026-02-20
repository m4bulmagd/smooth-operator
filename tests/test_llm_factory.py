from unittest.mock import patch

from app.services.llm.base import BaseLlmClient
from app.services.llm.gemini import GeminiLlmClient
from app.services.llm.mistral import MistralLlmClient


class TestLlmFactory:
    """Test LLM factory dispatch in isolation by patching settings."""

    def _get_factory_client(self, provider: str):
        """Patch settings and call the factory."""
        with patch("app.services.llm.factory.settings") as mock_settings:
            mock_settings.llm_provider = provider
            from app.services.llm.factory import LlmFactory

            return LlmFactory.get_client()

    def test_gemini_dispatch(self):
        with patch("app.services.llm.gemini.settings") as mock_g_settings:
            mock_g_settings.gemini_api_key = "gk_test"
            client = self._get_factory_client("gemini")
            assert isinstance(client, GeminiLlmClient)

    def test_mistral_dispatch(self):
        with patch("app.services.llm.mistral.settings") as mock_m_settings:
            mock_m_settings.mistral_api_key = "mk_test"
            client = self._get_factory_client("mistral")
            assert isinstance(client, MistralLlmClient)

    def test_unknown_provider_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown llm_provider"):
            self._get_factory_client("unknown_provider")

    def test_all_providers_return_base_type(self):
        """Every registered provider should return a BaseLlmClient."""
        from app.services.llm.factory import _PROVIDERS

        for name, cls in _PROVIDERS.items():
            assert issubclass(
                cls, BaseLlmClient
            ), f"Provider '{name}' class {cls} is not a BaseLlmClient subclass"
