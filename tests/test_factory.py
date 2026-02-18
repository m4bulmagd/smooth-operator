import os
import pytest
from unittest.mock import patch, MagicMock

from app.services.stt.base import RealtimeSttClient
from app.services.stt.elevenlabs import ElevenLabsRealtimeClient
from app.services.stt.mistral import MistralRealtimeClient


class TestSttFactory:
    """Test factory dispatch in isolation by patching settings."""

    def _get_factory_client(self, provider: str):
        """Patch settings and call the factory."""
        with patch("app.services.stt.factory.settings") as mock_settings:
            mock_settings.stt_provider = provider
            from app.services.stt.factory import SttFactory

            return SttFactory.get_client()

    def test_elevenlabs_dispatch(self):
        with patch("app.services.stt.elevenlabs.settings") as mock_el_settings:
            mock_el_settings.elevenlabs_api_key = "sk_test"
            client = self._get_factory_client("elevenlabs")
            assert isinstance(client, ElevenLabsRealtimeClient)

    def test_mistral_dispatch(self):
        with patch("app.services.stt.mistral.settings") as mock_m_settings:
            mock_m_settings.mistral_api_key = "mk_test"
            client = self._get_factory_client("mistral")
            assert isinstance(client, MistralRealtimeClient)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown stt_provider"):
            self._get_factory_client("unknown_provider")

    def test_all_providers_return_base_type(self):
        """Every registered provider should return a RealtimeSttClient."""
        from app.services.stt.factory import _PROVIDERS

        for name, cls in _PROVIDERS.items():
            assert issubclass(
                cls, RealtimeSttClient
            ), f"Provider '{name}' class {cls} is not a RealtimeSttClient subclass"
