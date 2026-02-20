import os
import pytest
from unittest.mock import patch
from pydantic import ValidationError


class TestSettingsValidation:
    """Test config validation in isolation — we construct Settings directly
    instead of importing the module-level `settings` singleton, so we can
    control env vars precisely."""

    def _make_settings(self, **overrides):
        """Create a Settings instance with controlled env vars."""
        env = {
            "PUBLIC_BASE_URL": "https://example.com",
            "ELEVENLABS_API_KEY": "sk_test_123",
            "GEMINI_API_KEY": "gk_test_123",
            **{k.upper(): v for k, v in overrides.items()},
        }
        with patch.dict(os.environ, env, clear=False):
            # Import inside so we get a fresh class each time
            from app.core.config import Settings

            return Settings(_env_file=None)  # Skip .env file

    def test_valid_elevenlabs_config(self):
        s = self._make_settings(stt_provider="elevenlabs", elevenlabs_api_key="sk_test")
        assert s.stt_provider == "elevenlabs"
        assert s.elevenlabs_api_key == "sk_test"

    def test_valid_mistral_config(self):
        s = self._make_settings(stt_provider="mistral", mistral_api_key="mk_test")
        assert s.stt_provider == "mistral"

    def test_missing_elevenlabs_key_raises(self):
        with pytest.raises(ValidationError, match="ELEVENLABS_API_KEY"):
            env = {
                "PUBLIC_BASE_URL": "https://example.com",
                "STT_PROVIDER": "elevenlabs",
                # Deliberately no ELEVENLABS_API_KEY
            }
            with patch.dict(os.environ, env, clear=False):
                # Remove ELEVENLABS_API_KEY if it exists in real env
                real_key = os.environ.pop("ELEVENLABS_API_KEY", None)
                try:
                    from app.core.config import Settings

                    Settings(_env_file=None)
                finally:
                    if real_key is not None:
                        os.environ["ELEVENLABS_API_KEY"] = real_key

    def test_missing_mistral_key_raises(self):
        with pytest.raises(ValidationError, match="MISTRAL_API_KEY"):
            # Don't set mistral_api_key at all
            env = {
                "PUBLIC_BASE_URL": "https://example.com",
                "STT_PROVIDER": "mistral",
            }
            with patch.dict(os.environ, env, clear=False):
                from app.core.config import Settings

                Settings(_env_file=None)

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValidationError):
            self._make_settings(stt_provider="invalid_provider")

    # ===== LLM Provider Validation =====

    def test_valid_gemini_config(self):
        s = self._make_settings(llm_provider="gemini", gemini_api_key="gk_test")
        assert s.llm_provider == "gemini"
        assert s.gemini_api_key == "gk_test"

    def test_valid_mistral_llm_config(self):
        s = self._make_settings(llm_provider="mistral", mistral_api_key="mk_test")
        assert s.llm_provider == "mistral"

    def test_missing_gemini_key_raises(self):
        with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
            env = {
                "PUBLIC_BASE_URL": "https://example.com",
                "ELEVENLABS_API_KEY": "sk_test",
                "LLM_PROVIDER": "gemini",
            }
            with patch.dict(os.environ, env, clear=False):
                real_key = os.environ.pop("GEMINI_API_KEY", None)
                try:
                    from app.core.config import Settings

                    Settings(_env_file=None)
                finally:
                    if real_key is not None:
                        os.environ["GEMINI_API_KEY"] = real_key

    def test_invalid_llm_provider_rejected(self):
        with pytest.raises(ValidationError):
            self._make_settings(llm_provider="invalid_llm")
