from typing import Literal, Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Maps each provider name to the Settings field that holds its API key.
_STT_PROVIDER_KEY_FIELDS: dict[str, str] = {
    "elevenlabs": "elevenlabs_api_key",
    "mistral": "mistral_api_key",
}

_LLM_PROVIDER_KEY_FIELDS: dict[str, str] = {
    "gemini": "gemini_api_key",
    "mistral": "mistral_api_key",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ===== REQUIRED FOR ALL DEPLOYMENTS =====
    public_base_url: str

    # ===== STT PROVIDER SELECTION =====
    stt_provider: Literal["elevenlabs", "mistral"] = "elevenlabs"

    # ===== LLM PROVIDER SELECTION =====
    llm_provider: Literal["gemini", "mistral"] = "gemini"

    # ===== PROVIDER API KEYS (optional - only required if selected) =====
    elevenlabs_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # ===== ELEVENLABS BEHAVIORAL SETTINGS =====
    elevenlabs_stt_model: str = "scribe_v2_realtime"
    elevenlabs_audio_format: str = "ulaw_8000"
    elevenlabs_commit_strategy: str = "vad"
    elevenlabs_include_timestamps: bool = False

    # ===== MISTRAL BEHAVIORAL SETTINGS (STT) =====
    mistral_stt_model: str = "voxtral-mini-transcribe-realtime-2602"

    # ===== MISTRAL LLM SETTINGS =====
    mistral_llm_model: str = "mistral-large-latest"

    # ===== GEMINI LLM SETTINGS =====
    gemini_llm_model: str = "gemini-2.5-flash"

    # ===== LLM GENERATION SETTINGS =====
    llm_temperature: float = 0.7
    llm_max_output_tokens: int = 256

    # ===== CONVERSATION SETTINGS =====
    max_conversation_history: int = 20

    # ===== KNOWLEDGE =====
    knowledge_file_path: str = "knowledge.yaml"

    @model_validator(mode="after")
    def _validate_provider_api_keys(self) -> "Settings":
        # Validate STT provider key
        stt_key_field = _STT_PROVIDER_KEY_FIELDS.get(self.stt_provider)
        if stt_key_field and getattr(self, stt_key_field) is None:
            raise ValueError(
                f"stt_provider is '{self.stt_provider}' but "
                f"{stt_key_field.upper()} is not set. "
                f"Please set it in your .env file or environment."
            )

        # Validate LLM provider key
        llm_key_field = _LLM_PROVIDER_KEY_FIELDS.get(self.llm_provider)
        if llm_key_field and getattr(self, llm_key_field) is None:
            raise ValueError(
                f"llm_provider is '{self.llm_provider}' but "
                f"{llm_key_field.upper()} is not set. "
                f"Please set it in your .env file or environment."
            )

        return self


settings = Settings()
