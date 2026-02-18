from typing import Literal, Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Maps each provider name to the Settings field that holds its API key.
_PROVIDER_KEY_FIELDS: dict[str, str] = {
    "elevenlabs": "elevenlabs_api_key",
    "mistral": "mistral_api_key",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ===== REQUIRED FOR ALL DEPLOYMENTS =====
    public_base_url: str

    # ===== PROVIDER SELECTION =====
    stt_provider: Literal["elevenlabs", "mistral"] = "elevenlabs"

    # ===== PROVIDER API KEYS (optional - only required if selected) =====
    elevenlabs_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None

    # ===== ELEVENLABS BEHAVIORAL SETTINGS =====
    elevenlabs_model_id: str = "scribe_v2_realtime"
    elevenlabs_audio_format: str = "ulaw_8000"
    elevenlabs_commit_strategy: str = "vad"
    elevenlabs_include_timestamps: bool = False

    # ===== MISTRAL BEHAVIORAL SETTINGS =====
    mistral_model: str = "voxtral-mini-transcribe-realtime-2602"

    @model_validator(mode="after")
    def _validate_provider_api_key(self) -> "Settings":
        key_field = _PROVIDER_KEY_FIELDS.get(self.stt_provider)
        if key_field and getattr(self, key_field) is None:
            raise ValueError(
                f"stt_provider is '{self.stt_provider}' but "
                f"{key_field.upper()} is not set. "
                f"Please set it in your .env file or environment."
            )
        return self


settings = Settings()
