from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ===== REQUIRED FOR ALL DEPLOYMENTS =====
    public_base_url: str

    # ===== PROVIDER SELECTION =====
    stt_provider: str = "elevenlabs"

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


settings = Settings()
