from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    elevenlabs_api_key: str
    public_base_url: str  # https://....

    # ElevenLabs realtime options
    elevenlabs_model_id: str = "scribe_v2_realtime"
    elevenlabs_audio_format: str = "ulaw_8000"  # matches Twilio
    elevenlabs_commit_strategy: str = "vad"
    elevenlabs_include_timestamps: bool = False

    # Mistral settings
    mistral_api_key: str = ""
    mistral_model: str = "voxtral-mini-transcribe-realtime-2602"

    # STT Provider
    stt_provider: str = "mistral"  # "elevenlabs" or "mistral"


settings = Settings()
