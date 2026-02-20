from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.twilio import router as twilio_router
from app.api.routes.settings import router as settings_router

app = FastAPI(title="Twilio Realtime Transcription (Pluggable STT)")

app.include_router(health_router)
app.include_router(twilio_router)
app.include_router(settings_router)
