from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import Response

from app.core.config import settings
from app.utils.urls import https_to_wss
from app.services.twilio.twiml import connect_stream_twiml
from app.services.twilio.stream import handle_twilio_stream

router = APIRouter(prefix="/twilio", tags=["twilio"])


@router.post("/inbound")
async def twilio_inbound_webhook(request: Request):
    """
    Twilio inbound webhook returns TwiML that instructs Twilio
    to start a Media Stream to our WebSocket endpoint.
    """
    base = settings.public_base_url.rstrip("/")
    stream_url = f"{https_to_wss(base)}/twilio/stream"

    xml = connect_stream_twiml(stream_url)
    return Response(content=xml, media_type="application/xml")


@router.websocket("/stream")
async def twilio_stream(ws: WebSocket):
    """WebSocket endpoint that receives Twilio's audio stream"""
    await handle_twilio_stream(ws)
