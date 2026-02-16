from typing import Literal, Optional
from pydantic import BaseModel


class TwilioStart(BaseModel):
    callSid: Optional[str] = None
    streamSid: Optional[str] = None


class TwilioMedia(BaseModel):
    payload: str  # base64 audio


class TwilioStop(BaseModel):
    callSid: Optional[str] = None
    streamSid: Optional[str] = None


class TwilioWsEvent(BaseModel):
    event: Literal["connected", "start", "media", "stop"]
    start: Optional[TwilioStart] = None
    media: Optional[TwilioMedia] = None
    stop: Optional[TwilioStop] = None
