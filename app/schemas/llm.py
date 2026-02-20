from typing import Literal, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LlmResponse(BaseModel):
    content: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
