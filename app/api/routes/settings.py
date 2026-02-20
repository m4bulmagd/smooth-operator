from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/settings")
def get_settings():
    return {
        "stt_provider": settings.stt_provider,
        "llm_provider": settings.llm_provider,
        "llm_temperature": settings.llm_temperature,
        "llm_max_output_tokens": settings.llm_max_output_tokens,
        "max_conversation_history": settings.max_conversation_history,
        "knowledge_file_path": settings.knowledge_file_path,
    }
