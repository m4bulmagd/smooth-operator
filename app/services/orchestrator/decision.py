"""Pure functions for response decision logic and prompt assembly."""

_SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful, friendly AI phone assistant. You are speaking with someone \
on a phone call. Keep your responses concise, natural, and conversational — \
as if you are talking, not writing.

Rules:
- Be brief: 1-3 sentences max unless the caller asks for detail.
- Be warm and professional.
- If you don't know something, say so honestly.
- Never make up facts — only use the knowledge provided below.
- Use natural spoken language (contractions, casual phrasing).

{knowledge_section}\
"""

_KNOWLEDGE_HEADER = """
Here is information you can use to answer questions:

{knowledge_context}

Use this information to answer questions accurately. If the caller asks \
something not covered here, let them know politely.\
"""


def should_respond(kind: str) -> bool:
    """Only respond to committed transcripts (complete thoughts)."""
    return kind == "committed"


def build_system_prompt(knowledge_context: str) -> str:
    """Assemble the full system prompt with optional knowledge injection."""
    if knowledge_context.strip():
        knowledge_section = _KNOWLEDGE_HEADER.format(
            knowledge_context=knowledge_context
        )
    else:
        knowledge_section = ""

    return _SYSTEM_PROMPT_TEMPLATE.format(knowledge_section=knowledge_section)
