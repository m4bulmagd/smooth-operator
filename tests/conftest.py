import os

# Set a dummy GEMINI_API_KEY so the module-level Settings() singleton
# doesn't fail during test collection (the default llm_provider is "gemini").
os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")
