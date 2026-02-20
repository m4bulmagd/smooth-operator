"""Tests for GeminiLlmClient and MistralLlmClient with mocked SDK clients."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.llm import ChatMessage, LlmResponse
from app.services.llm.gemini import GeminiLlmClient, _to_gemini_contents
from app.services.llm.mistral import MistralLlmClient, _to_mistral_messages


# ===== Gemini Conversion Tests =====


class TestToGeminiContents:
    def test_user_message(self):
        msgs = [ChatMessage(role="user", content="Hello")]
        result = _to_gemini_contents(msgs)
        assert len(result) == 1
        assert result[0].role == "user"

    def test_assistant_mapped_to_model(self):
        msgs = [ChatMessage(role="assistant", content="Hi there")]
        result = _to_gemini_contents(msgs)
        assert len(result) == 1
        assert result[0].role == "model"

    def test_system_messages_filtered_out(self):
        msgs = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello"),
        ]
        result = _to_gemini_contents(msgs)
        assert len(result) == 1
        assert result[0].role == "user"

    def test_empty_messages(self):
        result = _to_gemini_contents([])
        assert result == []

    def test_mixed_conversation(self):
        msgs = [
            ChatMessage(role="system", content="system"),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
            ChatMessage(role="user", content="How are you?"),
        ]
        result = _to_gemini_contents(msgs)
        assert len(result) == 3
        assert result[0].role == "user"
        assert result[1].role == "model"
        assert result[2].role == "user"


# ===== Mistral Conversion Tests =====


class TestToMistralMessages:
    def test_basic_messages(self):
        msgs = [ChatMessage(role="user", content="Hello")]
        result = _to_mistral_messages(msgs)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_system_prompt_prepended(self):
        msgs = [ChatMessage(role="user", content="Hello")]
        result = _to_mistral_messages(msgs, system_prompt="Be helpful")
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "Be helpful"}
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_system_messages_in_input_skipped(self):
        msgs = [
            ChatMessage(role="system", content="Old system msg"),
            ChatMessage(role="user", content="Hello"),
        ]
        result = _to_mistral_messages(msgs, system_prompt="New system")
        assert len(result) == 2
        assert result[0]["content"] == "New system"
        assert result[1]["content"] == "Hello"

    def test_no_system_prompt(self):
        msgs = [ChatMessage(role="user", content="Hello")]
        result = _to_mistral_messages(msgs)
        assert len(result) == 1
        # No system message prepended

    def test_empty_messages_with_system_prompt(self):
        result = _to_mistral_messages([], system_prompt="Be helpful")
        assert len(result) == 1
        assert result[0]["role"] == "system"


# ===== Gemini Client Tests =====


@pytest.mark.asyncio
class TestGeminiLlmClient:
    def _make_mock_response(self, text="Hello!", prompt_tokens=10, completion_tokens=5):
        response = MagicMock()
        response.text = text
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = prompt_tokens
        response.usage_metadata.candidates_token_count = completion_tokens
        return response

    async def test_generate_returns_llm_response(self):
        mock_response = self._make_mock_response()

        with patch("app.services.llm.gemini.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.gemini_llm_model = "test-model"
            mock_settings.llm_temperature = 0.7
            mock_settings.llm_max_output_tokens = 256

            client = GeminiLlmClient()
            client._client = MagicMock()
            client._client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            msgs = [ChatMessage(role="user", content="Hello")]
            result = await client.generate(msgs, system_prompt="Be helpful")

            assert isinstance(result, LlmResponse)
            assert result.content == "Hello!"
            assert result.model == "test-model"
            assert result.prompt_tokens == 10
            assert result.completion_tokens == 5

    async def test_generate_handles_none_text(self):
        mock_response = self._make_mock_response(text=None)

        with patch("app.services.llm.gemini.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.gemini_llm_model = "test-model"
            mock_settings.llm_temperature = 0.7
            mock_settings.llm_max_output_tokens = 256

            client = GeminiLlmClient()
            client._client = MagicMock()
            client._client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            msgs = [ChatMessage(role="user", content="Hello")]
            result = await client.generate(msgs)

            assert result.content == ""

    async def test_generate_handles_missing_usage_metadata(self):
        response = MagicMock()
        response.text = "Hi"
        response.usage_metadata = None

        with patch("app.services.llm.gemini.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.gemini_llm_model = "test-model"
            mock_settings.llm_temperature = 0.7
            mock_settings.llm_max_output_tokens = 256

            client = GeminiLlmClient()
            client._client = MagicMock()
            client._client.aio.models.generate_content = AsyncMock(
                return_value=response
            )

            msgs = [ChatMessage(role="user", content="Hello")]
            result = await client.generate(msgs)

            assert result.content == "Hi"
            assert result.prompt_tokens is None
            assert result.completion_tokens is None


# ===== Mistral Client Tests =====


@pytest.mark.asyncio
class TestMistralLlmClient:
    def _make_mock_response(
        self, content="Hello!", prompt_tokens=10, completion_tokens=5
    ):
        choice = MagicMock()
        choice.message.content = content
        response = MagicMock()
        response.choices = [choice]
        response.usage = MagicMock()
        response.usage.prompt_tokens = prompt_tokens
        response.usage.completion_tokens = completion_tokens
        return response

    async def test_generate_returns_llm_response(self):
        mock_response = self._make_mock_response()

        with patch("app.services.llm.mistral.settings") as mock_settings:
            mock_settings.mistral_api_key = "test-key"
            mock_settings.mistral_llm_model = "test-model"

            client = MistralLlmClient()
            client._client = MagicMock()
            client._client.chat.complete_async = AsyncMock(return_value=mock_response)

            msgs = [ChatMessage(role="user", content="Hello")]
            result = await client.generate(msgs, system_prompt="Be helpful")

            assert isinstance(result, LlmResponse)
            assert result.content == "Hello!"
            assert result.model == "test-model"
            assert result.prompt_tokens == 10
            assert result.completion_tokens == 5

    async def test_generate_handles_none_response(self):
        with patch("app.services.llm.mistral.settings") as mock_settings:
            mock_settings.mistral_api_key = "test-key"
            mock_settings.mistral_llm_model = "test-model"

            client = MistralLlmClient()
            client._client = MagicMock()
            client._client.chat.complete_async = AsyncMock(return_value=None)

            msgs = [ChatMessage(role="user", content="Hello")]
            result = await client.generate(msgs)

            assert result.content == ""
            assert result.prompt_tokens is None
            assert result.completion_tokens is None

    async def test_generate_handles_empty_choices(self):
        response = MagicMock()
        response.choices = []

        with patch("app.services.llm.mistral.settings") as mock_settings:
            mock_settings.mistral_api_key = "test-key"
            mock_settings.mistral_llm_model = "test-model"

            client = MistralLlmClient()
            client._client = MagicMock()
            client._client.chat.complete_async = AsyncMock(return_value=response)

            msgs = [ChatMessage(role="user", content="Hello")]
            result = await client.generate(msgs)

            assert result.content == ""

    async def test_generate_handles_none_message_content(self):
        choice = MagicMock()
        choice.message.content = None
        response = MagicMock()
        response.choices = [choice]
        response.usage = MagicMock()
        response.usage.prompt_tokens = 5
        response.usage.completion_tokens = 3

        with patch("app.services.llm.mistral.settings") as mock_settings:
            mock_settings.mistral_api_key = "test-key"
            mock_settings.mistral_llm_model = "test-model"

            client = MistralLlmClient()
            client._client = MagicMock()
            client._client.chat.complete_async = AsyncMock(return_value=response)

            msgs = [ChatMessage(role="user", content="Hello")]
            result = await client.generate(msgs)

            assert result.content == ""
