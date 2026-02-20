import os
import pytest
from unittest.mock import patch

from app.schemas.knowledge import KnowledgeData
from app.services.knowledge.simple import SimpleKnowledgeProvider


class TestKnowledgeData:
    def test_format_for_prompt_empty(self):
        data = KnowledgeData()
        assert data.format_for_prompt() == ""

    def test_format_for_prompt_with_entries(self):
        data = KnowledgeData(
            entries={
                "about_me": {"name": "Jane", "role": "Engineer"},
                "faq": {"pricing": "$100/hr"},
            }
        )
        result = data.format_for_prompt()
        assert "## About Me" in result
        assert "- Name: Jane" in result
        assert "- Role: Engineer" in result
        assert "## Faq" in result
        assert "- Pricing: $100/hr" in result


@pytest.mark.asyncio
class TestSimpleKnowledgeProvider:
    async def test_load_valid_yaml(self, tmp_path):
        yaml_file = tmp_path / "knowledge.yaml"
        yaml_file.write_text("about_me:\n  name: Test User\n  role: Tester\n")

        with patch("app.services.knowledge.simple.settings") as mock_settings:
            mock_settings.knowledge_file_path = str(yaml_file)
            provider = SimpleKnowledgeProvider()
            await provider.load()
            context = await provider.get_context()

        assert "Test User" in context
        assert "Tester" in context

    async def test_load_missing_file(self, tmp_path):
        with patch("app.services.knowledge.simple.settings") as mock_settings:
            mock_settings.knowledge_file_path = str(tmp_path / "missing.yaml")
            provider = SimpleKnowledgeProvider()
            await provider.load()
            context = await provider.get_context()

        assert context == ""

    async def test_load_invalid_yaml(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("just a plain string")

        with patch("app.services.knowledge.simple.settings") as mock_settings:
            mock_settings.knowledge_file_path = str(yaml_file)
            provider = SimpleKnowledgeProvider()
            await provider.load()
            context = await provider.get_context()

        assert context == ""
