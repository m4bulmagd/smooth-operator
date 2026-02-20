from pydantic import BaseModel, Field


class KnowledgeData(BaseModel):
    """Structured representation of the knowledge base loaded from YAML."""

    entries: dict[str, dict[str, str]] = Field(default_factory=dict)

    def format_for_prompt(self) -> str:
        """Format all knowledge entries into a prompt-friendly string."""
        if not self.entries:
            return ""

        lines: list[str] = []
        for category, items in self.entries.items():
            heading = category.replace("_", " ").title()
            lines.append(f"## {heading}")
            for key, value in items.items():
                label = key.replace("_", " ").title()
                lines.append(f"- {label}: {value}")
            lines.append("")  # blank line between categories

        return "\n".join(lines).strip()
