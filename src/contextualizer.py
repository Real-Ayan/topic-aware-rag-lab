"""LLM context-prefix generation per chunk."""

from __future__ import annotations

from rich.progress import track

from src.config import settings
from src.llm import chat

SYSTEM_PROMPT = (
    "You are a document analyst. Given the full chunk text and its source filename, "
    "write a single sentence (max 30 words) that situates this chunk within the document. "
    "State the document name, topic covered, and any named entity if present."
)


class Contextualizer:
    def contextualize(self, chunk: dict) -> str:
        user = (
            f"Source: {chunk['source']}, page {chunk['page']}\n"
            f"Chunk text: {chunk['content']}\n\n"
            "Response (one sentence only):"
        )
        return chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            model=settings.context_model,
            max_completion_tokens=120,
            temperature=0.2,
            reasoning_effort="none",
        )

    def contextualize_many(self, chunks: list[dict], *, desc: str = "Contextualizing") -> list[str]:
        return [self.contextualize(chunk) for chunk in track(chunks, description=desc)]
