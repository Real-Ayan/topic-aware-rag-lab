"""OpenAI embeddings wrapper with batching + usage tracking."""

from __future__ import annotations

from openai import OpenAI

from src.config import settings
from src.run_context import get_run

_BATCH_SIZE = 100


class Embedder:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        run = get_run()
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda d: d.index)
            vectors.extend(item.embedding for item in ordered)
            if run is not None:
                usage = getattr(response, "usage", None)
                tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
                run.usage.add_embedding(tokens=tokens, texts=len(batch))
        if run is not None:
            run.write_metadata(status="running")
        return vectors
