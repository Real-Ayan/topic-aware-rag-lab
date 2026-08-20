"""Chroma persistent store helpers."""

from __future__ import annotations

import chromadb
from chromadb.api.models.Collection import Collection

from src.config import settings


class ChromaStore:
    def __init__(self, path: str | None = None) -> None:
        self.client = chromadb.PersistentClient(path=path or settings.CHROMA_PATH)

    def get_or_create(self, name: str) -> Collection:
        return self.client.get_or_create_collection(name=name)

    def reset_collection(self, name: str) -> Collection:
        """Drop and recreate so re-runs do not accumulate duplicates."""
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        return self.client.create_collection(name=name)

    def add(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        *,
        reset: bool = True,
    ) -> None:
        collection = (
            self.reset_collection(collection_name)
            if reset
            else self.get_or_create(collection_name)
        )
        # Chroma batches internally; keep chunks modest for large corpora
        batch = 200
        for i in range(0, len(ids), batch):
            collection.add(
                ids=ids[i : i + batch],
                documents=documents[i : i + batch],
                embeddings=embeddings[i : i + batch],
                metadatas=metadatas[i : i + batch],
            )

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> list[dict]:
        collection = self.client.get_collection(collection_name)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k or settings.TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        docs = result.get("documents", [[]])[0] or []
        metas = result.get("metadatas", [[]])[0] or []
        dists = result.get("distances", [[]])[0] or []
        out: list[dict] = []
        for content, metadata, distance in zip(docs, metas, dists):
            out.append(
                {
                    "content": content,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )
        return out
