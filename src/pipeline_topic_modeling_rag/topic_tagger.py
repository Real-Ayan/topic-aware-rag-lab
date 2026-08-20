"""Assign topic_slug to each source chunk."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.io_utils import write_jsonl


def tag_chunks(
    chunks: list[dict],
    topics: list[int],
    probs,
    catalog: list[dict],
    out_path: Path,
) -> list[dict]:
    id_to_slug = {t["topic_id"]: t["slug"] for t in catalog}
    tags: list[dict] = []

    for i, chunk in enumerate(chunks):
        topic_id = int(topics[i])
        slug = id_to_slug.get(topic_id, "uncategorized")
        probability = 0.0
        if probs is not None and topic_id != -1:
            try:
                row = probs[i]
                if isinstance(row, (list, np.ndarray)):
                    # BERTopic probs align to non-outlier topics; fall back safely
                    probability = float(np.max(row)) if len(row) else 0.0
                else:
                    probability = float(row)
            except Exception:
                probability = 0.0

        tags.append(
            {
                "chunk_id": chunk["chunk_id"],
                "topic_slug": slug,
                "topic_id": topic_id,
                "probability": round(probability, 4),
            }
        )

    write_jsonl(out_path, tags)
    return tags
