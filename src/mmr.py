"""Maximal Marginal Relevance (MMR) for diverse retrieval."""

from __future__ import annotations

import numpy as np


def _as_unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def mmr_select(
    query_embedding: list[float] | np.ndarray,
    candidates: list[dict],
    *,
    top_k: int,
    lambda_mult: float = 0.7,
) -> list[dict]:
    """
    Select top_k candidates balancing relevance to query vs diversity.

    Each candidate should include:
      - content
      - embedding (list[float])  OR we fall back to order-only (no MMR)
    lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity
    """
    if not candidates or top_k <= 0:
        return []
    if len(candidates) <= top_k:
        return candidates

    if any(c.get("embedding") is None for c in candidates):
        # Cannot MMR without vectors — keep original rank order
        return candidates[:top_k]

    q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    q = _as_unit(q)[0]
    emb = _as_unit(np.asarray([c["embedding"] for c in candidates], dtype=np.float32))

    # Relevance = cosine similarity to query
    relevance = emb @ q

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        if not selected:
            pick = int(remaining[int(np.argmax(relevance[remaining]))])
            selected.append(pick)
            remaining.remove(pick)
            continue

        selected_emb = emb[selected]
        # Max similarity to any already-selected doc
        sim_to_selected = emb[remaining] @ selected_emb.T
        max_sim = np.max(sim_to_selected, axis=1)
        mmr_scores = lambda_mult * relevance[remaining] - (1.0 - lambda_mult) * max_sim
        pick_local = int(np.argmax(mmr_scores))
        pick = remaining[pick_local]
        selected.append(pick)
        remaining.remove(pick)

    return [candidates[i] for i in selected]
