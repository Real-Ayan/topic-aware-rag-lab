"""Topic discovery: UMAP + HDBSCAN/KMeans, LLM labels from random cluster samples."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from rich.console import Console
from rich.progress import track
from sklearn.cluster import KMeans
from umap import UMAP

from src.config import settings
from src.io_utils import ensure_dir, write_json
from src.llm import chat

console = Console()

LABEL_SYSTEM = (
    "You are naming topics for a company knowledge base. "
    "You will see random example chunks from ONE cluster. "
    "Infer the shared theme. Reply with JSON only."
)


def _sample_chunks(chunks: list[dict], idxs: list[int], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    chosen = idxs[:] if len(idxs) <= n else rng.sample(idxs, n)
    samples = []
    for i in chosen:
        c = chunks[i]
        text = c["content"]
        if len(text) > 600:
            text = text[:600] + "…"
        samples.append(
            {
                "chunk_id": c["chunk_id"],
                "source": c["source"],
                "excerpt": text,
            }
        )
    return samples


def _label_topic_from_samples(samples: list[dict], topic_id: int) -> dict:
    blocks = []
    for s in samples:
        blocks.append(f"[{s['chunk_id']}] source={s['source']}\n{s['excerpt']}")
    user = (
        f"Cluster id: {topic_id}\n"
        f"Here are {len(samples)} random example chunks from this cluster:\n\n"
        + "\n\n---\n\n".join(blocks)
        + "\n\n"
        "Name this topic based on the examples (not stopwords).\n"
        'Reply JSON: {"slug": "lowercase_underscore", "label": "3-6 word title", '
        '"summary": "one sentence describing the theme"}'
    )
    raw = chat(
        [
            {"role": "system", "content": LABEL_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=settings.LLM_MODEL,
        max_completion_tokens=200,
        temperature=0.2,
        response_format={"type": "json_object"},
        reasoning_effort="low",
    ) or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    slug = str(data.get("slug", f"topic_{topic_id}")).strip().lower().replace(" ", "_")
    slug = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in slug)
    while "__" in slug:
        slug = slug.replace("__", "_")
    label = str(data.get("label", slug.replace("_", " ").title())).strip()
    summary = str(data.get("summary", "")).strip()
    return {"slug": slug or f"topic_{topic_id}", "label": label, "summary": summary}


def _fit_clusters(texts: list[str], emb: np.ndarray) -> tuple[BERTopic | None, list[int], object]:
    """Fit clustering; returns (optional BERTopic, topic ids, probs)."""
    method = settings.CLUSTER_METHOD.lower().strip()

    umap_model = UMAP(
        n_neighbors=settings.UMAP_N_NEIGHBORS,
        n_components=settings.UMAP_N_COMPONENTS,
        min_dist=settings.UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )

    if method == "kmeans":
        n = max(2, min(settings.N_TOPICS, len(texts) - 1))
        console.print(f"[cyan]Clustering with KMeans k={n} (UMAP → {settings.UMAP_N_COMPONENTS}d)...[/cyan]")
        reduced = umap_model.fit_transform(emb)
        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = km.fit_predict(reduced)
        return None, [int(x) for x in labels], None

    console.print(
        f"[cyan]Clustering with HDBSCAN[/cyan] "
        f"min_cluster_size={settings.HDBSCAN_MIN_CLUSTER_SIZE}, "
        f"min_samples={settings.HDBSCAN_MIN_SAMPLES}, "
        f"selection={settings.HDBSCAN_SELECTION_METHOD}"
    )
    hdbscan_kwargs: dict = {
        "min_cluster_size": settings.HDBSCAN_MIN_CLUSTER_SIZE,
        "metric": "euclidean",
        "cluster_selection_method": settings.HDBSCAN_SELECTION_METHOD,
        "prediction_data": True,
    }
    if settings.HDBSCAN_MIN_SAMPLES is not None:
        hdbscan_kwargs["min_samples"] = settings.HDBSCAN_MIN_SAMPLES
    hdbscan_model = HDBSCAN(**hdbscan_kwargs)
    # Skip heavy c-TF-IDF representation for labeling; still fit BERTopic for consistency
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
        embedding_model=None,
    )
    topics, probs = topic_model.fit_transform(texts, embeddings=emb)
    return topic_model, [int(t) for t in topics], probs


def save_embeddings_cache(
    out_dir: Path,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    ensure_dir(out_dir)
    np.save(out_dir / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    write_json(
        out_dir / "embedding_cache_meta.json",
        {
            "chunk_ids": [c["chunk_id"] for c in chunks],
            "n_chunks": len(chunks),
            "embedding_model": settings.EMBEDDING_MODEL,
            "dim": len(embeddings[0]) if embeddings else 0,
        },
    )


def load_embeddings_cache(out_dir: Path) -> tuple[np.ndarray, list[str]]:
    emb_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "embedding_cache_meta.json"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No embedding cache in {out_dir}. Run embed step first "
            f"(or omit --reuse-embeddings)."
        )
    emb = np.load(emb_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return emb, list(meta["chunk_ids"])


def discover_topics(
    chunks: list[dict],
    embeddings: list[list[float]] | np.ndarray,
    artifacts_dir: Path | None = None,
    *,
    save_bertopic: bool = True,
) -> tuple[object, list[int], object, list[dict]]:
    """
    Cluster chunks and label each topic with an LLM using random samples
    (does NOT rely on c-TF-IDF keyword lists).
    """
    texts = [c["content"] for c in chunks]
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.shape[0] != len(chunks):
        raise ValueError(f"embeddings ({emb.shape[0]}) != chunks ({len(chunks)})")

    topic_model, topics, probs = _fit_clusters(texts, emb)
    unique_topics = sorted({t for t in topics if t != -1})

    catalog: list[dict] = []
    used_slugs: set[str] = set()

    console.print(
        f"[cyan]Labeling {len(unique_topics)} topics from "
        f"{settings.TOPIC_LABEL_SAMPLES} random samples each (LLM)...[/cyan]"
    )

    for topic_id in track(unique_topics, description="LLM topic labels"):
        idxs = [i for i, t in enumerate(topics) if t == topic_id]
        samples = _sample_chunks(
            chunks, idxs, settings.TOPIC_LABEL_SAMPLES, seed=42 + int(topic_id)
        )
        labeled = _label_topic_from_samples(samples, topic_id)
        slug = labeled["slug"]
        if slug in used_slugs or slug == "uncategorized":
            slug = f"{slug}_{topic_id}"
        used_slugs.add(slug)

        catalog.append(
            {
                "topic_id": int(topic_id),
                "slug": slug,
                "label": labeled["label"],
                "summary": labeled.get("summary", ""),
                "sample_chunk_ids": [s["chunk_id"] for s in samples],
                "chunk_count": len(idxs),
                "labeling_method": "llm_random_samples",
            }
        )

    outlier_count = sum(1 for t in topics if t == -1)
    catalog.append(
        {
            "topic_id": -1,
            "slug": "uncategorized",
            "label": "Uncategorized (outliers)",
            "summary": "",
            "sample_chunk_ids": [],
            "chunk_count": outlier_count,
            "labeling_method": "outlier",
        }
    )

    out_root = artifacts_dir or Path("artifacts/topic_modeling")
    ensure_dir(out_root)
    write_json(out_root / "discovered_topics.json", catalog)
    write_json(
        out_root / "cluster_config.json",
        {
            "cluster_method": settings.CLUSTER_METHOD,
            "n_topics": settings.N_TOPICS,
            "umap_n_components": settings.UMAP_N_COMPONENTS,
            "umap_n_neighbors": settings.UMAP_N_NEIGHBORS,
            "umap_min_dist": settings.UMAP_MIN_DIST,
            "hdbscan_min_cluster_size": settings.HDBSCAN_MIN_CLUSTER_SIZE,
            "hdbscan_min_samples": settings.HDBSCAN_MIN_SAMPLES,
            "hdbscan_selection_method": settings.HDBSCAN_SELECTION_METHOD,
            "topic_label_samples": settings.TOPIC_LABEL_SAMPLES,
            "topics_found": len(unique_topics),
            "outliers": outlier_count,
        },
    )

    if save_bertopic and topic_model is not None:
        model_dir = out_root / "bertopic_model"
        ensure_dir(model_dir)
        try:
            topic_model.save(str(model_dir), serialization="safetensors", save_ctfidf=True)
        except Exception as exc:
            console.print(f"[yellow]BERTopic save skipped:[/yellow] {exc}")

    console.print(f"[green]Saved topics:[/green] {out_root / 'discovered_topics.json'}")
    console.print(f"[green]Topics (excl. outliers):[/green] {len(unique_topics)}")
    console.print(f"[yellow]Outliers:[/yellow] {outlier_count}")
    return topic_model, topics, probs, catalog
