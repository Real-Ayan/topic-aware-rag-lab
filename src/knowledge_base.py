"""Named knowledge bases backed by independent Chroma stores (run-scoped)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.chroma_store import ChromaStore
from src.config import settings
from src.embedder import Embedder
from src.io_utils import answer_with_context, ensure_dir, read_json, write_json
from src.mmr import mmr_select
from src.run_context import RUNS_DIR, get_run


def _kb_meta(base_name: str) -> dict:
    pipelines = {
        settings.KB_CONTEXTUAL: {
            "pipeline": "pipeline_contextual_rag",
            "description": "Contextual RAG: source chunks with LLM context prefixes.",
        },
        settings.KB_TOPIC_MODELING: {
            "pipeline": "pipeline_topic_modeling_rag",
            "description": (
                "Topic-tagged original chunks (topics filter/route; no rewrite KB). "
                "Retrieval uses MMR for diversity."
            ),
        },
    }
    info = pipelines.get(
        base_name,
        {"pipeline": "unknown", "description": base_name},
    )
    return {
        "base_name": base_name,
        "pipeline": info["pipeline"],
        "description": info["description"],
    }


def resolve_kb(base_or_full: str) -> tuple[str, Path, Path]:
    """
    Resolve a KB name to (collection_name, chroma_path, root_dir).
    Accepts base names (uses active run) or full `{base}__{run_id}` names.
    """
    run = get_run()
    if "__" in base_or_full:
        name = base_or_full
        matches = list(RUNS_DIR.glob(f"*/knowledge_bases/{name}"))
        if matches:
            root = matches[0]
            return name, root / "chroma", root
        if run is not None:
            root = run.knowledge_bases_dir / name
            return name, root / "chroma", root
        raise KeyError(f"Knowledge base not found: {name}")

    base = base_or_full
    if run is None:
        root = Path(settings.KNOWLEDGE_BASES_DIR) / base
        return base, root / "chroma", root

    name = run.kb_name(base)
    root = run.knowledge_bases_dir / name
    return name, Path(run.kb_chroma_path(base)), root


def list_knowledge_bases() -> list[dict]:
    """Scan all runs for built knowledge bases."""
    out: list[dict] = []
    if RUNS_DIR.exists():
        for kb_root in sorted(RUNS_DIR.glob("*/knowledge_bases/*")):
            if not kb_root.is_dir():
                continue
            manifest_path = kb_root / "manifest.json"
            entry = {
                "name": kb_root.name,
                "path": str(kb_root),
                "ready": manifest_path.exists(),
                "run_id": kb_root.parent.parent.name,
            }
            if manifest_path.exists():
                entry["manifest"] = read_json(manifest_path)
            out.append(entry)
    # Also legacy top-level KBs
    legacy = Path(settings.KNOWLEDGE_BASES_DIR)
    if legacy.exists():
        for kb_root in sorted(legacy.glob("kb_*")):
            if not kb_root.is_dir() or kb_root.name == "runs":
                continue
            manifest_path = kb_root / "manifest.json"
            entry = {
                "name": kb_root.name,
                "path": str(kb_root),
                "ready": manifest_path.exists(),
                "run_id": None,
            }
            if manifest_path.exists():
                entry["manifest"] = read_json(manifest_path)
            out.append(entry)
    return out


def kb_store(base_or_full: str) -> tuple[ChromaStore, str]:
    """Return (store, collection_name) for a KB."""
    name, chroma_path, _root = resolve_kb(base_or_full)
    ensure_dir(chroma_path)
    return ChromaStore(path=str(chroma_path)), name


def write_kb_manifest(
    base_name: str,
    *,
    chunk_count: int,
    extra: dict | None = None,
) -> Path:
    name, chroma_path, root = resolve_kb(base_name)
    ensure_dir(root)
    base = base_name.split("__", 1)[0] if "__" in base_name else base_name
    meta = _kb_meta(base)
    run = get_run()
    manifest = {
        "name": name,
        "base_name": base,
        "pipeline": meta["pipeline"],
        "description": meta["description"],
        "collection": name,
        "chroma_path": str(chroma_path),
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_count": chunk_count,
        "run_id": run.run_id if run else None,
        "built_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    path = root / "manifest.json"
    write_json(path, manifest)

    index_path = (run.knowledge_bases_dir if run else Path(settings.KNOWLEDGE_BASES_DIR)) / "index.json"
    ensure_dir(index_path.parent)
    write_json(
        index_path,
        {
            "knowledge_bases": list_knowledge_bases(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return path


def _route_by_topic(candidates: list[dict], *, peek: int = 5) -> list[dict]:
    """
    Soft topic routing: look at early hits, pick majority topic_slug
    (ignoring uncategorized when possible), keep same-topic candidates
    if enough remain; otherwise keep all.
    """
    if not candidates:
        return candidates
    peek_hits = candidates[: min(peek, len(candidates))]
    counts: dict[str, int] = {}
    for h in peek_hits:
        slug = (h.get("metadata") or {}).get("topic_slug")
        if not slug or slug == "uncategorized":
            continue
        counts[slug] = counts.get(slug, 0) + 1
    if not counts:
        return candidates
    winner = max(counts, key=counts.get)
    filtered = [
        h for h in candidates if (h.get("metadata") or {}).get("topic_slug") == winner
    ]
    # Need at least 2 same-topic hits or routing is pointless / harmful
    if len(filtered) >= 2:
        return filtered
    return candidates


def retrieve(
    kb_name: str,
    query: str,
    top_k: int | None = None,
    *,
    use_mmr: bool | None = None,
    topic_route: bool | None = None,
) -> list[dict]:
    """
    Retrieve with optional topic soft-routing + MMR diversity.

    Fetches RETRIEVE_CANDIDATES, optionally narrows by majority topic among
    early hits, then applies MMR down to top_k.
    """
    k = top_k or settings.TOP_K
    fetch_k = max(k, settings.RETRIEVE_CANDIDATES)
    do_mmr = settings.MMR_ENABLED if use_mmr is None else use_mmr
    do_route = settings.TOPIC_ROUTE_ENABLED if topic_route is None else topic_route

    store, collection = kb_store(kb_name)
    embedder = Embedder()
    query_embedding = embedder.embed([query])[0]

    # Only apply topic routing when this collection has topic tags
    candidates = store.search(
        collection,
        query_embedding,
        top_k=fetch_k,
        include_embeddings=do_mmr,
    )

    has_topics = any((c.get("metadata") or {}).get("topic_slug") for c in candidates)
    if do_route and has_topics:
        candidates = _route_by_topic(candidates)

    if do_mmr:
        selected = mmr_select(
            query_embedding,
            candidates,
            top_k=k,
            lambda_mult=settings.MMR_LAMBDA,
        )
    else:
        selected = candidates[:k]

    # Drop embeddings from return payload (large / not needed by callers)
    for item in selected:
        item.pop("embedding", None)
    return selected


def ask(kb_name: str, question: str, top_k: int | None = None) -> dict:
    hits = retrieve(kb_name, question, top_k=top_k)
    contexts = [h["content"] for h in hits]
    answer = answer_with_context(question, contexts)
    return {
        "knowledge_base": kb_name,
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "hits": hits,
    }
