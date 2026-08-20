"""pipeline_topic_modeling_rag — Topic Modeling RAG orchestrator (run-scoped)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console

from src.chunker import chunk_records
from src.config import settings
from src.contextualizer import Contextualizer
from src.embedder import Embedder
from src.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl
from src.knowledge_base import kb_store, write_kb_manifest
from src.loader import load_raw_documents
from src.pipeline_topic_modeling_rag.topic_discovery import (
    discover_topics,
    load_embeddings_cache,
    save_embeddings_cache,
)
from src.pipeline_topic_modeling_rag.topic_doc_builder import build_topic_documents
from src.pipeline_topic_modeling_rag.topic_tagger import tag_chunks
from src.run_context import finish_run, get_run, load_run, start_run

console = Console()


def _prepare_chunks_and_embeddings(
    out_root: Path,
    raw_dir: str | Path | None,
    *,
    reuse_embeddings: bool,
) -> tuple[list[dict], np.ndarray]:
    chunks_path = out_root / "big_chunks.jsonl"
    if reuse_embeddings:
        emb, cached_ids = load_embeddings_cache(out_root)
        if chunks_path.exists():
            big_chunks = read_jsonl(chunks_path)
        else:
            raise FileNotFoundError(
                f"Reuse embeddings requested but missing {chunks_path}"
            )
        ids = [c["chunk_id"] for c in big_chunks]
        if ids != cached_ids:
            raise ValueError(
                "Cached embedding chunk_ids do not match big_chunks.jsonl. "
                "Re-run without --reuse-embeddings."
            )
        console.print(
            f"[green]Reused embeddings[/green] ({emb.shape[0]} x {emb.shape[1]}) "
            f"from {out_root / 'embeddings.npy'}"
        )
        return big_chunks, emb

    console.print("[cyan]1. Load + big chunks...[/cyan]")
    records = load_raw_documents(raw_dir)
    big_chunks = chunk_records(
        records,
        chunk_size=settings.BIG_CHUNK_SIZE,
        chunk_overlap=settings.BIG_CHUNK_OVERLAP,
    )
    write_jsonl(chunks_path, big_chunks)
    console.print(f"  {len(big_chunks)} big chunks")

    console.print("[cyan]2. Embed big chunks (for clustering only)...[/cyan]")
    embedder = Embedder()
    cluster_embeddings = embedder.embed([c["content"] for c in big_chunks])
    save_embeddings_cache(out_root, big_chunks, cluster_embeddings)
    console.print(f"[green]Cached embeddings →[/green] {out_root / 'embeddings.npy'}")
    return big_chunks, np.asarray(cluster_embeddings, dtype=np.float32)


def run_clustering_only(
    *,
    run_id: str | None = None,
    reuse_embeddings: bool = True,
    raw_dir: str | Path | None = None,
    build_topic_docs: bool = False,
) -> dict:
    """
    Re-run clustering + LLM sample labeling.
    Default: reuse cached embeddings from the run (skip embed API cost).
    """
    if run_id:
        run = load_run(run_id)
        run.pipeline = "topic_clustering_only"
    else:
        run = start_run("topic_clustering_only")
    console.rule("[bold]topic clustering[/bold]")
    console.print(f"[bold cyan]run_id:[/bold cyan] {run.run_id}  ← timestamp id (see runs/latest.json)")
    out_root = run.artifacts_dir / "topic_modeling"
    ensure_dir(out_root)

    try:
        big_chunks, emb = _prepare_chunks_and_embeddings(
            out_root, raw_dir, reuse_embeddings=reuse_embeddings
        )
        console.print("[cyan]Cluster + LLM sample labels...[/cyan]")
        _model, topics, probs, catalog = discover_topics(
            big_chunks, emb, artifacts_dir=out_root
        )
        tags = tag_chunks(
            big_chunks,
            topics,
            probs,
            catalog,
            out_path=out_root / "topic_tags.jsonl",
        )
        result: dict = {"catalog": catalog, "tags": tags}

        if build_topic_docs:
            console.print("[cyan]Building topic docs...[/cyan]")
            index = build_topic_documents(
                big_chunks, tags, catalog, out_root / "topic_docs"
            )
            result["index"] = index

        meta = finish_run(
            status="completed",
            stage="clustering",
            topics_found=len([t for t in catalog if t["topic_id"] != -1]),
            reused_embeddings=reuse_embeddings,
        )
        console.print(f"[green]Run metadata:[/green] {meta}")
        console.print(f"[cyan]Token usage:[/cyan] {run.usage.as_dict()}")
        return result
    except Exception:
        finish_run(status="failed")
        raise


def run_pipeline_topic_modeling_rag(
    raw_dir: str | Path | None = None,
    *,
    run_id: str | None = None,
    reuse_embeddings: bool = False,
) -> dict:
    run = start_run("pipeline_topic_modeling_rag", run_id=run_id) if not run_id else load_run(run_id)
    if run_id:
        run.pipeline = "pipeline_topic_modeling_rag"
    console.rule("[bold]pipeline_topic_modeling_rag[/bold]")
    console.print(f"[bold cyan]run_id:[/bold cyan] {run.run_id}  ← timestamp id (see runs/latest.json)")
    out_root = run.artifacts_dir / "topic_modeling"
    ensure_dir(out_root)

    try:
        big_chunks, emb = _prepare_chunks_and_embeddings(
            out_root, raw_dir, reuse_embeddings=reuse_embeddings
        )

        console.print("[cyan]3. Topic discovery (cluster + LLM samples)...[/cyan]")
        _model, topics, probs, catalog = discover_topics(
            big_chunks, emb, artifacts_dir=out_root
        )

        console.print("[cyan]4. Tag source chunks...[/cyan]")
        tags = tag_chunks(
            big_chunks,
            topics,
            probs,
            catalog,
            out_path=out_root / "topic_tags.jsonl",
        )

        console.print("[cyan]5. Iterative LLM topic document building...[/cyan]")
        docs_dir = out_root / "topic_docs"
        index = build_topic_documents(big_chunks, tags, catalog, docs_dir)

        console.print("[cyan]6. Re-chunk + contextualize + embed topic docs...[/cyan]")
        final_chunks: list[dict] = []
        for slug, meta in index.items():
            path = out_root / meta["path"]
            text = path.read_text(encoding="utf-8")
            records = [{"text": text, "source": f"{slug}.md", "page": 0}]
            small = chunk_records(
                records,
                chunk_size=settings.SMALL_CHUNK_SIZE,
                chunk_overlap=settings.SMALL_CHUNK_OVERLAP,
            )
            for sc in small:
                sc["topic_slug"] = slug
                sc["source_doc_slug"] = slug
                sc["chunk_id"] = f"{slug}_{sc['chunk_id']}"
                final_chunks.append(sc)

        console.print(f"  {len(final_chunks)} small chunks from topic docs")
        contextualizer = Contextualizer()
        contexts = contextualizer.contextualize_many(
            final_chunks, desc="Contextualizing topic chunks"
        )

        enriched: list[dict] = []
        for chunk, context in zip(final_chunks, contexts):
            embed_text = f"{context}\n\n{chunk['content']}"
            enriched.append({**chunk, "context": context, "embed_text": embed_text})

        embedder = Embedder()
        embeddings = embedder.embed([c["embed_text"] for c in enriched])
        kb_base = settings.KB_TOPIC_MODELING
        console.print(f"[cyan]Storing knowledge base:[/cyan] {run.kb_name(kb_base)}")
        store, collection = kb_store(kb_base)
        store.add(
            collection,
            ids=[c["chunk_id"] for c in enriched],
            documents=[c["content"] for c in enriched],
            embeddings=embeddings,
            metadatas=[
                {
                    "chunk_id": c["chunk_id"],
                    "topic_slug": c["topic_slug"],
                    "source_doc_slug": c["source_doc_slug"],
                    "source": c["source"],
                    "page": int(c["page"]),
                    "knowledge_base": collection,
                    "run_id": run.run_id,
                }
                for c in enriched
            ],
            reset=True,
        )
        write_jsonl(out_root / "final_chunks.jsonl", enriched)

        total = len(big_chunks)
        uncategorized = sum(1 for t in tags if t["topic_slug"] == "uncategorized")
        covered = total - uncategorized
        coverage = {
            "total_chunks": total,
            "uncategorized": uncategorized,
            "covered": covered,
            "coverage_pct": round((covered / total * 100) if total else 0.0, 2),
        }
        write_json(out_root / "coverage.json", coverage)
        manifest = write_kb_manifest(
            kb_base, chunk_count=len(enriched), extra={"coverage": coverage}
        )
        meta_path = finish_run(
            status="completed",
            coverage=coverage,
            knowledge_base=collection,
            topics_found=len([t for t in catalog if t["topic_id"] != -1]),
        )
        console.print(
            f"[green]Coverage:[/green] {coverage['coverage_pct']}% "
            f"({covered}/{total}); outliers excluded: {uncategorized}"
        )
        console.print(f"[green]Knowledge base:[/green] {collection}")
        console.print(f"[green]Manifest:[/green] {manifest}")
        console.print(f"[green]Run metadata:[/green] {meta_path}")
        console.print(f"[cyan]Token usage:[/cyan] {get_run().usage.as_dict()}")
        return {"catalog": catalog, "coverage": coverage, "index": index}
    except Exception:
        finish_run(status="failed")
        raise
