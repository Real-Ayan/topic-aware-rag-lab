"""pipeline_contextual_rag — Contextual RAG (run-scoped)."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from src.chunker import chunk_records
from src.config import settings
from src.contextualizer import Contextualizer
from src.embedder import Embedder
from src.io_utils import ensure_dir, write_jsonl
from src.knowledge_base import kb_store, write_kb_manifest
from src.loader import load_raw_documents
from src.run_context import finish_run, get_run, start_run

console = Console()


def run_pipeline_contextual_rag(
    raw_dir: str | Path | None = None,
    *,
    run_id: str | None = None,
) -> list[dict]:
    run = start_run("pipeline_contextual_rag", run_id=run_id)
    console.rule(f"[bold]pipeline_contextual_rag[/bold]")
    console.print(f"[bold cyan]run_id:[/bold cyan] {run.run_id}  ← timestamp id (see runs/latest.json)")
    out_root = run.artifacts_dir / "contextual"
    ensure_dir(out_root)

    try:
        console.print("[cyan]1. Loading raw documents...[/cyan]")
        records = load_raw_documents(raw_dir)

        console.print("[cyan]2. Chunking (big chunks)...[/cyan]")
        chunks = chunk_records(
            records,
            chunk_size=settings.BIG_CHUNK_SIZE,
            chunk_overlap=settings.BIG_CHUNK_OVERLAP,
        )
        console.print(f"  {len(chunks)} chunks")

        console.print("[cyan]3. Contextualizing chunks (LLM)...[/cyan]")
        contextualizer = Contextualizer()
        contexts = contextualizer.contextualize_many(chunks)

        enriched: list[dict] = []
        for chunk, context in zip(chunks, contexts):
            embed_text = f"{context}\n\n{chunk['content']}"
            enriched.append({**chunk, "context": context, "embed_text": embed_text})

        console.print("[cyan]4. Embedding...[/cyan]")
        embedder = Embedder()
        embeddings = embedder.embed([c["embed_text"] for c in enriched])

        kb_base = settings.KB_CONTEXTUAL
        console.print(f"[cyan]5. Storing knowledge base:[/cyan] {run.kb_name(kb_base)}")
        store, collection = kb_store(kb_base)
        store.add(
            collection,
            ids=[c["chunk_id"] for c in enriched],
            documents=[c["content"] for c in enriched],
            embeddings=embeddings,
            metadatas=[
                {
                    "chunk_id": c["chunk_id"],
                    "source": c["source"],
                    "page": int(c["page"]),
                    "context": c["context"][:500],
                    "knowledge_base": collection,
                    "run_id": run.run_id,
                }
                for c in enriched
            ],
            reset=True,
        )

        write_jsonl(out_root / "chunks.jsonl", enriched)
        manifest = write_kb_manifest(kb_base, chunk_count=len(enriched))
        meta_path = finish_run(
            status="completed",
            chunk_count=len(enriched),
            knowledge_base=collection,
        )
        console.print(f"[green]Wrote[/green] {out_root / 'chunks.jsonl'}")
        console.print(f"[green]Knowledge base:[/green] {collection}")
        console.print(f"[green]Manifest:[/green] {manifest}")
        console.print(f"[green]Run metadata:[/green] {meta_path}")
        console.print(f"[cyan]Token usage:[/cyan] {get_run().usage.as_dict()}")
        return enriched
    except Exception:
        finish_run(status="failed")
        raise
