"""
Re-run clustering / rebuild topic KB without full pipeline cost.

  # Re-index existing tags+embeddings into KB (cheapest A+D rebuild)
  uv run scripts/run_topic_clustering.py --run-id <id> --rebuild-kb-only

  # Re-cluster + relabel + rebuild KB (reuse embeddings)
  uv run scripts/run_topic_clustering.py --run-id <id> --reuse-embeddings --rebuild-kb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from src.config import settings
from src.io_utils import read_jsonl
from src.knowledge_base import write_kb_manifest
from src.pipeline_topic_modeling_rag.pipeline_topic_modeling_rag import (  # noqa: E402
    _store_tagged_originals,
    run_clustering_only,
)
from src.pipeline_topic_modeling_rag.topic_discovery import load_embeddings_cache
from src.run_context import finish_run, get_latest_run_id, load_run

console = Console()


def rebuild_kb_only(run_id: str) -> None:
    run = load_run(run_id)
    run.pipeline = "rebuild_topic_kb_only"
    out_root = run.artifacts_dir / "topic_modeling"
    tags_path = out_root / "topic_tags.jsonl"
    if not tags_path.exists():
        raise FileNotFoundError(f"Missing {tags_path}")

    console.rule("[bold]rebuild topic KB from tagged originals[/bold]")
    console.print(f"[bold cyan]run_id:[/bold cyan] {run.run_id}")
    big_chunks = read_jsonl(out_root / "big_chunks.jsonl")
    emb, cached_ids = load_embeddings_cache(out_root)
    if [c["chunk_id"] for c in big_chunks] != cached_ids:
        raise ValueError("chunk ids out of sync with embeddings cache")
    tags = read_jsonl(tags_path)

    collection, coverage = _store_tagged_originals(
        run=run,
        big_chunks=big_chunks,
        emb=emb,
        tags=tags,
        out_root=out_root,
    )
    manifest = write_kb_manifest(
        settings.KB_TOPIC_MODELING,
        chunk_count=len(big_chunks),
        extra={"coverage": coverage},
    )
    meta = finish_run(
        status="completed",
        stage="rebuild_kb_only",
        knowledge_base=collection,
        coverage=coverage,
        index_mode="topic_tagged_originals",
    )
    console.print(f"[green]KB:[/green] {collection}")
    console.print(f"[green]Manifest:[/green] {manifest}")
    console.print(f"[green]Run metadata:[/green] {meta}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster / rebuild topic KB")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--reuse-embeddings", action="store_true", default=True)
    parser.add_argument("--reembed", action="store_true")
    parser.add_argument(
        "--rebuild-kb-only",
        action="store_true",
        help="Skip clustering; re-index from existing topic_tags.jsonl + embeddings.npy",
    )
    parser.add_argument(
        "--rebuild-kb",
        action="store_true",
        help="After re-clustering, also rebuild the topic KB from tagged originals",
    )
    parser.add_argument(
        "--build-topic-docs",
        action="store_true",
        help="Also write optional LLM overview markdown (NOT used for retrieval)",
    )
    parser.add_argument("--from-latest", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id
    if args.from_latest:
        run_id = get_latest_run_id()
        if not run_id:
            raise SystemExit("No runs found — run a pipeline first")
        print(f"Using latest run_id={run_id}")

    if args.rebuild_kb_only:
        if not run_id:
            raise SystemExit("--rebuild-kb-only requires --run-id or --from-latest")
        rebuild_kb_only(run_id)
        return

    reuse = not args.reembed
    if not (args.run_id or args.from_latest) and not args.reembed:
        reuse = False

    run_clustering_only(
        run_id=run_id,
        reuse_embeddings=reuse,
        build_topic_docs=args.build_topic_docs,
        rebuild_kb=args.rebuild_kb,
    )


if __name__ == "__main__":
    main()
