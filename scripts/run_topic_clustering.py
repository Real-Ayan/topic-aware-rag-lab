"""
Re-run clustering + LLM sample labeling without re-embedding.

  1) Run topic pipeline once (caches embeddings.npy)
  2) Change cluster knobs in .env
  3) uv run scripts/run_topic_clustering.py --run-id <id> --reuse-embeddings

General knobs (not corpus-specific): see README "Clustering — general ideas".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline_topic_modeling_rag.pipeline_topic_modeling_rag import (  # noqa: E402
    run_clustering_only,
)
from src.run_context import RUNS_DIR, get_latest_run_id  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-only (skip embeddings by default)")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Existing run id that already has embeddings.npy (or omit to create new + embed)",
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        default=True,
        help="Reuse cached embeddings (default: true)",
    )
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Force re-embed (ignore cache)",
    )
    parser.add_argument(
        "--build-topic-docs",
        action="store_true",
        help="Also rebuild topic markdown docs after labeling",
    )
    parser.add_argument(
        "--from-latest",
        action="store_true",
        help="Use runs/latest.json run_id",
    )
    args = parser.parse_args()

    run_id = args.run_id
    if args.from_latest:
        run_id = get_latest_run_id()
        if not run_id:
            raise SystemExit("No runs found — run a pipeline first")
        print(f"Using latest run_id={run_id}")

    reuse = not args.reembed
    if args.run_id or args.from_latest:
        reuse = not args.reembed
    elif not args.reembed:
        reuse = False

    run_clustering_only(
        run_id=run_id,
        reuse_embeddings=reuse,
        build_topic_docs=args.build_topic_docs,
    )


if __name__ == "__main__":
    main()
