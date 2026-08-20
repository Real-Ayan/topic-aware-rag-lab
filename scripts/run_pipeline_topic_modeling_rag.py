"""Entry: run pipeline_topic_modeling_rag (new run, caches embeddings)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline_topic_modeling_rag import run_pipeline_topic_modeling_rag  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="Resume/use this run id")
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Skip embedding; load embeddings.npy from the run's topic_modeling artifacts",
    )
    parser.add_argument(
        "--build-topic-docs",
        action="store_true",
        help="Also write optional LLM overview docs (not indexed for retrieval)",
    )
    args = parser.parse_args()
    run_pipeline_topic_modeling_rag(
        run_id=args.run_id,
        reuse_embeddings=args.reuse_embeddings,
        build_topic_docs=args.build_topic_docs,
    )


if __name__ == "__main__":
    main()
