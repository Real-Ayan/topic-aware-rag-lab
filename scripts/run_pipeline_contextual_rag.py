"""Entry: run pipeline_contextual_rag (creates a new run + run_metadata.json)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline_contextual_rag import run_pipeline_contextual_rag  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="Optional fixed run id (else auto)")
    args = parser.parse_args()
    run_pipeline_contextual_rag(run_id=args.run_id)


if __name__ == "__main__":
    main()
