"""List experiment runs (timestamp ids, newest first)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table

from src.run_context import RUNS_DIR, get_latest_run_id

console = Console()


def main() -> None:
    latest = get_latest_run_id()
    index_path = RUNS_DIR / "index.json"
    runs: list[dict] = []
    if index_path.exists():
        runs = json.loads(index_path.read_text(encoding="utf-8")).get("runs") or []
    else:
        for p in sorted(RUNS_DIR.glob("*/run_metadata.json"), reverse=True):
            data = json.loads(p.read_text(encoding="utf-8"))
            runs.append(
                {
                    "run_id": data.get("run_id", p.parent.name),
                    "pipeline": data.get("pipeline"),
                    "status": data.get("status"),
                    "started_at_local": data.get("started_at_local"),
                    "path": str(p.parent),
                }
            )
        runs.sort(key=lambda r: r["run_id"], reverse=True)

    table = Table(title="Runs (newest first)")
    table.add_column("run_id")
    table.add_column("latest?")
    table.add_column("pipeline")
    table.add_column("status")
    table.add_column("started_at_local")
    for r in runs:
        rid = r.get("run_id", "")
        table.add_row(
            rid,
            "← latest" if rid == latest else "",
            str(r.get("pipeline") or ""),
            str(r.get("status") or ""),
            str(r.get("started_at_local") or ""),
        )
    console.print(table)
    if latest:
        console.print(f"\n[green]Latest run_id:[/green] {latest}")
        console.print(f"[dim]Pointer:[/dim] {RUNS_DIR / 'latest.json'}")


if __name__ == "__main__":
    main()
