"""CLI to list / query run-scoped knowledge bases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table

from src.config import settings
from src.knowledge_base import ask, list_knowledge_bases, retrieve
from src.run_context import RUNS_DIR, load_run

console = Console()


def cmd_list(_: argparse.Namespace) -> None:
    table = Table(title="Knowledge bases (all runs)")
    table.add_column("Name")
    table.add_column("Run")
    table.add_column("Ready")
    table.add_column("Chunks")
    table.add_column("Path")
    for kb in list_knowledge_bases():
        chunks = ""
        if kb.get("manifest"):
            chunks = str(kb["manifest"].get("chunk_count", ""))
        table.add_row(
            kb["name"],
            str(kb.get("run_id") or "-"),
            "yes" if kb["ready"] else "no",
            chunks,
            kb["path"],
        )
    console.print(table)


def cmd_query(args: argparse.Namespace) -> None:
    if args.run_id:
        load_run(args.run_id)
    hits = retrieve(args.kb, args.query, top_k=args.top_k)
    console.print(f"[bold]KB:[/bold] {args.kb}")
    console.print(f"[bold]Query:[/bold] {args.query}\n")
    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata") or {}
        console.print(
            f"[cyan]#{i}[/cyan] distance={hit['distance']:.4f} "
            f"source={meta.get('source', '?')} id={meta.get('chunk_id', '?')}"
        )
        console.print(hit["content"][:500] + ("…" if len(hit["content"]) > 500 else ""))
        console.print()


def cmd_ask(args: argparse.Namespace) -> None:
    if args.run_id:
        load_run(args.run_id)
    result = ask(args.kb, args.question, top_k=args.top_k)
    console.print(f"[bold]KB:[/bold] {result['knowledge_base']}")
    console.print(f"[bold]Q:[/bold] {result['question']}")
    console.print(f"\n[green]Answer:[/green]\n{result['answer']}\n")
    if args.json:
        print(
            json.dumps(
                {k: result[k] for k in ("knowledge_base", "question", "answer", "contexts")},
                indent=2,
            )
        )


def cmd_compare(args: argparse.Namespace) -> None:
    run = load_run(args.run_id) if args.run_id else None
    if run is None and (RUNS_DIR / "latest.json").exists():
        rid = json.loads((RUNS_DIR / "latest.json").read_text())["run_id"]
        run = load_run(rid)
        console.print(f"[dim]Using latest run {rid}[/dim]")
    if run is None:
        raise SystemExit("Pass --run-id or run a pipeline first")

    for base in (settings.KB_CONTEXTUAL, settings.KB_TOPIC_MODELING):
        kb = run.kb_name(base)
        console.rule(kb)
        try:
            result = ask(kb, args.question, top_k=args.top_k)
            console.print(result["answer"])
        except Exception as exc:
            console.print(f"[red]Failed:[/red] {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query named RAG knowledge bases")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List knowledge bases across runs")
    p_list.set_defaults(func=cmd_list)

    p_query = sub.add_parser("query", help="Retrieve chunks only")
    p_query.add_argument("--kb", required=True, help="Full KB name or base name with --run-id")
    p_query.add_argument("--run-id", default=None)
    p_query.add_argument("--query", required=True)
    p_query.add_argument("--top-k", type=int, default=settings.TOP_K)
    p_query.set_defaults(func=cmd_query)

    p_ask = sub.add_parser("ask", help="Retrieve + answer")
    p_ask.add_argument("--kb", required=True)
    p_ask.add_argument("--run-id", default=None)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--top-k", type=int, default=settings.TOP_K)
    p_ask.add_argument("--json", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_compare = sub.add_parser("compare", help="Ask both KBs in a run")
    p_compare.add_argument("--run-id", default=None)
    p_compare.add_argument("--question", required=True)
    p_compare.add_argument("--top-k", type=int, default=settings.TOP_K)
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
