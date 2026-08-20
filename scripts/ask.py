"""
Ask one question against contextual + topic KBs from chosen run ids.

Inspect retrieved chunks (no answer generation):
  uv run scripts/ask.py --contextual-run ... --topic-run ... --inspect-only "What is RPI Sentinel?"

Show answer + chunk metadata:
  uv run scripts/ask.py --contextual-run ... --topic-run ... --show-contexts "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import settings
from src.rag import ask_both, inspect_both

console = Console()


def _print_hits(title: str, hits: list[dict], *, color: str, preview: int = 220) -> None:
    console.print(f"\n[bold {color}]{title}[/bold {color}]  ({len(hits)} hits)")
    for i, h in enumerate(hits, 1):
        dist = h.get("distance")
        dist_s = f"{dist:.4f}" if isinstance(dist, (int, float)) else "?"
        topic = h.get("topic_slug") or "-"
        source = h.get("source") or "?"
        chunk_id = h.get("chunk_id") or "?"
        content = (h.get("content") or "").replace("\n", " ")
        if len(content) > preview:
            content = content[:preview] + "…"
        console.print(
            f"  [{color}]#{i}[/{color}] dist={dist_s}  topic={topic}\n"
            f"      source={source}\n"
            f"      id={chunk_id}\n"
            f"      {content}"
        )


def _hits_table(hits: list[dict]) -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", width=3)
    table.add_column("dist", width=8)
    table.add_column("topic")
    table.add_column("source")
    for i, h in enumerate(hits, 1):
        dist = h.get("distance")
        dist_s = f"{dist:.4f}" if isinstance(dist, (int, float)) else "?"
        table.add_row(
            str(i),
            dist_s,
            str(h.get("topic_slug") or "-"),
            str(h.get("source") or "?"),
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask both RAGs using explicit run ids for each knowledge base"
    )
    parser.add_argument("--contextual-run", required=True)
    parser.add_argument("--topic-run", required=True)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--top-k", type=int, default=settings.TOP_K)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--show-contexts",
        action="store_true",
        help="Print retrieved chunk source/topic/distance + preview",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only retrieve+print chunks (no LLM answer) — best for spotting bad chunks",
    )
    args = parser.parse_args()

    question = args.question or console.input("[bold]Question:[/bold] ").strip()
    if not question:
        raise SystemExit("No question provided.")

    console.print(
        f"[dim]contextual KB ← run {args.contextual_run}\n"
        f"topic KB       ← run {args.topic_run}[/dim]\n"
    )

    if args.inspect_only:
        result = inspect_both(
            question,
            contextual_run_id=args.contextual_run,
            topic_run_id=args.topic_run,
            top_k=args.top_k,
        )
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return
        console.print(Panel(question, title="Question (inspect-only)", border_style="cyan"))
        _print_hits("Contextual RAG hits", result["contextual_rag"]["hits"], color="green")
        console.print(_hits_table(result["contextual_rag"]["hits"]))
        _print_hits("Topic RAG hits", result["topic_modeling_rag"]["hits"], color="magenta")
        console.print(_hits_table(result["topic_modeling_rag"]["hits"]))
        console.print(
            "\n[dim]Check: are sources on-topic? any near-dup blurbs? wrong product file?[/dim]"
        )
        return

    result = ask_both(
        question,
        contextual_run_id=args.contextual_run,
        topic_run_id=args.topic_run,
        top_k=args.top_k,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    console.print(Panel(question, title="Question", border_style="cyan"))

    ctx = result["contextual_rag"]
    console.print(
        Panel(
            ctx["answer"],
            title=f"Contextual RAG  |  {ctx['knowledge_base']}",
            border_style="green",
        )
    )
    if args.show_contexts:
        _print_hits("Contextual hits", ctx["hits"], color="green")

    topic = result["topic_modeling_rag"]
    console.print(
        Panel(
            topic["answer"],
            title=f"Topic Modeling RAG  |  {topic['knowledge_base']}",
            border_style="magenta",
        )
    )
    if args.show_contexts:
        _print_hits("Topic hits", topic["hits"], color="magenta")


if __name__ == "__main__":
    main()
