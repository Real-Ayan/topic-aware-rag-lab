"""
Ask one question against contextual + topic KBs from chosen run ids.

Example (your current runs):
  uv run scripts/ask.py \\
    --contextual-run 20260820_172037 \\
    --topic-run 20260820_172419 \\
    "What AI products does RPI offer?"
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

from src.config import settings
from src.rag import ask_both

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask both RAGs using explicit run ids for each knowledge base"
    )
    parser.add_argument(
        "--contextual-run",
        required=True,
        help="Run id that built kb_contextual_rag (e.g. 20260820_172037)",
    )
    parser.add_argument(
        "--topic-run",
        required=True,
        help="Run id that built kb_topic_modeling_rag (e.g. 20260820_172419)",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question text (omit to be prompted)",
    )
    parser.add_argument("--top-k", type=int, default=settings.TOP_K)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-contexts", action="store_true")
    args = parser.parse_args()

    question = args.question or console.input("[bold]Question:[/bold] ").strip()
    if not question:
        raise SystemExit("No question provided.")

    console.print(
        f"[dim]contextual KB ← run {args.contextual_run}\n"
        f"topic KB       ← run {args.topic_run}[/dim]\n"
    )

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
        for i, c in enumerate(ctx["contexts"], 1):
            console.print(f"  [green]#{i}[/green] {c[:300]}{'…' if len(c) > 300 else ''}")

    topic = result["topic_modeling_rag"]
    console.print(
        Panel(
            topic["answer"],
            title=f"Topic Modeling RAG  |  {topic['knowledge_base']}",
            border_style="magenta",
        )
    )
    if args.show_contexts:
        for i, c in enumerate(topic["contexts"], 1):
            console.print(f"  [magenta]#{i}[/magenta] {c[:300]}{'…' if len(c) > 300 else ''}")


if __name__ == "__main__":
    main()
