"""Shared helpers for artifact I/O and answering."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import settings
from src.llm import chat


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, data: object) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def answer_with_context(question: str, contexts: list[str]) -> str:
    context_block = "\n---\n".join(contexts) if contexts else "(no context retrieved)"
    return chat(
        [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided context. "
                    "Be concise and factual. If the context is insufficient, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion: {question}",
            },
        ],
        model=settings.eval_model,
        max_completion_tokens=1024,
        temperature=0.1,
        reasoning_effort="low",
    )
