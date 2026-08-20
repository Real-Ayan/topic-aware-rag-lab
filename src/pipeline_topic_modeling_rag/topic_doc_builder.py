"""Iterative LLM topic document builder."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import track

from src.config import settings
from src.io_utils import ensure_dir, write_json
from src.llm import chat

console = Console()

DRAFT_SYSTEM = (
    "You are building a structured knowledge document for a company knowledge base. "
    "Write a well-structured Markdown document about the given topic. "
    "Use only the information from the provided source chunks. Do not invent facts. "
    "Include a YAML frontmatter block with: slug, sources (list of filenames), chunk_ids_used."
)

UPDATE_SYSTEM = (
    "You are updating a company knowledge document. "
    "Add any new, non-duplicate information from the new source chunks below. "
    "Do not repeat what is already in the document. Do not invent facts. "
    "Update the YAML frontmatter chunk_ids_used list."
)


def _format_chunks(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(
            f"[CHUNK {c['chunk_id']}] (source: {c['source']}, page {c['page']})\n"
            f"{c['content']}"
        )
    return "\n---\n".join(blocks)


def _batched(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_topic_documents(
    chunks: list[dict],
    tags: list[dict],
    catalog: list[dict],
    docs_dir: Path,
) -> dict:
    ensure_dir(docs_dir)

    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    slug_to_chunk_ids: dict[str, list[str]] = {}
    for tag in tags:
        slug_to_chunk_ids.setdefault(tag["topic_slug"], []).append(tag["chunk_id"])

    index: dict = {}
    topics = [t for t in catalog if t["topic_id"] != -1]

    for topic in track(topics, description="Building topic docs"):
        slug = topic["slug"]
        label = topic["label"]
        ids = slug_to_chunk_ids.get(slug, [])
        topic_chunks = [chunk_by_id[i] for i in ids if i in chunk_by_id]
        topic_chunks.sort(key=lambda c: (c["source"], c["page"], c["chunk_id"]))

        if not topic_chunks:
            continue

        batches = _batched(topic_chunks, settings.TOPIC_DOC_BATCH_SIZE)
        current_doc = ""

        for bi, batch in enumerate(batches):
            if bi == 0:
                user = (
                    f"Topic: {label}\n"
                    f"Slug: {slug}\n"
                    f"Source chunks:\n---\n{_format_chunks(batch)}\n---"
                )
                messages = [
                    {"role": "system", "content": DRAFT_SYSTEM},
                    {"role": "user", "content": user},
                ]
            else:
                user = (
                    f"Current document:\n{current_doc}\n\n"
                    f"New source chunks:\n---\n{_format_chunks(batch)}\n---"
                )
                messages = [
                    {"role": "system", "content": UPDATE_SYSTEM},
                    {"role": "user", "content": user},
                ]

            current_doc = chat(
                messages,
                model=settings.LLM_MODEL,
                max_completion_tokens=4096,
                temperature=0.2,
                reasoning_effort="low",
            )

        out_path = docs_dir / f"{slug}.md"
        out_path.write_text(current_doc + "\n", encoding="utf-8")

        sources = sorted({c["source"] for c in topic_chunks})
        index[slug] = {
            "slug": slug,
            "label": label,
            "path": f"topic_docs/{slug}.md",
            "source_chunk_count": len(topic_chunks),
            "sources": sources,
        }
        console.print(f"  wrote {out_path.name} ({len(topic_chunks)} chunks)")

    write_json(docs_dir / "_index.json", index)
    return index
