"""Token-aware chunking via LangChain + tiktoken."""

from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


def _splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _chunk_id_prefix(source: str) -> str:
    """Build a filesystem-safe, unique prefix from a relative source path."""
    path = Path(source)
    # Include parent folder when nested so IDs stay unique across dirs
    parts = list(path.parts[:-1]) + [path.stem]
    slug = "_".join(parts)
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in slug)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "doc"


def chunk_records(
    records: list[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Split load records into chunk dicts with stable chunk_ids."""
    splitter = _splitter(chunk_size, chunk_overlap)
    chunks: list[dict] = []

    for record in records:
        text = record["text"]
        source = record["source"]
        page = record["page"]
        prefix = _chunk_id_prefix(source)
        pieces = splitter.split_text(text)

        cursor = 0
        for i, content in enumerate(pieces, start=1):
            start = text.find(content, cursor)
            if start < 0:
                start = cursor
            end = start + len(content)
            cursor = end
            chunks.append(
                {
                    "chunk_id": f"{prefix}_p{page}_c{i:03d}",
                    "content": content,
                    "source": source,
                    "page": page,
                    "char_start": start,
                    "char_end": end,
                }
            )

    return chunks
