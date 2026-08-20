"""Simple dual-RAG: pick contextual + topic runs, ask one question."""

from __future__ import annotations

from src.config import settings
from src.knowledge_base import ask


def kb_name_for_run(base: str, run_id: str) -> str:
    return f"{base}__{run_id}"


def ask_both(
    question: str,
    *,
    contextual_run_id: str,
    topic_run_id: str,
    top_k: int | None = None,
) -> dict:
    """
    Ask one question against KBs from two run ids.

    Knowledge bases are resolved as:
      kb_contextual_rag__{contextual_run_id}
      kb_topic_modeling_rag__{topic_run_id}
    """
    ctx_kb = kb_name_for_run(settings.KB_CONTEXTUAL, contextual_run_id)
    topic_kb = kb_name_for_run(settings.KB_TOPIC_MODELING, topic_run_id)

    contextual = ask(ctx_kb, question, top_k=top_k)
    topic = ask(topic_kb, question, top_k=top_k)

    return {
        "question": question,
        "contextual_rag": {
            "run_id": contextual_run_id,
            "knowledge_base": ctx_kb,
            "answer": contextual["answer"],
            "contexts": contextual["contexts"],
        },
        "topic_modeling_rag": {
            "run_id": topic_run_id,
            "knowledge_base": topic_kb,
            "answer": topic["answer"],
            "contexts": topic["contexts"],
        },
    }
