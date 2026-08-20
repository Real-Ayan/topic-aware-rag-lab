"""Thin OpenAI chat helper compatible with GPT-5.6 reasoning models."""

from __future__ import annotations

from openai import OpenAI

from src.config import settings
from src.run_context import get_run

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def _is_reasoning_model(model: str) -> bool:
    name = model.lower()
    return name.startswith("gpt-5") or name.startswith("o1") or name.startswith("o3") or name.startswith("o4")


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    max_completion_tokens: int = 2048,
    temperature: float | None = None,
    response_format: dict | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Chat Completions wrapper that omits params unsupported by GPT-5.x."""
    model = model or settings.LLM_MODEL
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }

    if response_format is not None:
        kwargs["response_format"] = response_format

    if _is_reasoning_model(model):
        effort = reasoning_effort or settings.REASONING_EFFORT
        if effort:
            kwargs["reasoning_effort"] = effort
    elif temperature is not None:
        kwargs["temperature"] = temperature

    response = get_client().chat.completions.create(**kwargs)

    usage = getattr(response, "usage", None)
    run = get_run()
    if run is not None:
        if usage is not None:
            run.usage.add_llm(
                prompt=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion=int(getattr(usage, "completion_tokens", 0) or 0),
                total=int(getattr(usage, "total_tokens", 0) or 0) or None,
            )
        else:
            run.usage.add_llm()
        # Persist incrementally so a crash still leaves usable counts
        if run.usage.llm_calls % 5 == 0:
            run.write_metadata(status="running")

    return (response.choices[0].message.content or "").strip()
