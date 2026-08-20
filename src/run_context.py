"""Per-run context: isolated artifacts/KBs + token usage tracking."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import ROOT_DIR, settings

RUNS_DIR = ROOT_DIR / "runs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class UsageCounters:
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    embedding_calls: int = 0
    embedding_tokens: int = 0
    embedding_texts: int = 0

    def add_llm(self, prompt: int = 0, completion: int = 0, total: int | None = None) -> None:
        self.llm_calls += 1
        self.llm_prompt_tokens += prompt
        self.llm_completion_tokens += completion
        self.llm_total_tokens += total if total is not None else (prompt + completion)

    def add_embedding(self, tokens: int = 0, texts: int = 0) -> None:
        self.embedding_calls += 1
        self.embedding_tokens += tokens
        self.embedding_texts += texts

    def as_dict(self) -> dict[str, int]:
        return {
            "llm_calls": self.llm_calls,
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_completion_tokens": self.llm_completion_tokens,
            "llm_total_tokens": self.llm_total_tokens,
            "embedding_calls": self.embedding_calls,
            "embedding_tokens": self.embedding_tokens,
            "embedding_texts": self.embedding_texts,
            "estimated_total_tokens": self.llm_total_tokens + self.embedding_tokens,
        }


@dataclass
class RunContext:
    run_id: str
    started_at: str
    pipeline: str = ""
    usage: UsageCounters = field(default_factory=UsageCounters)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return RUNS_DIR / self.run_id

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def knowledge_bases_dir(self) -> Path:
        return self.root / "knowledge_bases"

    @property
    def metadata_path(self) -> Path:
        return self.root / "run_metadata.json"

    def kb_name(self, base: str) -> str:
        """Run-scoped KB folder/collection name."""
        return f"{base}__{self.run_id}"

    def kb_chroma_path(self, base: str) -> str:
        return str(self.knowledge_bases_dir / self.kb_name(base) / "chroma")

    def model_config_snapshot(self) -> dict[str, Any]:
        return {
            "llm_model": settings.LLM_MODEL,
            "context_model": settings.context_model,
            "eval_model": settings.eval_model,
            "embedding_model": settings.EMBEDDING_MODEL,
            "reasoning_effort": settings.REASONING_EFFORT,
            "big_chunk_size": settings.BIG_CHUNK_SIZE,
            "big_chunk_overlap": settings.BIG_CHUNK_OVERLAP,
            "small_chunk_size": settings.SMALL_CHUNK_SIZE,
            "small_chunk_overlap": settings.SMALL_CHUNK_OVERLAP,
            "umap_n_components": settings.UMAP_N_COMPONENTS,
            "umap_n_neighbors": settings.UMAP_N_NEIGHBORS,
            "umap_min_dist": settings.UMAP_MIN_DIST,
            "hdbscan_min_cluster_size": settings.HDBSCAN_MIN_CLUSTER_SIZE,
            "hdbscan_min_samples": settings.HDBSCAN_MIN_SAMPLES,
            "hdbscan_selection_method": settings.HDBSCAN_SELECTION_METHOD,
            "cluster_method": settings.CLUSTER_METHOD,
            "n_topics": settings.N_TOPICS,
            "topic_label_samples": settings.TOPIC_LABEL_SAMPLES,
            "top_k": settings.TOP_K,
        }

    def write_metadata(self, *, status: str = "running", **extra: Any) -> Path:
        ensure_dir(self.root)
        now_utc = datetime.now(timezone.utc).isoformat()
        now_local = datetime.now().astimezone().isoformat(timespec="seconds")
        payload = {
            "run_id": self.run_id,
            "status": status,
            "pipeline": self.pipeline,
            "started_at": self.started_at,
            "started_at_local": self.extra.get("started_at_local"),
            "updated_at": now_utc,
            "updated_at_local": now_local,
            "model_config": self.model_config_snapshot(),
            "token_usage": self.usage.as_dict(),
            "paths": {
                "root": str(self.root),
                "artifacts": str(self.artifacts_dir),
                "knowledge_bases": str(self.knowledge_bases_dir),
                "kb_contextual": self.kb_name(settings.KB_CONTEXTUAL),
                "kb_topic_modeling": self.kb_name(settings.KB_TOPIC_MODELING),
            },
            "notes": self.notes,
            **{k: v for k, v in self.extra.items() if k != "started_at_local"},
            **extra,
        }
        # Keep started_at_local stable on the payload
        if self.extra.get("started_at_local"):
            payload["started_at_local"] = self.extra["started_at_local"]
        write_json(self.metadata_path, payload)
        _update_latest_and_index(self, status=status)
        return self.metadata_path


_current: RunContext | None = None


def new_run_id() -> str:
    """
    Timestamp run id in local time: YYYYMMDD_HHMMSS
    Lexicographic sort == chronological order. Suffix only on collision.
    """
    ensure_dir(RUNS_DIR)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    candidate = stamp
    if (RUNS_DIR / candidate).exists():
        candidate = f"{stamp}_{uuid.uuid4().hex[:4]}"
    return candidate


def _update_latest_and_index(ctx: RunContext, *, status: str) -> None:
    """Write latest.json + index.json so the newest run is always obvious."""
    latest = {
        "run_id": ctx.run_id,
        "pipeline": ctx.pipeline,
        "status": status,
        "started_at_local": ctx.extra.get("started_at_local"),
        "updated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "path": str(ctx.root),
        "metadata": str(ctx.metadata_path),
    }
    write_json(RUNS_DIR / "latest.json", latest)

    runs: list[dict[str, Any]] = []
    for meta_path in sorted(RUNS_DIR.glob("*/run_metadata.json"), reverse=True):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        runs.append(
            {
                "run_id": data.get("run_id", meta_path.parent.name),
                "pipeline": data.get("pipeline"),
                "status": data.get("status"),
                "started_at_local": data.get("started_at_local"),
                "updated_at_local": data.get("updated_at_local"),
                "path": str(meta_path.parent),
            }
        )
    # Timestamp ids sort newest-first when reversed alphabetically
    runs.sort(key=lambda r: r["run_id"], reverse=True)
    write_json(
        RUNS_DIR / "index.json",
        {
            "latest_run_id": ctx.run_id,
            "runs": runs,
            "updated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )


def get_latest_run_id() -> str | None:
    latest_path = RUNS_DIR / "latest.json"
    if latest_path.exists():
        try:
            return json.loads(latest_path.read_text(encoding="utf-8")).get("run_id")
        except Exception:
            pass
    # Fallback: newest timestamp folder name
    dirs = sorted(
        (p for p in RUNS_DIR.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    return dirs[0].name if dirs else None


def start_run(pipeline: str, run_id: str | None = None) -> RunContext:
    global _current
    rid = run_id or new_run_id()
    local_now = datetime.now().astimezone()
    ctx = RunContext(
        run_id=rid,
        started_at=datetime.now(timezone.utc).isoformat(),
        pipeline=pipeline,
        extra={"started_at_local": local_now.isoformat(timespec="seconds")},
    )
    ensure_dir(ctx.artifacts_dir)
    ensure_dir(ctx.knowledge_bases_dir)
    ctx.write_metadata(status="running")
    _current = ctx
    return ctx


def get_run() -> RunContext | None:
    return _current


def require_run() -> RunContext:
    if _current is None:
        raise RuntimeError("No active run. Call start_run() first.")
    return _current


def load_run(run_id: str) -> RunContext:
    """Resume an existing run (keeps prior usage if metadata exists)."""
    global _current
    root = RUNS_DIR / run_id
    meta_path = root / "run_metadata.json"
    usage = UsageCounters()
    started_at = datetime.now(timezone.utc).isoformat()
    pipeline = ""
    extra: dict[str, Any] = {}
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        started_at = data.get("started_at", started_at)
        pipeline = data.get("pipeline", "")
        u = data.get("token_usage") or {}
        usage = UsageCounters(
            llm_calls=int(u.get("llm_calls", 0)),
            llm_prompt_tokens=int(u.get("llm_prompt_tokens", 0)),
            llm_completion_tokens=int(u.get("llm_completion_tokens", 0)),
            llm_total_tokens=int(u.get("llm_total_tokens", 0)),
            embedding_calls=int(u.get("embedding_calls", 0)),
            embedding_tokens=int(u.get("embedding_tokens", 0)),
            embedding_texts=int(u.get("embedding_texts", 0)),
        )
        extra = {k: v for k, v in data.items() if k not in {
            "run_id", "status", "pipeline", "started_at", "updated_at",
            "model_config", "token_usage", "paths", "notes",
        }}
    ctx = RunContext(
        run_id=run_id,
        started_at=started_at,
        pipeline=pipeline,
        usage=usage,
        extra=extra,
    )
    ensure_dir(ctx.artifacts_dir)
    ensure_dir(ctx.knowledge_bases_dir)
    _current = ctx
    return ctx


def finish_run(status: str = "completed", **extra: Any) -> Path:
    ctx = require_run()
    return ctx.write_metadata(status=status, **extra)
