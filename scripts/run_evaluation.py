"""Query both pipelines and score with RAGAS."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset
from rich.console import Console
from rich.progress import track

from src.config import settings
from src.embedder import Embedder
from src.io_utils import answer_with_context, ensure_dir, read_json, write_json
from src.knowledge_base import kb_store

console = Console()


def _load_qa_pairs() -> list[dict]:
    path = ROOT / "qa_pairs.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run generate_testset.py, review candidates, "
            "and save the verified set as qa_pairs.json."
        )
    data = read_json(path)
    if not isinstance(data, list) or not data:
        raise ValueError("qa_pairs.json must be a non-empty JSON array")
    return data


def _run_pipeline(name: str, kb_name: str, qa_pairs: list[dict], embedder: Embedder):
    answers: list[str] = []
    contexts: list[list[str]] = []
    store, collection = kb_store(kb_name)
    for q in track(qa_pairs, description=f"Querying {name}"):
        question = q["question"]
        q_emb = embedder.embed([question])[0]
        hits = store.search(collection, q_emb, settings.TOP_K)
        ctx_texts = [h["content"] for h in hits]
        answer = answer_with_context(question, ctx_texts)
        answers.append(answer)
        contexts.append(ctx_texts)
    return answers, contexts


def _evaluate(qa_pairs: list[dict], answers: list[str], contexts: list[list[str]]):
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    dataset = Dataset.from_dict(
        {
            "question": [q["question"] for q in qa_pairs],
            "answer": answers,
            "contexts": contexts,
            "ground_truth": [q["ground_truth"] for q in qa_pairs],
        }
    )
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )


def _summary_from_result(result) -> dict:
    keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    out = {}
    for k in keys:
        try:
            out[k] = float(result[k])
        except Exception:
            out[k] = None
    return out


def main() -> None:
    import argparse

    from src.run_context import finish_run, load_run

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Run id whose KBs to evaluate")
    args = parser.parse_args()

    console.rule("[bold]Evaluation — RAGAS[/bold]")
    run = load_run(args.run_id)
    run.pipeline = (run.pipeline or "") + "+evaluation"
    qa_pairs = _load_qa_pairs()
    embedder = Embedder()

    kb_ctx = run.kb_name(settings.KB_CONTEXTUAL)
    kb_topic = run.kb_name(settings.KB_TOPIC_MODELING)

    answers_ctx, contexts_ctx = _run_pipeline(
        "pipeline_contextual_rag", kb_ctx, qa_pairs, embedder
    )
    answers_topic, contexts_topic = _run_pipeline(
        "pipeline_topic_modeling_rag", kb_topic, qa_pairs, embedder
    )

    console.print("[cyan]Scoring pipeline_contextual_rag with RAGAS...[/cyan]")
    result_ctx = _evaluate(qa_pairs, answers_ctx, contexts_ctx)
    console.print("[cyan]Scoring pipeline_topic_modeling_rag with RAGAS...[/cyan]")
    result_topic = _evaluate(qa_pairs, answers_topic, contexts_topic)

    coverage_path = run.artifacts_dir / "topic_modeling" / "coverage.json"
    coverage = read_json(coverage_path) if coverage_path.exists() else {}

    per_question = []
    for i, q in enumerate(qa_pairs):
        per_question.append(
            {
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "pipeline_contextual_rag": {
                    "answer": answers_ctx[i],
                    "contexts": contexts_ctx[i],
                },
                "pipeline_topic_modeling_rag": {
                    "answer": answers_topic[i],
                    "contexts": contexts_topic[i],
                },
            }
        )

    results = {
        "run_id": run.run_id,
        "summary": {
            "pipeline_contextual_rag": _summary_from_result(result_ctx),
            "pipeline_topic_modeling_rag": _summary_from_result(result_topic),
        },
        "pipeline_topic_modeling_rag_coverage": coverage,
        "per_question": per_question,
    }

    try:
        results["pipeline_contextual_rag_rows"] = json.loads(
            result_ctx.to_pandas().to_json(orient="records")
        )
        results["pipeline_topic_modeling_rag_rows"] = json.loads(
            result_topic.to_pandas().to_json(orient="records")
        )
    except Exception:
        pass

    out = run.artifacts_dir / "results.json"
    ensure_dir(out.parent)
    write_json(out, results)
    finish_run(status="completed", evaluation=results["summary"])
    console.print(f"[green]Wrote[/green] {out}")
    console.print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
