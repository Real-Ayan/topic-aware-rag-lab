"""Generate candidate Q&A pairs with RAGAS TestsetGenerator (then human-review)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rich.console import Console

from src.config import settings
from src.io_utils import write_json
from src.loader import load_raw_documents

console = Console()


def main() -> None:
    console.rule("[bold]Generate candidate test set[/bold]")
    records = load_raw_documents()
    docs = [
        Document(page_content=r["text"], metadata={"source": r["source"], "page": r["page"]})
        for r in records
    ]

    # RAGAS API has evolved; try modern path then fall back.
    try:
        from ragas.testset import TestsetGenerator
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        generator_llm = LangchainLLMWrapper(
            ChatOpenAI(model=settings.LLM_MODEL, api_key=settings.OPENAI_API_KEY)
        )
        embedding = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
        )
        generator = TestsetGenerator(llm=generator_llm, embedding_model=embedding)
        testset = generator.generate_with_langchain_docs(docs, testset_size=30)
        df = testset.to_pandas()
    except Exception as modern_err:
        console.print(f"[yellow]Modern RAGAS path failed ({modern_err}); trying legacy...[/yellow]")
        from ragas.testset.generator import TestsetGenerator
        from ragas.testset.evolutions import simple, reasoning, multi_context

        generator = TestsetGenerator.with_openai()
        testset = generator.generate_with_langchain_docs(
            docs,
            test_size=30,
            distributions={simple: 0.4, reasoning: 0.3, multi_context: 0.3},
        )
        df = testset.to_pandas()

    out = ROOT / "candidate_qa_pairs.json"
    # Normalize columns for human review
    records_out = []
    for _, row in df.iterrows():
        records_out.append(
            {
                "question": row.get("user_input") or row.get("question") or "",
                "ground_truth": row.get("reference") or row.get("ground_truth") or "",
                "contexts": row.get("reference_contexts") or row.get("contexts") or [],
                "evolution_type": row.get("evolution_type") or row.get("synthesizer_name") or "",
            }
        )
    write_json(out, records_out)
    console.print(f"[green]Wrote[/green] {out} ({len(records_out)} candidates)")
    console.print(
        "Human review next: keep/fix ~20–30 items → save as [bold]qa_pairs.json[/bold]"
    )


if __name__ == "__main__":
    main()
