# topic-aware-rag-lab

Standalone experiment lab for topic-aware RAG. First study compares **contextual RAG** (Pipeline A) against **BERTopic topic-modeling RAG** (Pipeline B) on company PDFs and text, scored with RAGAS.

| | |
|---|---|
| **LLM / embeddings** | OpenAI only (`gpt-4o-mini`, `text-embedding-3-small`) |
| **Vector store** | Chroma (local, disk-persisted) |
| **Topic discovery** | BERTopic with pre-computed OpenAI embeddings (`embedding_model=None`) |
| **Eval** | RAGAS + custom topic coverage metric |

Full blueprint: [`plan/Experiement/rag-experiment-plan.md`](plan/Experiement/rag-experiment-plan.md)  
Concept notes: [`plan/Experiement/rag-concept-research-topic-modeling-rag.md`](plan/Experiement/rag-concept-research-topic-modeling-rag.md)

---

## What we compare

**Pipeline A — Contextual RAG**  
Load → chunk → LLM context prefix per chunk → embed → Chroma (`contextual_rag`).

**Pipeline B — Topic Modeling RAG**  
Load → big chunks → embed → BERTopic topics → tag chunks → iterative LLM topic docs → re-chunk / contextualize → embed → Chroma (`topic_rag`). Outlier chunks (`-1`) become `uncategorized` and are excluded from topic docs (tracked via coverage %).

Both pipelines are queried with the same human-verified `qa_pairs.json` and scored with faithfulness, answer relevancy, context precision, and context recall.

---

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

Install **CPU Torch first**, then the rest. BERTopic pulls `sentence-transformers` → `torch`; without the CPU index, uv downloads large NVIDIA/CUDA wheels. This project uses OpenAI embeddings only, so GPU Torch is unnecessary.

```bash
uv venv
source .venv/bin/activate

# 1. CPU Torch (do this first)
uv add torch --index https://download.pytorch.org/whl/cpu

# 2. Project deps
uv add -r requirements.txt
```

If Torch was already resolved with CUDA packages, recreate the env:

```bash
rm -rf .venv
uv venv
source .venv/bin/activate
uv add torch --index https://download.pytorch.org/whl/cpu
uv add -r requirements.txt
```

Copy env template and set your key:

```bash
cp .env.example .env   # create when available
# OPENAI_API_KEY=sk-...
```

---

## Planned layout

```
topic-aware-rag-lab/
├── pyproject.toml / requirements.txt
├── .env                          # gitignored
├── data/raw/                     # drop PDFs and .txt files here
├── qa_pairs.json                 # human-verified ground truth
├── src/
│   ├── config.py
│   ├── loader.py / chunker.py / embedder.py / chroma_store.py / contextualizer.py
│   ├── pipeline_a.py
│   └── pipeline_b/               # discovery, tagging, topic doc builder
├── scripts/
│   ├── run_pipeline_a.py
│   ├── run_pipeline_b.py
│   ├── generate_testset.py
│   └── run_evaluation.py
├── artifacts/                    # gitignored, written at runtime
└── plan/Experiement/             # design docs
```

---

## Run order (once code exists)

```bash
# 1. Put PDFs / .txt into data/raw/

# 2. Contextual RAG
uv run scripts/run_pipeline_a.py

# 3. Topic modeling RAG — then eyeball topics / tags / topic docs
uv run scripts/run_pipeline_b.py

# 4. Candidate Q&A → human review → qa_pairs.json
uv run scripts/generate_testset.py

# 5. RAGAS comparison → artifacts/results.json
uv run scripts/run_evaluation.py
```

---

## Status

Scaffolding + experiment plan. Implementation of `src/` and `scripts/` is next.
