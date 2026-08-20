# RAG Experiment Plan: Contextual RAG vs Topic Modeling RAG

> **Status:** Plan / blueprint — use this to build a standalone experiment repo.  
> **Goal:** Compare a standard contextual RAG pipeline against a BERTopic-based topic modeling RAG pipeline over real company data. Measure retrieval quality with RAGAS.  
> **Data:** Company PDFs + text files.  
> **LLM/Embeddings:** OpenAI only.  
> **Vector store:** Chroma (local, disk-persisted).

---

## Table of Contents

1. [New repo setup](#1-new-repo-setup)
2. [Folder structure](#2-folder-structure)
3. [Dependencies](#3-dependencies)
4. [Configuration](#4-configuration)
5. [Shared utilities](#5-shared-utilities)
6. [pipeline_contextual_rag](#6-pipeline-a--contextual-rag)
7. [pipeline_topic_modeling_rag](#7-pipeline-b--topic-modeling-rag)
   - 7a. Load + big chunks
   - 7b. Embed big chunks
   - 7c. BERTopic topic discovery
   - 7d. Tag source chunks
   - 7e. Iterative LLM topic document building
   - 7f. Re-chunk + contextualize + embed topic docs
8. [Generating the ground truth test set](#8-generating-the-ground-truth-test-set)
9. [Evaluation with RAGAS](#9-evaluation-with-ragas)
10. [Coverage metric (custom)](#10-coverage-metric-custom)
11. [Artifact reference](#11-artifact-reference)
12. [Run order](#12-run-order)
13. [What each artifact tells you](#13-what-each-artifact-tells-you)
14. [Known trade-offs and accepted decisions](#14-known-trade-offs-and-accepted-decisions)

---

## 1. New repo setup

Create a new standalone Python project (not inside this repo).

```bash
mkdir rag-experiment
cd rag-experiment
git init
uv init          # or: python -m venv .venv && source .venv/bin/activate
```

This repo is fully self-contained. It does **not** depend on this main app's database or config.

---

## 2. Folder structure

```
rag-experiment/
├── pyproject.toml            ← or requirements.txt
├── .env                      ← gitignored
├── .env.example
├── README.md
│
├── data/
│   └── raw/                  ← drop your PDFs and .txt files here
│
├── qa_pairs.json             ← ground truth Q&A (human-verified, see §8)
│
├── src/
│   ├── config.py             ← settings (API key, model names, chunk params)
│   ├── loader.py             ← PDF + text loading → list of (text, source, page)
│   ├── chunker.py            ← RecursiveCharacterTextSplitter wrapper
│   ├── embedder.py           ← OpenAI embeddings wrapper (batched)
│   ├── chroma_store.py       ← Chroma add / similarity_search wrapper
│   ├── contextualizer.py     ← LLM context prefix per chunk
│   │
│   ├── pipeline_contextual_rag.py         ← pipeline_contextual_rag: contextual RAG end-to-end
│   │
│   └── pipeline_topic_modeling_rag/
│       ├── __init__.py
│       ├── pipeline_topic_modeling_rag.py     ← pipeline_topic_modeling_rag orchestrator
│       ├── topic_discovery.py  ← BERTopic: embed chunks → topics
│       ├── topic_tagger.py     ← assign topic_slug to each chunk
│       └── topic_doc_builder.py ← iterative LLM topic document creation
│
├── scripts/
│   ├── run_pipeline_contextual_rag.py     ← entry: load data → run contextual → dump artifacts
│   ├── run_pipeline_topic_modeling_rag.py     ← entry: load data → run topic-modeling → dump artifacts
│   ├── generate_testset.py   ← RAGAS TestsetGenerator → candidate qa_pairs
│   └── run_evaluation.py     ← query both pipelines → RAGAS → results
│
└── artifacts/                ← gitignored, auto-created at runtime
    ├── contextual/
    │   └── chunks.jsonl
    └── topic_modeling/
        ├── big_chunks.jsonl
        ├── discovered_topics.json
        ├── topic_tags.jsonl
        ├── bertopic_model/        ← saved BERTopic model
        ├── topic_docs/
        │   ├── _index.json
        │   └── *.md               ← one per discovered topic
        └── final_chunks.jsonl
```

---

## 3. Dependencies

```toml
# pyproject.toml [project.dependencies]
openai                    # LLM + embeddings
chromadb                  # local vector store
bertopic                  # topic discovery
umap-learn                # dimensionality reduction (used by BERTopic)
hdbscan                   # density clustering (used by BERTopic)
langchain                 # RecursiveCharacterTextSplitter, document loaders
langchain-openai          # OpenAI LLM + embeddings via LangChain
langchain-community       # PyMuPDFLoader
pymupdf                   # PDF text extraction
ragas                     # evaluation metrics
datasets                  # HuggingFace datasets (required by RAGAS)
python-dotenv
pydantic-settings
rich                      # optional: pretty progress output
```

**Note on BERTopic + OpenAI embeddings:** BERTopic can accept pre-computed embeddings directly, so you do not need sentence-transformers. Pass `embedding_model=None` and supply your own numpy array.

---

## 4. Configuration

`src/config.py` — single settings object used everywhere.

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-5.6-terra"        # used for context prefix + topic doc building
    EMBEDDING_MODEL: str = "text-embedding-3-large"  # 3072 dims

    # Chunking
    BIG_CHUNK_SIZE: int = 800               # tokens, for raw source chunks
    BIG_CHUNK_OVERLAP: int = 100
    SMALL_CHUNK_SIZE: int = 350             # tokens, for topic doc re-chunking
    SMALL_CHUNK_OVERLAP: int = 50

    # BERTopic
    UMAP_N_COMPONENTS: int = 5
    UMAP_MIN_DIST: float = 0.0
    HDBSCAN_MIN_CLUSTER_SIZE: int = 3       # tune down if your corpus is small

    # Chroma
    CHROMA_PATH: str = "./chroma_db"
    COLLECTION_CONTEXTUAL: str = "contextual_rag"
    COLLECTION_TOPIC: str = "topic_rag"

    # Retrieval
    TOP_K: int = 5

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 5. Shared utilities

### `src/loader.py`

Returns a flat list of records: `{"text": str, "source": str, "page": int}`.

- PDFs: use `langchain_community.document_loaders.PyMuPDFLoader`. Each loaded `Document` is one page; `source` = filename, `page` = page number.
- Text files: plain `open().read()`. Treat entire file as page 0.
- Name convention for chunk IDs: `{source_basename}_p{page}` → e.g. `brochure_p3`.

### `src/chunker.py`

Thin wrapper around `langchain.text_splitter.RecursiveCharacterTextSplitter`.  
Takes a list of `(text, source, page)` records and returns a list of chunk dicts:

```json
{
  "chunk_id": "brochure_p3_c001",
  "content": "...",
  "source": "brochure.pdf",
  "page": 3,
  "char_start": 0,
  "char_end": 900
}
```

### `src/embedder.py`

Wraps `openai.OpenAI().embeddings.create(model=..., input=[...])`.  
Always batch embed (max 100 texts per call) to stay within rate limits.  
Returns a list of `list[float]` in the same order as input.

### `src/chroma_store.py`

Wraps Chroma's `PersistentClient`. Two methods:

- `add(collection_name, chunks, embeddings, metadatas)` — adds with pre-computed embeddings.
- `search(collection_name, query_embedding, top_k)` → returns `[{content, metadata, distance}]`.

Use `chromadb.PersistentClient(path=settings.CHROMA_PATH)`.

### `src/contextualizer.py`

For each chunk, makes one LLM call with this prompt template:

```
System: You are a document analyst. Given the full chunk text and its source filename,
write a single sentence (max 30 words) that situates this chunk within the document.
State the document name, topic covered, and any named entity if present.

User:
Source: {source}, page {page}
Chunk text: {content}

Response (one sentence only):
```

Returns the context string. Call this per-chunk. To save cost, you can skip chunks where content already starts with a clear heading.

---

## 6. pipeline_contextual_rag

`src/pipeline_contextual_rag.py`

```
Load raw files
  → big chunks  (BIG_CHUNK_SIZE / BIG_CHUNK_OVERLAP)
  → for each chunk: LLM context prefix
  → embed (context + "\n\n" + content)
  → store in Chroma collection: contextual_rag
  → dump: artifacts/contextual/chunks.jsonl
```

Each entry in `chunks.jsonl`:

```json
{
  "chunk_id": "...",
  "content": "...",
  "context": "This chunk is from ...",
  "embed_text": "context\n\ncontent",
  "source": "brochure.pdf",
  "page": 3
}
```

At query time (`scripts/run_evaluation.py`), embed the query → `chroma_store.search(collection_contextual, query_embedding, TOP_K)` → pass returned chunks as context to LLM.

---

## 7. pipeline_topic_modeling_rag

`src/pipeline_topic_modeling_rag/pipeline_topic_modeling_rag.py` orchestrates the steps below.

---

### 7a. Load + big chunks

Identical to pipeline_contextual_rag steps 1–2. Use the **same chunk size settings** so the comparison is fair.

Dump to `artifacts/topic_modeling/big_chunks.jsonl`.

---

### 7b. Embed big chunks

Embed raw `content` only (no context prefix at this stage).  
Store embeddings in memory as a numpy array aligned with the chunk list.  
These embeddings are used only for BERTopic clustering — not for retrieval.

---

### 7c. BERTopic topic discovery

`src/pipeline_topic_modeling_rag/topic_discovery.py`

```python
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

umap_model = UMAP(
    n_components=settings.UMAP_N_COMPONENTS,
    min_dist=settings.UMAP_MIN_DIST,
    metric="cosine",
    random_state=42,
)
hdbscan_model = HDBSCAN(
    min_cluster_size=settings.HDBSCAN_MIN_CLUSTER_SIZE,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True,
    verbose=True,
    embedding_model=None,   # we supply pre-computed embeddings
)
topics, probs = topic_model.fit_transform(chunk_texts, embeddings=embeddings_array)
```

**LLM labeling step:**  
BERTopic gives keyword lists per topic (c-TF-IDF). For each topic, call OpenAI:

```
System: You are naming topics for a company knowledge base.
User: Here are the top keywords for a cluster of document chunks:
{keywords}

Give a short, lowercase, underscore-separated slug (e.g. "company_projects", "team_bios").
Then give a human-readable label (3-5 words).
Reply in JSON: {"slug": "...", "label": "..."}
```

**Outliers:** BERTopic assigns `-1` to chunks that don't fit any cluster. These are labeled `topic_slug = "uncategorized"` and excluded from pipeline_topic_modeling_rag's topic docs (accepted data loss).

Save `artifacts/topic_modeling/discovered_topics.json`:

```json
[
  {
    "topic_id": 0,
    "slug": "company_projects",
    "label": "Company Projects",
    "keywords": ["project", "client", "delivery", "2024"],
    "representative_chunk_ids": ["brochure_p3_c001", "..."],
    "chunk_count": 18
  },
  ...
  {
    "topic_id": -1,
    "slug": "uncategorized",
    "label": "Uncategorized (outliers)",
    "chunk_count": 7
  }
]
```

**EYEBALL CHECK POINT:** Open `discovered_topics.json` and verify:
- Are the slugs sensible?
- Are there too many micro-topics or too few broad ones?
- Tune `HDBSCAN_MIN_CLUSTER_SIZE` up (fewer, bigger topics) or down (more granular) and re-run.

Also save the BERTopic model: `topic_model.save("artifacts/topic_modeling/bertopic_model")` — lets you reload without re-running.

---

### 7d. Tag source chunks

`src/pipeline_topic_modeling_rag/topic_tagger.py`

Map each chunk index to its assigned `topic_slug` from the BERTopic output.

Save `artifacts/topic_modeling/topic_tags.jsonl` — one line per chunk:

```json
{"chunk_id": "brochure_p3_c001", "topic_slug": "company_projects", "topic_id": 0, "probability": 0.87}
{"chunk_id": "notes_p0_c003",    "topic_slug": "uncategorized",     "topic_id": -1, "probability": 0.0}
```

**EYEBALL CHECK POINT:** Sample 10-15 entries. Do the assigned topics match your intuition for that chunk's content?

---

### 7e. Iterative LLM topic document building

`src/pipeline_topic_modeling_rag/topic_doc_builder.py`

For each non-`uncategorized` topic:

1. Collect all chunks with that `topic_slug`, sorted by `source` → `page` → `chunk_id` order.
2. Split into batches of 5 chunks (tune this based on LLM context window usage).
3. **Batch 1 — initial draft:**

```
System: You are building a structured knowledge document for a company knowledge base.
Write a well-structured Markdown document about the topic: "{topic_label}".
Use only the information from the provided source chunks. Do not invent facts.
Include a YAML frontmatter block with: slug, sources (list of filenames), chunk_ids_used.

User:
Topic: {topic_label}
Source chunks:
---
[CHUNK brochure_p3_c001] (source: brochure.pdf, page 3)
{content}
---
[CHUNK ...] ...
---
```

4. **Subsequent batches — update:**

```
System: You are updating a company knowledge document.
Add any new, non-duplicate information from the new source chunks below.
Do not repeat what is already in the document. Do not invent facts.
Update the YAML frontmatter chunk_ids_used list.

User:
Current document:
{current_doc}

New source chunks:
---
[CHUNK ...] ...
---
```

5. After all batches: write final `current_doc` to `artifacts/topic_modeling/topic_docs/{slug}.md`.

Update `artifacts/topic_modeling/topic_docs/_index.json`:

```json
{
  "company_projects": {
    "slug": "company_projects",
    "label": "Company Projects",
    "path": "topic_docs/company_projects.md",
    "source_chunk_count": 18,
    "sources": ["brochure.pdf", "projects.txt"]
  }
}
```

**EYEBALL CHECK POINT:** Open 2-3 topic docs. Do they read like coherent summaries? Are there obvious invented facts? Cross-check a claim against `topic_tags.jsonl` → back to `big_chunks.jsonl` to find the source chunk.

---

### 7f. Re-chunk + contextualize + embed topic docs

For each topic doc `*.md`:

1. **Re-chunk** at `SMALL_CHUNK_SIZE / SMALL_CHUNK_OVERLAP`. These chunks are smaller since the content is already clean and deduplicated.
2. **Contextualize** each small chunk with the same LLM context prefix step from `src/contextualizer.py`.
3. **Embed** `context + "\n\n" + content`.
4. **Store in Chroma** collection `topic_rag`. Metadata: `topic_slug`, `source_doc_slug`, `chunk_id`.
5. Dump to `artifacts/topic_modeling/final_chunks.jsonl`.

---

## 8. Generating the ground truth test set

### Approach: RAGAS TestsetGenerator + human review

RAGAS provides `TestsetGenerator` which reads your source documents and produces candidate Q&A pairs automatically.

```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

# Load all raw docs as LangChain Documents
docs = [...]   # loaded from data/raw/

generator = TestsetGenerator.with_openai()
testset = generator.generate_with_langchain_docs(
    docs,
    test_size=30,
    distributions={
        simple: 0.4,          # single-document factual
        reasoning: 0.3,       # requires inference
        multi_context: 0.3,   # spans multiple docs
    }
)
testset.to_pandas().to_json("candidate_qa_pairs.json", orient="records", indent=2)
```

### Human review tips

After generation you will have `candidate_qa_pairs.json` with fields: `question`, `ground_truth`, `contexts`, `evolution_type`.

**What to do for each entry:**

1. Read the question and the generated `ground_truth` answer.
2. Ask yourself: is the answer actually correct given your company data?
3. **Keep** the entry if the answer is right (or fix the answer text).
4. **Delete** the entry if the question is nonsensical, unanswerable, or the answer is wrong and you cannot easily fix it.
5. Aim for **20-30 final entries** covering all discovered topics.

**Tips for a useful test set:**

- Make sure you have at least 2-3 questions per major topic (not all questions about one topic).
- Include at least 5 multi-context questions — these are where topic modeling RAG is most likely to show a difference.
- Avoid yes/no questions — open-ended answers give RAGAS more signal.
- If the auto-generated questions are too generic ("What is the company?"), manually write a few specific ones about facts only your data knows.
- The `ground_truth` does not need to be a verbatim quote — a paraphrase of the correct answer is fine.

Save your final human-verified set as `qa_pairs.json`:

```json
[
  {
    "question": "What projects did the company deliver in 2024?",
    "ground_truth": "The company delivered Project Alpha for Client X and a redesign for Client Y in 2024.",
    "evolution_type": "multi_context"
  },
  ...
]
```

---

## 9. Evaluation with RAGAS

`scripts/run_evaluation.py`

### Step 1 — Query both pipelines

For each question in `qa_pairs.json`:

```
query_embedding = embedder.embed([question])[0]

# pipeline_contextual_rag
ctx_a = chroma_store.search(COLLECTION_CONTEXTUAL, query_embedding, TOP_K)
answer_a = llm.ask(question, contexts=ctx_a)

# pipeline_topic_modeling_rag
ctx_b = chroma_store.search(COLLECTION_TOPIC, query_embedding, TOP_K)
answer_b = llm.ask(question, contexts=ctx_b)
```

LLM answer generation prompt:

```
System: Answer the question using only the provided context. Be concise and factual.
User:
Context:
{context_1}
---
{context_2}
...

Question: {question}
```

### Step 2 — Build RAGAS datasets

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

dataset_a = Dataset.from_dict({
    "question":     [q["question"] for q in qa_pairs],
    "answer":       [answers_a[i] for i in range(len(qa_pairs))],
    "contexts":     [contexts_a[i] for i in range(len(qa_pairs))],  # list of strings
    "ground_truth": [q["ground_truth"] for q in qa_pairs],
})

result_a = evaluate(dataset_a, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
```

Repeat for pipeline_topic_modeling_rag.

### Step 3 — Compare and save

```python
import json
results = {
    "pipeline_contextual_rag": result_a.to_pandas().to_dict(),
    "pipeline_topic_modeling_rag": result_b.to_pandas().to_dict(),
    "summary": {
        "pipeline_contextual_rag": {
            "faithfulness": result_a["faithfulness"],
            "answer_relevancy": result_a["answer_relevancy"],
            "context_precision": result_a["context_precision"],
            "context_recall": result_a["context_recall"],
        },
        "pipeline_topic_modeling_rag": { ... }
    }
}
with open("artifacts/results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### RAGAS metrics explained

| Metric | What it measures | What a low score means |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | LLM is making things up beyond what was retrieved |
| **Answer Relevancy** | Does the answer actually address the question? | Retrieval found off-topic chunks |
| **Context Precision** | Of the retrieved chunks, how many were relevant? | Too much noise in top-k |
| **Context Recall** | Did retrieval capture all info needed to answer? | Important chunks were missed |

---

## 10. Coverage metric (custom)

At the end of pipeline_topic_modeling_rag, calculate:

```python
total_chunks    = len(big_chunks)
uncategorized   = sum(1 for t in topic_tags if t["topic_slug"] == "uncategorized")
covered         = total_chunks - uncategorized
coverage_pct    = covered / total_chunks * 100

print(f"Coverage: {coverage_pct:.1f}% ({covered}/{total_chunks} chunks in topic docs)")
print(f"Accepted data loss: {uncategorized} chunks ({100 - coverage_pct:.1f}%)")
```

Add this to `artifacts/results.json` under `"pipeline_topic_modeling_rag_coverage"`.

---

## 11. Artifact reference

| File | When written | What to check |
|---|---|---|
| `artifacts/contextual/chunks.jsonl` | After pipeline_contextual_rag | Chunk sizes, context prefix quality |
| `artifacts/topic_modeling/big_chunks.jsonl` | Step 7a | Same chunking as pipeline_contextual_rag |
| `artifacts/topic_modeling/discovered_topics.json` | Step 7c | Topic labels, slug quality, outlier count |
| `artifacts/topic_modeling/topic_tags.jsonl` | Step 7d | Are chunks tagged to the right topic? |
| `artifacts/topic_modeling/topic_docs/*.md` | Step 7e | Coherence, hallucination spot-check |
| `artifacts/topic_modeling/topic_docs/_index.json` | Step 7e | Source coverage per topic |
| `artifacts/topic_modeling/final_chunks.jsonl` | Step 7f | Final embed text quality |
| `artifacts/results.json` | Evaluation | RAGAS scores + coverage |

---

## 12. Run order

```
1. Drop your files into data/raw/

2. uv run scripts/run_pipeline_contextual_rag.py
   → artifacts/contextual/chunks.jsonl
   → Chroma: contextual_rag populated

3. uv run scripts/run_pipeline_topic_modeling_rag.py
   → artifacts/topic_modeling/ fully populated
   → Chroma: topic_rag populated
   → EYEBALL: discovered_topics.json, topic_tags.jsonl, topic_docs/*.md

4. uv run scripts/generate_testset.py
   → candidate_qa_pairs.json
   → HUMAN REVIEW: verify + trim to qa_pairs.json

5. uv run scripts/run_evaluation.py
   → artifacts/results.json
```

---

## 13. What each artifact tells you

**Are big chunks good or not?**  
Open `big_chunks.jsonl`. Sample 5-10 chunks. Check: are they coherent paragraphs, or do they cut mid-sentence? Are chunk sizes roughly uniform? If chunks are cutting badly, decrease `BIG_CHUNK_SIZE` or switch to a heading-aware splitter.

**Are generated topic tags good or not?**  
Open `topic_tags.jsonl`. For each unique `topic_slug`, read a few of its assigned chunks in `big_chunks.jsonl`. Do they belong together? If not: try increasing `HDBSCAN_MIN_CLUSTER_SIZE` (merges small noisy clusters) or increasing `UMAP_N_COMPONENTS`.

**Did the LLM hallucinate in topic docs?**  
Spot-check: pick a claim in a topic doc → find which chunks fed that topic via `_index.json` → search `big_chunks.jsonl` for those chunk IDs → verify the claim appears in the source chunk text. If it doesn't appear anywhere, that's a hallucination.

**What is the output quality?**  
`artifacts/results.json` → summary section → compare RAGAS scores across pipelines. A higher `context_recall` in pipeline_topic_modeling_rag means it retrieved more of what was needed. A higher `faithfulness` means less hallucination in final answers.

---

## 14. Known trade-offs and accepted decisions

| Decision | Trade-off accepted |
|---|---|
| Outlier chunks excluded from pipeline_topic_modeling_rag | Some source data is not represented in retrieval. Coverage metric quantifies this. |
| Iterative LLM doc building | 1 LLM call per batch of 5 chunks per topic. Can be expensive on large corpora. Tune batch size. |
| Single topic per chunk (hard assignment) | BERTopic can do soft multi-label but hard assignment keeps tagging simple for the POC. |
| No BM25 / hybrid search | Both pipelines use dense-only retrieval. Adding BM25 would help keyword-heavy queries but adds complexity. Defer. |
| RAGAS uses LLM-as-judge | RAGAS metrics themselves call OpenAI. This adds cost and a circular dependency (same model evaluating same model). Acceptable for POC. |
| No re-run / dedup logic | If you re-run, Chroma collections accumulate duplicates. Either delete and recreate collections, or add a content hash check before adding. |

---

## Document history

| Date | Note |
|---|---|
| 2026-08-20 | Initial experiment plan from design discussion |
