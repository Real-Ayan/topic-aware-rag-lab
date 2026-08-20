# topic-aware-rag-lab

Standalone experiment lab comparing **pipeline_contextual_rag** vs **pipeline_topic_modeling_rag**.

Each execution creates a **run** under `runs/{run_id}/` with its own artifacts, knowledge bases, and `run_metadata.json` (models + estimated token usage).

---

## Put your files here

```
data/raw/   # recursive: all nested .pdf / .docx / .txt
```

---

## Setup

```bash
uv venv && source .venv/bin/activate
uv add torch --index https://download.pytorch.org/whl/cpu
uv add -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY
```

---

## Run order

```bash
uv run scripts/run_pipeline_contextual_rag.py
uv run scripts/run_pipeline_topic_modeling_rag.py

# Re-cluster without re-embedding (after changing cluster settings in .env)
uv run scripts/run_topic_clustering.py --run-id <RUN_ID> --reuse-embeddings

uv run scripts/generate_testset.py
uv run scripts/run_evaluation.py --run-id <RUN_ID>
```

---

## Run IDs, metadata & tokens

Run ids are **local timestamps**: `YYYYMMDD_HHMMSS` (e.g. `20260820_171521`). Newest sorts last alphabetically / first in the list.

```bash
uv run scripts/list_runs.py   # marks ← latest
cat runs/latest.json          # pointer to last run
cat runs/index.json           # all runs
```

Each run also has `runs/{run_id}/run_metadata.json` with `model_config` + API `token_usage`.

---

## Knowledge bases

| Base name | On disk |
|---|---|
| `kb_contextual_rag` | `runs/{run_id}/knowledge_bases/kb_contextual_rag__{run_id}/` |
| `kb_topic_modeling_rag` | `runs/{run_id}/knowledge_bases/kb_topic_modeling_rag__{run_id}/` |

```bash
uv run scripts/query_knowledge_base.py list
uv run scripts/query_knowledge_base.py compare --run-id <RUN_ID> --question "..."
```

---

## Clustering — general ideas (not corpus-specific)

Defaults follow common BERTopic practice (`hdbscan` + `eom`). Treat `.env` as knobs you change, then re-cluster with cached embeddings.

| Goal | What to try |
|---|---|
| Fewer / broader topics | Raise `HDBSCAN_MIN_CLUSTER_SIZE`; keep `HDBSCAN_SELECTION_METHOD=eom` |
| More / finer topics | Lower `HDBSCAN_MIN_CLUSTER_SIZE`; try `HDBSCAN_SELECTION_METHOD=leaf` |
| Fixed topic count | `CLUSTER_METHOD=kmeans` and set `N_TOPICS` to your target |
| Stable geometry | Adjust `UMAP_N_NEIGHBORS` / `UMAP_N_COMPONENTS` (larger neighbors → smoother) |
| Better topic names | Raise `TOPIC_LABEL_SAMPLES` (LLM sees more excerpts; no c-TF-IDF labels) |
| Cheap iteration | Embed once → change knobs → `run_topic_clustering.py --reuse-embeddings` |

Topic labels always come from **random chunk samples → LLM**, not keyword lists.
