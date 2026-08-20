# topic-aware-rag-lab

Experiment lab comparing **contextual RAG** vs **topic-tagged RAG** on company docs (PDF / DOCX / TXT).

Each pipeline run creates a timestamped folder under `runs/` with its own artifacts, Chroma knowledge base, and `run_metadata.json` (models + token usage).

Findings write-up: [`plan/Experiement/rag-experiment-findings.md`](plan/Experiement/rag-experiment-findings.md)

---

## 1. Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv && source .venv/bin/activate

# CPU Torch first (avoids huge CUDA wheels; embeddings are OpenAI-only)
uv add torch --index https://download.pytorch.org/whl/cpu
uv add -r requirements.txt

cp .env.example .env
# edit .env → set OPENAI_API_KEY=sk-...
```

Useful `.env` knobs: `LLM_MODEL`, `EMBEDDING_MODEL`, `TOP_K`, `MMR_*`, clustering settings (see §7).

---

## 2. Put your documents here

```text
data/raw/
```

- Loads **recursively** (nested folders OK)
- Supported: `.pdf`, `.docx`, `.txt`, `.md`
- Skips `README.md` / `.gitkeep`

---

## 3. Build the two knowledge bases

Run each pipeline once (each creates its **own** `run_id`):

```bash
# A) Contextual RAG — chunk + LLM context prefix + embed
uv run scripts/run_pipeline_contextual_rag.py

# B) Topic RAG — cluster/tag + index ORIGINAL chunks (no rewrite KB)
uv run scripts/run_pipeline_topic_modeling_rag.py
```

Note the printed `run_id` (also in `runs/latest.json`). Example:

| Pipeline | Example run id |
|---|---|
| Contextual | `20260820_172037` |
| Topic | `20260820_172419` |

List runs anytime:

```bash
uv run scripts/list_runs.py
```

---

## 4. Ask a question (main tool)

Compare **both** KBs with one question. Pass the run id for each:

```bash
uv run scripts/ask.py \
  --contextual-run 20260820_172037 \
  --topic-run 20260820_172419 \
  "What is RPI Sentinel?"
```

Interactive (prompts for the question):

```bash
uv run scripts/ask.py \
  --contextual-run 20260820_172037 \
  --topic-run 20260820_172419
```

### Inspect retrieved chunks (spot bad context)

**Retrieve only** — no LLM answer cost:

```bash
uv run scripts/ask.py \
  --contextual-run 20260820_172037 \
  --topic-run 20260820_172419 \
  --inspect-only \
  --top-k 8 \
  "What is RPI Sentinel?"
```

Shows per hit: `distance`, `topic_slug`, `source`, `chunk_id`, text preview.

**Answer + chunk dump:**

```bash
uv run scripts/ask.py \
  --contextual-run 20260820_172037 \
  --topic-run 20260820_172419 \
  --show-contexts \
  "What is PetHero?"
```

**JSON output** (for scripts):

```bash
uv run scripts/ask.py \
  --contextual-run 20260820_172037 \
  --topic-run 20260820_172419 \
  --json \
  "What cloud services does RPI offer?"
```

What to check in hits:
- Is `source` the right product file?
- Are #2–#5 near-duplicates / sibling products?
- Is `distance` much worse after hit #1?

---

## 5. What each pipeline stores

| Pipeline | Knowledge base path | Contents |
|---|---|---|
| Contextual | `runs/{id}/knowledge_bases/kb_contextual_rag__{id}/` | Chunks + LLM context prefixes |
| Topic | `runs/{id}/knowledge_bases/kb_topic_modeling_rag__{id}/` | **Original** chunks tagged with `topic_slug` |

Topic approach (A+D):
- **D** — topics tag originals (not free-form rewritten mega-docs)
- **A** — retrieval uses candidate fetch + soft topic route + **MMR** diversity

Optional human-readable topic overviews (not used for retrieval):

```bash
uv run scripts/run_pipeline_topic_modeling_rag.py --build-topic-docs
```

---

## 6. Rebuild / re-cluster topic KB

Cheapest: re-index from existing tags + embeddings (no re-embed, no re-cluster):

```bash
uv run scripts/run_topic_clustering.py \
  --run-id 20260820_172419 \
  --rebuild-kb-only
```

Re-cluster with cached embeddings (after changing `.env` cluster knobs):

```bash
uv run scripts/run_topic_clustering.py \
  --run-id 20260820_172419 \
  --reuse-embeddings \
  --rebuild-kb
```

---

## 7. Clustering knobs (general)

Edit `.env`, then re-cluster with `--reuse-embeddings`.

| Goal | Try |
|---|---|
| Fewer / broader topics | Raise `HDBSCAN_MIN_CLUSTER_SIZE`; `HDBSCAN_SELECTION_METHOD=eom` |
| More / finer topics | Lower `HDBSCAN_MIN_CLUSTER_SIZE`; try `leaf` |
| Fixed topic count | `CLUSTER_METHOD=kmeans` + `N_TOPICS=...` |
| Better topic names | Raise `TOPIC_LABEL_SAMPLES` |

---

## 8. Scripts reference

| Script | Purpose |
|---|---|
| `scripts/run_pipeline_contextual_rag.py` | Build contextual KB |
| `scripts/run_pipeline_topic_modeling_rag.py` | Build topic-tagged KB |
| `scripts/run_topic_clustering.py` | Re-cluster / rebuild topic KB |
| `scripts/ask.py` | Ask both KBs / inspect contexts |
| `scripts/list_runs.py` | List runs (marks ← latest) |
| `scripts/query_knowledge_base.py` | List/query KBs |
| `scripts/generate_testset.py` | Candidate Q&A (then human-review → `qa_pairs.json`) |
| `scripts/run_evaluation.py --run-id ...` | RAGAS eval (needs `qa_pairs.json`) |

---

## 9. Layout

```text
topic-aware-rag-lab/
├── data/raw/                 ← drop docs here
├── runs/{run_id}/            ← artifacts + knowledge_bases + run_metadata.json
├── scripts/                  ← CLI entrypoints
├── src/                      ← pipelines + retrieve/ask
├── plan/Experiement/         ← design + findings
├── .env                      ← secrets (gitignored)
└── README.md                 ← this guide
```
