# Experiment Findings: Contextual RAG vs Topic Modeling RAG

**Date:** 2026-08-20  
**Corpus:** RPI company docs under `data/raw/` (recursive; mostly `.docx` + one sitemap `.txt`)  
**Models:** OpenAI (`LLM_MODEL` / `EMBEDDING_MODEL` from `.env`; typically GPT-5.6 family + `text-embedding-3-large`)

---

## 1. What we built

| Pipeline | Idea | Knowledge base |
|---|---|---|
| **Contextual RAG** | Chunk → LLM context prefix → embed → Chroma | `kb_contextual_rag__{run_id}` |
| **Topic Modeling RAG** | Chunk → embed → cluster/tag topics → (originally rewrite) → embed | `kb_topic_modeling_rag__{run_id}` |

Runs are isolated under `runs/{YYYYMMDD_HHMMSS}/` with `run_metadata.json` (model config + token usage).

Compare answers:

```bash
uv run scripts/ask.py \
  --contextual-run <CONTEXTUAL_RUN_ID> \
  --topic-run <TOPIC_RUN_ID> \
  "Your question"
```

---

## 2. Runs used for comparison

| Role | Run id | Notes |
|---|---|---|
| Contextual KB | `20260820_172037` | Built successfully; product facts retained in chunks |
| Topic KB (initial) | `20260820_172419` | Clustering + **LLM topic-doc rewrite** indexed for retrieval |
| Topic KB (after fix) | same `20260820_172419` | Rebuilt with **tagged originals** (`--rebuild-kb-only`) |

---

## 3. Finding: free-form topic rewrite destroys named entities

### Observation

Topic discovery collapsed the corpus into **2 broad topics** (~64 + ~53 chunks). The main product topic doc after iterative LLM rewrite was effectively **empty** (~1 byte). Key product names (**Sentinel, PetHero, HeroHR, RecoAI, eDiagnos, …**) were **absent** from the rewritten retrieval corpus.

### Mini qualitative test (before fix)

| Question | Contextual (`172037`) | Topic rewrite KB (`172419`) |
|---|---|---|
| What is RPI Sentinel? | Specific, correct | “Context does not explain…” |
| What cloud services does RPI provide? | Listed concrete services | “Context does not specify…” |
| What is PetHero? | Specific, correct | “Context does not explain…” |

**Verdict:** Contextual clearly better for product/project name questions under the rewrite-based topic KB.

### Why rewrite failed (general, not corpus-specific)

The rewrite step was meant to **dedupe and clean** overlapping marketing text. In practice, free-form iterative summarization:

- drops rare proper nouns (product / project names)
- merges specifics into vague themes (“AI solutions”)
- can fail or empty-out on very large mega-topics (many batches)
- optimizes for fluent prose, not for **inventory recall**

So: *cleaner text ≠ better retrieval* when **names are the payload**.

---

## 4. Finding: duplicates are real — but rewrite is the wrong tool

Duplicates / near-duplicates in this kind of corpus are usually:

- overlapping product blurbs across files
- shared marketing language
- boilerplate / sitemap repetition

Those hurt top-k redundancy. They do **not** justify discarding original chunks.

---

## 5. Fix adopted: A + D

| Letter | Change |
|---|---|
| **D** | Topics **tag** original chunks (`topic_slug`); **index originals** in Chroma. Optional LLM topic markdown is artifact-only, **not** the retrieval corpus. |
| **A** | Retrieval: fetch extra candidates → soft topic route (majority `topic_slug` among early hits) → **MMR** for diversity. |

Rebuild existing topic run without re-embedding:

```bash
uv run scripts/run_topic_clustering.py --run-id 20260820_172419 --rebuild-kb-only
```

### Retest after A+D

| Question | Contextual | Topic (tagged originals + MMR) |
|---|---|---|
| What is RPI Sentinel? | Correct | Correct (retrieved `RPI Sentinel.docx`) |
| What is PetHero? | Correct | Correct |

Both pipelines now answer specific product questions well. Topic RAG stopped failing because names live in the indexed text again.

---

## 6. Design takeaways

1. **Use topics for routing / organization, not as a replacement corpus** unless the rewrite is structured and name-preserving.
2. **Dedup with MMR (and/or near-dup collapse), not with essay rewriting.**
3. **Contextual RAG remains a strong baseline** for factoid / named-entity questions.
4. **Topic rewrite may still help humans** (overview docs) but should not be the sole vector index when product names matter.
5. **Mega-clusters** (few topics over many products) amplify rewrite failure; finer clustering helps routing, but does not replace keeping originals.

---

## 7. Practical comparison model

| Approach | Good for | Weak for |
|---|---|---|
| Contextual (raw + context prefix) | Names, FAQs, specific facts | Redundant near-dup chunks in top-k |
| Topic rewrite → embed | High-level narrative summaries | Product catalogs, proper nouns |
| Topic tags + originals + MMR (A+D) | Names + some topical focus + less redundancy | Still depends on cluster quality for routing |

---

## 8. Open follow-ups

- Near-duplicate collapse at **index time** (hash / embedding threshold) in addition to MMR
- Structured **product cards** `{name, summary, source_chunk_ids}` if catalog Q&A is primary
- Full **RAGAS** eval once `qa_pairs.json` is human-verified
- Cluster quality still coarse (2 topics in this run); tune generally via `.env` + `run_topic_clustering.py --reuse-embeddings` (not corpus-overfit)

---

## 9. Commands cheat sheet

```bash
# Build
uv run scripts/run_pipeline_contextual_rag.py
uv run scripts/run_pipeline_topic_modeling_rag.py

# Rebuild topic KB as tagged originals
uv run scripts/run_topic_clustering.py --run-id <TOPIC_RUN> --rebuild-kb-only

# Ask both
uv run scripts/ask.py --contextual-run <CTX_RUN> --topic-run <TOPIC_RUN> "..."

# List runs
uv run scripts/list_runs.py
```
