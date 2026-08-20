# RAG Concept Research: Topic Modeling + RAG

> Research notes from design discussions (2026-08-20).  
> Scope: contextual RAG, data cleaning, topic discovery (UMAP/HDBSCAN/BERTopic), dynamic knowledge MD, metadata isolation, and graph-style uses of topics.  
> Status: **research / design** — not yet an implementation checklist for production code.

---

## Table of Contents

1. [Why this research exists](#1-why-this-research-exists)
2. [Problems we saw in naive RAG](#2-problems-we-saw-in-naive-rag)
3. [Contextual RAG (baseline enrichment)](#3-contextual-rag-baseline-enrichment)
4. [Metadata vs embedding text](#4-metadata-vs-embedding-text)
5. [Ingestion CLI → later REST](#5-ingestion-cli--later-rest)
6. [PDF / text loading (incl. VLM)](#6-pdf--text-loading-incl-vlm)
7. [Shared embedding & vector settings](#7-shared-embedding--vector-settings)
8. [Data cleaning: why it matters more than context prefixes](#8-data-cleaning-why-it-matters-more-than-context-prefixes)
9. [Two-pass cleaning & dynamic knowledge MD](#9-two-pass-cleaning--dynamic-knowledge-md)
10. [Topic modeling for discovery (not as the cleaned KB)](#10-topic-modeling-for-discovery-not-as-the-cleaned-kb)
11. [Recommended pipeline order](#11-recommended-pipeline-order)
12. [Uses of topic modeling with RAG](#12-uses-of-topic-modeling-with-rag)
13. [Topic tags → light Graph RAG](#13-topic-tags--light-graph-rag)
14. [Study materials (papers, code, docs)](#14-study-materials-papers-code-docs)
15. [Suggested POC build order (this repo)](#15-suggested-poc-build-order-this-repo)
16. [Open questions](#16-open-questions)
17. [Glossary](#17-glossary)

---

## 1. Why this research exists

We want a company chatbot RAG over messy PDFs/text (later HTML too). A plain “chunk → embed → top-k” pipeline fails on:

- Unknown / evolving topics (not only “projects” and “company profile”)
- Duplicate facts across sources
- Old vs new information competing in the same vector space
- Mixed topics inside a single geometric chunk
- Metadata leakage if we stuff source/ops fields into the text the model sees

This note captures a research direction: **topic modeling for discovery + LLM canonicalization + contextual embedding**, with optional **topic-filtered** and **graph-expanded** retrieval later.

It aligns with (but goes beyond) the ingestion sketch and the vector repository idea in §7.

---

## 2. Problems we saw in naive RAG

| Symptom | Likely cause |
|---|---|
| Same address retrieved from 3 different docs | Near-duplicate facts; top-k wasted on duplicates |
| Old project facts rank above newer ones | Cosine similarity ignores time/`status` |
| Event details mixed with project data in one chunk | Recursive splitter is size-based, not semantic |
| Orphan chunks with no company/project name | Chunk text lacks situating context |
| Fear of leaking source/path to the client | Metadata mixed into the string returned to the LLM/user |

**Important split:**

- **Contextual RAG** helps orphan / situating problems.
- **Cleaning + topic structure** helps duplicates, time conflicts, and topic soup.
- **Metadata columns + API projection** helps leakage control.

Do not expect contextual prefixes alone to fix dirty corpora.

---

## 3. Contextual RAG (baseline enrichment)

### Idea

Anthropic-style (and similar) **contextual retrieval**:

1. Split document into chunks (e.g. `RecursiveCharacterTextSplitter`).
2. For each chunk, ask an LLM for a short **situating context**  
   (“This chunk is from Pacco’s 2024 case study on Project Alpha…”).
3. Embed `context + "\n\n" + content` (fixed template).
4. Store content (and optionally context) for generation; keep ops fields in metadata.

### Does it make sense for us?

**Yes**, as a retrieval-quality layer **after** (or on top of) cleaner units.  
It is **not** a substitute for dedup, temporal resolution, or topic isolation.

### Chunking defaults (from product plan; tunable)

- Chunk size ~512–800 tokens (plan often cites ~600)
- Overlap ~100 tokens
- Recursive character splitter is fine for POC; prefer section/heading-aware splits when structure exists

---

## 4. Metadata vs embedding text

### Question

Should `source` live inside the embedded string (`<context><content><metadata>`)?

### Recommendation

| Field | Where | Purpose |
|---|---|---|
| `content` | DB column | What the answer path should quote |
| Contextualized text | What gets vectorized (`context + content`) | Similarity search |
| `source`, `page_number`, `doc_id`, `topic`, `ingested_at`, `as_of`, `status` | **Columns / JSONB** | Filter, cite, re-ingest, ACL |
| Client/tool DTO | Projected subset | Leakage control |

**Rules of thumb:**

- Canonical source belongs in **DB metadata**, not only inside embedding text.
- A short situating phrase mentioning the document title *inside* the LLM context string can help retrieval — still keep the canonical `source` column.
- Do **not** use free-form `<metadata>` blobs in the embedding string as the system of record: you lose SQL filters, isolation, and safe client projection.
- Leakage is controlled at the **rag tool / API boundary**, not by omitting DB columns.

### Current schema touchpoint

Existing model sketch (`rag.document_chunks`): `content`, `embedding`, `source`, `page_number` — good starting point; later extend with `topic`, `document_id`, hashes, `as_of` / `status` as needed.

---

## 5. Ingestion CLI → later REST

### Decision for POC

- Start with a **CLI** for ingest (manual when data changes).
- Same pipeline modules later behind `POST /api/v1/documents` (async job).

### Composable stages

```
load → (optional discover/clean) → chunk → contextualize → embed → vector_repo.add
```

CLI examples (conceptual):

```bash
uv run python -m app.ingestion.cli ingest ./data/foo.pdf
uv run python -m app.ingestion.cli ingest ./data/notes.txt
uv run python -m app.ingestion.cli discover-topics ./data/raw
uv run python -m app.ingestion.cli build-knowledge ./data/raw --out ./knowledge
uv run python -m app.ingestion.cli embed-knowledge ./knowledge
```

CLI and REST must call the **same** embeddings factory + vector repository.

---

## 6. PDF / text loading (incl. VLM)

### Text files

Direct load → same clean/chunk/contextualize/embed path.

### PDFs

```
PDF → PyMuPDF (fitz)
  ├─ enough extractable text → text path
  └─ image/scanned → VLM on page images → derived text
       → store derived text as a document record (avoid re-calling VLM on every re-embed)
       → same downstream pipeline
```

Keep raw + hash for audit and re-runs.

---

## 7. Shared embedding & vector settings

Group everything used by **both ingestion and retrieval**:

| Concern | Suggested home |
|---|---|
| Embedding model, dims (1536), API key | Shared settings (`LLMSettings` / `RAGSettings`) |
| Chunk size, overlap, contextualize prompt | RAG config |
| Vector backend (`neon` / local) | `VECTOR_STORE_BACKEND` |
| Add / search / delete-by-source | `VectorRepository` abstraction |

Pattern from plan:

```
app/repositories/vector/
  base.py      # add_documents(), similarity_search(), delete_by_source()
  neon.py      # pgvector
```

One embeddings client; one distance metric (cosine / HNSW as already indexed).  
**HNSW** = vector neighbor index in pgvector. **HDBSCAN** = clustering for topics. Do not confuse them.

---

## 8. Data cleaning: why it matters more than context prefixes

Dirty multi-source company data produces:

1. **Duplicate facts, many sources**
2. **Temporal conflict** (old ≈ new in embedding space)
3. **Topic soup** in geometric chunks

Contextual RAG slightly helps (1) via better situating; barely helps (2)/(3).

**Priority:** structure and clean before (or instead of) embedding raw PDF sludge long-term.

---

## 9. Two-pass cleaning & dynamic knowledge MD

### Constraint

We do **not** know the full topic list in advance. Projects and company profile are examples only; new concepts appear over time → need **dynamic** knowledge files / units.

### Anti-pattern

One ever-growing mega-markdown becomes a merge/hallucination sink.

### Preferred shape

Typed units (files or DB rows), upserted by **stable slug**:

```text
knowledge/
  _index.json                 # slug → path, status, updated_at
  company_address.md
  project_alpha.md
  event_launch_2024.md
  ...
```

Frontmatter example:

```yaml
---
slug: company_address
status: active
as_of: 2025-01
sources:
  - brochure.pdf#p2
  - about.html
---
```

### Two passes (conceptual)

**Pass A — Inventory / discovery**  
Find concepts/topics + provenance (sources, pages, snippets).  
Can be LLM map-reduce **or** clustering (see §10).

**Pass B — Materialize clean units**  
For each topic/entity: gather snippets → LLM writes one canonical unit with:

- resolved current vs historical (`as_of`, `status`)
- citations / sources
- **one topic per unit** (no event+project mashup)

Then chunk those units (often 1 unit ≈ 1 chunk) and run contextual RAG on **clean** text.

### Risks

- LLM merge can invent “current” facts → cite-or-drop; keep raw docs
- Full-corpus “see everything” only scales to POC size; later: per-doc then merge by entity key
- Re-runs must **diff/upsert** `_index.json`, not wipe and renumber forever

---

## 10. Topic modeling for discovery (not as the cleaned KB)

### Clarification

Clustering is for **topic discovery / grouping**.  
LLM summarization/labeling + canonical MD is for **cleaning**.  
Clusters ≠ cleaned knowledge base.

### Pipeline (research target)

```text
chunks → embed → UMAP → HDBSCAN → clusters
                                   ↓
                      LLM summarizes cluster snippets → topic label + slug
                                   ↓
                      write/update knowledge/{slug}.md
                                   ↓
                      contextualize → embed → pgvector
```

This is close to **BERTopic** (embed → UMAP → HDBSCAN → topic representation), with LLM naming instead of (or in addition to) c-TF-IDF keywords.

### PCA / SVD note

PCA/SVD + cluster can work but is blunter than UMAP+HDBSCAN for modern embedding topic models. Prefer BERTopic or UMAP+HDBSCAN directly.

### Embed chunks, not whole multi-topic PDFs

A full brochure as one vector collapses mixed themes. Discover topics on **chunk/section** embeddings.

### Cluster IDs are unstable

Never use `cluster_12` as a permanent product ID.  
Use LLM-derived **slugs** (`company_address`, `project_alpha`) and upsert.

### LLM-first vs cluster-first

| Order | When |
|---|---|
| **Cluster → LLM label → MD** | Unknown schema, messy multi-doc corpus (preferred for “I don’t know what’s in the data”) |
| **LLM inventory → optional cluster** | Tiny corpus; themes mostly known; clustering is optional QA |
| **Hybrid** | Clusters propose groups; LLM names, merges, writes MD; human glance on `_index` |

**Recommendation for this project:** cluster-first discovery; LLM for naming + canonical MD.

---

## 11. Recommended pipeline order

End-to-end research pipeline:

```text
raw PDFs / text
  → extract (VLM if scanned)
  → section / chunk
  → embed chunks
  → UMAP + HDBSCAN (or BERTopic)
  → LLM: label topics + assign / merge → knowledge/{slug}.md
  → contextualize each clean unit
  → embed → rag.document_chunks (+ topic metadata)
```

Retrieval v1: similarity over clean units.  
Retrieval v2: topic filter and/or graph expansion (see §12–§13).

---

## 12. Uses of topic modeling with RAG

Practical uses, light → heavy:

### A. Topic as metadata filter (high ROI)

Offline: assign `topic_slug` on chunks.  
Online: map query → 1–3 topics → search **only inside** those topics.

Fixes cross-topic pollution (events vs projects).

### B. Topic-scoped cleaning / canonical MD

Offline discovery → dynamic knowledge units → embed those.

Fixes duplicates and unknown schema.

### C. Multi-topic routing for the agent

Broad questions (“Tell me about the company”) → agent picks topic scopes / tools instead of one flat top-k.

### D. Dedup / merge within a topic

Near-duplicates inside one cluster → one canonical unit.

### E. Hierarchical topics

Coarse topic → re-cluster → subtopics (`projects` / `project_alpha`).

### F. Query-time topic assignment

BERTopic/centroid match on the query (as in Topic-RAG / AT-RAG). Defer until clean units work.

### G. Community / theme summaries

Summarize each topic/community; embed summaries for **global** questions (“What themes appear in our corpus?”). Related to GraphRAG community reports.

### H. Graph expansion from topic tags

See next section.

---

## 13. Topic tags → light Graph RAG

### Idea

Topics become **tags/nodes**. Connect chunks (and topics) so retrieval can find **related** context beyond pure cosine top-k.

### Useful graph shapes

```text
[Topic: projects] --contains--> [Chunk A]
[Topic: projects] --contains--> [Chunk B]
[Chunk A] --same_source--> [Chunk B]
[Chunk A] --co_occurs / related--> [Chunk C]
[Topic: projects] --related_to--> [Topic: clients]
[Entity: Project Alpha] --mentioned_in--> [Chunk A]
```

### Edge sources

| Edge | How |
|---|---|
| chunk → topic | Cluster membership / soft membership |
| topic ↔ topic | Shared chunks, centroid distance, or LLM “related” |
| chunk ↔ chunk | Same doc/page, overlap, high cosine, co-citation in MD |
| entity → chunk | Optional NER / LLM pass on cluster summaries |
| supersedes / older_than | Dates + LLM inside a topic (temporal conflicts) |

### Retrieval pattern

1. Vector search → seed chunk(s)  
2. Expand along edges (same topic, related topic, same entity)  
3. Rerank → LLM context  

### POC without Neo4j

Stay on Postgres/Neon:

- `document_chunks.topic_slug`
- `topics(slug, label, summary, …)`
- `topic_relations(a, b, weight)`
- Expand with SQL filters / second searches; no graph DB required initially.

### What not to do first

- Full Microsoft GraphRAG (entity extract + Leiden communities + community reports) on day one  
- Unstable cluster IDs as node IDs  
- Connecting every high-cosine pair (noisy complete graph)

**Build order:** topic tags + clean MD → topic filter → light topic-neighborhood expansion → richer entity graph if needed.

---

## 14. Study materials (papers, code, docs)

> **Why “Topic-RAG” is hard to Google:** it is mainly a **paper/framework name** (2025, historical newspapers + BERTopic), not a mainstream product keyword. Search for **“Topic-RAG BERTopic”**, **“topic-based retrieval-augmented generation”**, or **“AT-RAG BERTopic”**. Broader terms: **“topic-aware RAG”**, **“BERTopic RAG”**, **“GraphRAG”**.

### 14.1 Topic-RAG (exact name you asked about)

| Resource | Link | What to study |
|---|---|---|
| Paper (Cambridge / CHR) | [Cambridge Core article](https://www.cambridge.org/core/journals/computational-humanities-research/article/topicrag-for-historical-newspapers-enhancing-information-retrieval-in-humanities-research-through-topicbased-retrievalaugmented-generation/054E6DA3416EA982F61858B30E7CAB6D) | BERTopic maps query → topics; retrieve only inside those topics; vs baseline RAG |
| DOI | [https://doi.org/10.1017/chr.2025.10018](https://doi.org/10.1017/chr.2025.10018) | Canonical citation |
| PDF (Cambridge) | [Direct PDF link](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/054E6DA3416EA982F61858B30E7CAB6D/S2977815825100183a.pdf/topic-rag-for-historical-newspapers-enhancing-information-retrieval-in-humanities-research-through-topic-based-retrieval-augmented-generation.pdf) | Full text if accessible |
| GitHub (replication) | [KeerthanaMurugaraj/Topic-RAG-for-Historical-Newspapers](https://github.com/KeerthanaMurugaraj/Topic-RAG-for-Historical-Newspapers) | `topic_rag.py`, `topic_rag_plus.py` (chunking for long docs) |

**Core takeaway:** topic modeling as a **search-space filter** for RAG, not only as offline labeling.

### 14.2 AT-RAG (topic filtering + iterative reasoning)

| Resource | Link | What to study |
|---|---|---|
| Paper (arXiv HTML) | [AT-RAG: Adaptive Topic RAG](https://arxiv.org/html/2410.12886v1) | BERTopic assigns topics to queries; multi-hop / CoT-style retrieval |
| arXiv PDF | [https://arxiv.org/pdf/2410.12886](https://arxiv.org/pdf/2410.12886) | Downloadable PDF |
| GitHub | [MrRezaeiUofT/AT-RAG](https://github.com/MrRezaeiUofT/AT-RAG) | Trainer + topic-aware RAG code |

### 14.3 ChatCM-RAG (BERTopic + RAG in medicine lit)

| Resource | Link | What to study |
|---|---|---|
| Journal page | [ChatCM-RAG (CTD)](https://journal.hep.com.cn/ctd/EN/10.1002/ctd2.70136) | UMAP + HDBSCAN topics; topic-aware retrieval metrics |
| Hugging Face note (from paper) | Search `fc28/ChatCM-RAG` on Hugging Face | Released artifacts if still public |

### 14.4 BERTopic / clustering stack (foundations)

| Resource | Link | What to study |
|---|---|---|
| BERTopic docs | [https://maartengr.github.io/BERTopic/](https://maartengr.github.io/BERTopic/) | Embed → UMAP → HDBSCAN → c-TF-IDF; customization |
| BERTopic GitHub | [MaartenGr/BERTopic](https://github.com/MaartenGr/BERTopic) | Implementation reference |
| Original BERTopic paper | Search “BERTopic Grootendorst 2022” | Why neural topic models beat classic LDA for many corpora |
| UMAP | [UMAP docs](https://umap-learn.readthedocs.io/) | Dimensionality reduction used before clustering |
| HDBSCAN | [HDBSCAN docs](https://hdbscan.readthedocs.io/) | Density clustering; noise/`-1` outliers |

Related historical-topic comparison (same authors as Topic-RAG lineage):

- ACL Anthology: [Mining the Past: Classical vs Neural Topic Models on Historical Newspapers](https://aclanthology.org/2025.nlp4dh-1.39/)

### 14.5 Contextual RAG / enrichment

| Resource | Link | What to study |
|---|---|---|
| Anthropic — Contextual Retrieval | [anthropic.com engineering: contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) | Context prefix per chunk; BM25 + embedding hybrid often discussed alongside |

### 14.6 Graph RAG (topics/communities + connected retrieval)

Not the same as BERTopic, but the closest “connect clusters and retrieve related context” industrial system:

| Resource | Link | What to study |
|---|---|---|
| GraphRAG docs | [https://microsoft.github.io/graphrag/](https://microsoft.github.io/graphrag/) | Entity graph, communities, global vs local search |
| MSR paper page | [From Local to Global: Graph RAG…](https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/) | Motivation for community summaries |
| GitHub | [microsoft/graphrag](https://github.com/microsoft/graphrag) | Reference implementation |
| Intro blog | [GraphRAG unlocking LLM discovery](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/) | High-level narrative |

**How it relates to our notes:** GraphRAG builds **entity graphs + Leiden communities + summaries**. Our lighter path is **BERTopic-style topics as tags** → optional edges → expand. Same *family* of ideas (structure beyond flat vectors); different extraction stack.

### 14.7 Search queries that actually find things

If Google fails on `topic_rag`, try:

```text
Topic-RAG BERTopic historical newspapers
topic-based retrieval-augmented generation BERTopic
AT-RAG BERTopic adaptive topic
topic-aware RAG
BERTopic retrieval augmented generation
GraphRAG community summary RAG
contextual retrieval Anthropic
```

Scholar / arXiv often ranks these better than generic web search.

---

## 15. Suggested POC build order (this repo)

Aligned with discussion and the existing ingestion pipeline:

1. **Shared** `RAG`/embedding settings + embeddings factory + `VectorRepository` (Neon/pgvector) on `rag.document_chunks`
2. **CLI ingest** for `.txt` + PDF (text path; VLM path stub or second)
3. Recursive chunk → contextualize → embed → upsert; `delete_by_source` / content hash for re-ingest
4. **Offline** `discover-topics` (BERTopic or UMAP+HDBSCAN) → LLM labels → `knowledge/{slug}.md`
5. Prefer embedding **clean units** over raw PDF chunks once knowledge exists
6. Store `topic_slug` on chunks; add topic filter in rag tool
7. Optional: topic-neighborhood expansion (light graph) without Neo4j
8. Promote same pipeline to REST upload jobs

Defer: full GraphRAG entity+Leiden stack, query-time BERTopic on every request, perfect temporal graphs.

---

## 16. Open questions

- Soft multi-label topics per chunk vs single hard assignment?
- How aggressive should HDBSCAN `min_cluster_size` be on a small company corpus?
- Human review gate on `_index.json` before embed?
- Store `context` as its own column vs only inside the embedded string?
- When to introduce BM25 / hybrid search alongside vectors?
- Entity nodes (NER) vs topic-only graph for POC?

---

## 17. Glossary

| Term | Meaning here |
|---|---|
| **Contextual RAG** | LLM-written situating prefix + chunk content embedded together |
| **Topic discovery** | Unsupervised grouping (e.g. UMAP+HDBSCAN) to find unknown themes |
| **Canonical / clean unit** | One topic-scoped MD or DB row after LLM merge |
| **Topic filter** | Restrict vector search to chunks with matching `topic_slug` |
| **HNSW** | Approximate nearest-neighbor index for vectors (e.g. pgvector) |
| **HDBSCAN** | Density-based clustering used after UMAP in BERTopic-style pipelines |
| **BERTopic** | Library/pipeline: embeddings → UMAP → HDBSCAN → topic words/labels |
| **Graph RAG (light)** | Retrieve seed by vector, expand via topic/entity edges |
| **GraphRAG (Microsoft)** | Specific system: LLM entity graph + communities + summaries |
| **Topic-RAG** | Specific paper/system: BERTopic-guided topic-restricted RAG over archives |

---

## Document history

| Date | Note |
|---|---|
| 2026-08-20 | Initial research write-up from design chat: contextual RAG, cleaning, topic modeling, dynamic MD, study links, graph uses |

