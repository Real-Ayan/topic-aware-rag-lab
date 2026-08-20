"""Shared settings for the RAG experiment."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(ROOT_DIR / ".env"), str(ROOT_DIR / ".env.local")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI — GPT-5.6 family (2026): luna (cheap) | terra (balanced) | sol (flagship)
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-5.6-terra"
    CONTEXT_MODEL: str | None = None
    EVAL_MODEL: str | None = None
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    REASONING_EFFORT: str = "low"

    # Chunking (token-based via tiktoken)
    BIG_CHUNK_SIZE: int = 800
    BIG_CHUNK_OVERLAP: int = 100
    SMALL_CHUNK_SIZE: int = 350
    SMALL_CHUNK_OVERLAP: int = 50

    # Clustering / BERTopic (general defaults — tune per corpus via .env + re-cluster)
    CLUSTER_METHOD: str = "hdbscan"  # hdbscan | kmeans
    N_TOPICS: int = 10  # only used when CLUSTER_METHOD=kmeans
    UMAP_N_COMPONENTS: int = 5
    UMAP_N_NEIGHBORS: int = 15
    UMAP_MIN_DIST: float = 0.0
    HDBSCAN_MIN_CLUSTER_SIZE: int = 3
    HDBSCAN_MIN_SAMPLES: int | None = None  # None → HDBSCAN default
    HDBSCAN_SELECTION_METHOD: str = "eom"  # eom | leaf
    TOPIC_LABEL_SAMPLES: int = 10  # random excerpts per cluster for LLM labeling

    # Base KB names (each run creates {name}__{run_id})
    KNOWLEDGE_BASES_DIR: str = str(ROOT_DIR / "knowledge_bases")
    KB_CONTEXTUAL: str = "kb_contextual_rag"
    KB_TOPIC_MODELING: str = "kb_topic_modeling_rag"

    TOP_K: int = 5
    # Retrieve this many candidates before MMR / topic routing
    RETRIEVE_CANDIDATES: int = 20
    MMR_ENABLED: bool = True
    MMR_LAMBDA: float = 0.7  # 1=relevance only, 0=diversity only
    TOPIC_ROUTE_ENABLED: bool = True  # prefer majority topic among early hits
    TOPIC_DOC_BATCH_SIZE: int = 5

    DATA_RAW_DIR: str = str(ROOT_DIR / "data" / "raw")
    ARTIFACTS_DIR: str = str(ROOT_DIR / "artifacts")
    RUNS_DIR: str = str(ROOT_DIR / "runs")

    @property
    def context_model(self) -> str:
        return self.CONTEXT_MODEL or self.LLM_MODEL

    @property
    def eval_model(self) -> str:
        return self.EVAL_MODEL or self.LLM_MODEL


settings = Settings()
