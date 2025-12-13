from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Centralized configuration pulled from environment variables."""

    # ChromaDB Configuration
    chroma_persist_directory: str = Field(default="data/chroma_db", alias="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="smart-offer-finder", alias="CHROMA_COLLECTION_NAME")

    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    embedding_model: str = Field(default="ollama/multilingual-e5-base", alias="EMBEDDING_MODEL")

    # LLM Configuration (local or API-based)
    llm_model: str = Field(default="ollama/mistral", alias="LLM_MODEL")
    llm_base_url: Optional[str] = Field(default=None, alias="LLM_BASE_URL")
    llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")

    # Ingestion Configuration
    vectorstore_path: str = Field(default="data/vectorstore", alias="VECTORSTORE_PATH")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")

    # Reranker Configuration
    use_reranker: bool = Field(default=True, alias="USE_RERANKER")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    reranker_top_k: int = Field(default=4, alias="RERANKER_TOP_K")
    initial_retrieval_k: int = Field(default=20, alias="INITIAL_RETRIEVAL_K")

    # Hybrid Retriever Configuration
    use_hybrid_search: bool = Field(default=True, alias="USE_HYBRID_SEARCH")
    hybrid_bm25_weight: float = Field(default=0.4, alias="HYBRID_BM25_WEIGHT")
    hybrid_dense_weight: float = Field(default=0.6, alias="HYBRID_DENSE_WEIGHT")
    hybrid_bm25_k: int = Field(default=15, alias="HYBRID_BM25_K")
    hybrid_dense_k: int = Field(default=15, alias="HYBRID_DENSE_K")

    # Semantic Cache Configuration
    use_semantic_cache: bool = Field(default=False, alias="USE_SEMANTIC_CACHE")
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, alias="REDIS_PASSWORD")
    cache_similarity_threshold: float = Field(default=0.95, alias="CACHE_SIMILARITY_THRESHOLD")
    cache_ttl_seconds: int = Field(default=86400, alias="CACHE_TTL_SECONDS")


    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables (e.g., old Pinecone settings)


def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]
