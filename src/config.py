from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Centralized configuration pulled from environment variables."""

    # Pinecone Configuration
    pinecone_api_key: Optional[str] = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="smart-offer-finder", alias="PINECONE_INDEX_NAME")

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

    class Config:
        env_file = ".env"
        case_sensitive = False


def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]
