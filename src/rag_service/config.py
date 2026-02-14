from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="dev", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    chroma_path: Path = Field(default=Path("./chroma_db"), alias="CHROMA_PATH")
    chroma_collection: str = Field(default="industrial_rag", alias="CHROMA_COLLECTION")

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    index_dir: Path = Field(default=Path("./index"), alias="INDEX_DIR")

    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5", alias="EMBEDDING_MODEL")
    reranker_model: str = Field(default="BAAI/bge-reranker-base", alias="RERANKER_MODEL")

    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=180, alias="CHUNK_OVERLAP")
    top_k_dense: int = Field(default=20, alias="TOP_K_DENSE")
    top_k_bm25: int = Field(default=20, alias="TOP_K_BM25")
    top_k_final: int = Field(default=8, alias="TOP_K_FINAL")
    rrf_k: int = Field(default=60, alias="RRF_K")

    litellm_model: str = Field(default="gpt-4.1-mini", alias="LITELLM_MODEL")
    litellm_api_base: str = Field(default="", alias="LITELLM_API_BASE")
    litellm_temperature: float = Field(default=0.1, alias="LITELLM_TEMPERATURE")
    litellm_max_tokens: int = Field(default=700, alias="LITELLM_MAX_TOKENS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return settings
