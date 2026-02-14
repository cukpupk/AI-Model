from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_service.config import get_settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.reranker_model)
