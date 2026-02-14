from pathlib import Path

from rank_bm25 import BM25Okapi

from rag_service.chunking import chunk_document
from rag_service.config import get_settings
from rag_service.index_store import IndexStore
from rag_service.loaders import collect_documents
from rag_service.models import get_embedder
from rag_service.schemas import ChunkRecord
from rag_service.vector_store import VectorStore


class IngestionService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_store = IndexStore(self.settings.index_dir)
        self.embedder = get_embedder()
        self.vector_store = VectorStore()

    def ingest(self, source_dir: Path, glob_pattern: str = "**/*") -> tuple[int, int]:
        docs = collect_documents(source_dir=source_dir, glob_pattern=glob_pattern)

        all_chunks: list[ChunkRecord] = []
        for path, text in docs:
            all_chunks.extend(
                chunk_document(
                    path=path,
                    text=text,
                    size=self.settings.chunk_size,
                    overlap=self.settings.chunk_overlap,
                )
            )

        if not all_chunks:
            self.index_store.save_chunks([])
            return len(docs), 0

        texts = [c.text for c in all_chunks]
        vectors = self.embedder.encode(texts, normalize_embeddings=True).tolist()
        self.vector_store.replace_all(chunks=all_chunks, vectors=vectors)

        tokenized = [t.split() for t in texts]
        bm25 = BM25Okapi(tokenized)

        self.index_store.save_chunks(all_chunks)
        self.index_store.save_bm25(bm25)
        return len(docs), len(all_chunks)
