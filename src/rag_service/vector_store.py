import chromadb
from chromadb.api.models.Collection import Collection

from rag_service.config import get_settings
from rag_service.schemas import ChunkRecord


class VectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = chromadb.PersistentClient(path=str(self.settings.chroma_path))

    def _get_or_create_collection(self) -> Collection:
        return self.client.get_or_create_collection(
            name=self.settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def replace_all(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        try:
            self.client.get_collection(name=self.settings.chroma_collection)
            self.client.delete_collection(name=self.settings.chroma_collection)
        except Exception:
            pass
        collection = self._get_or_create_collection()

        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [{"source": c.source, "chunk_id": c.chunk_id} for c in chunks]
        collection.add(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)

    def search(self, query_vector: list[float], limit: int) -> list[dict]:
        try:
            collection = self.client.get_collection(name=self.settings.chroma_collection)
        except Exception:
            return []

        result = collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            include=["metadatas", "documents", "distances"],
        )

        metadatas = (result.get("metadatas") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[dict] = []
        for metadata, doc, distance in zip(metadatas, documents, distances, strict=False):
            md = metadata or {}
            dist = float(distance) if distance is not None else 1.0
            hits.append(
                {
                    "chunk_id": md.get("chunk_id", ""),
                    "source": md.get("source", ""),
                    "text": doc or "",
                    "score": 1.0 - dist,
                }
            )
        return hits
