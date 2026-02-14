from collections import defaultdict

from rank_bm25 import BM25Okapi

from rag_service.config import get_settings
from rag_service.models import get_embedder, get_reranker
from rag_service.schemas import ChunkRecord
from rag_service.vector_store import VectorStore


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return dict(scores)


class HybridRetriever:
    def __init__(self, chunks: list[ChunkRecord], bm25: BM25Okapi | None) -> None:
        self.settings = get_settings()
        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}
        self.bm25 = bm25
        self.vector_store = VectorStore()
        self.embedder = get_embedder()
        self.reranker = get_reranker()

    def _dense(self, query: str) -> list[dict]:
        qvec = self.embedder.encode(query, normalize_embeddings=True).tolist()
        return self.vector_store.search(query_vector=qvec, limit=self.settings.top_k_dense)

    def _lexical(self, query: str) -> list[dict]:
        if self.bm25 is None or not self.chunks:
            return []
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[: self.settings.top_k_bm25]
        return [
            {
                "chunk_id": self.chunks[idx].chunk_id,
                "source": self.chunks[idx].source,
                "text": self.chunks[idx].text,
                "score": float(score),
            }
            for idx, score in ranked
        ]

    def retrieve(self, query: str) -> list[dict]:
        dense_hits = self._dense(query)
        lexical_hits = self._lexical(query)

        dense_ids = [h["chunk_id"] for h in dense_hits]
        lexical_ids = [h["chunk_id"] for h in lexical_hits]
        fused = reciprocal_rank_fusion([dense_ids, lexical_ids], k=self.settings.rrf_k)

        merged: dict[str, dict] = {}
        for hit in dense_hits + lexical_hits:
            merged[hit["chunk_id"]] = hit

        candidates = [
            {
                "chunk_id": chunk_id,
                "source": merged.get(chunk_id, {}).get("source", self.chunk_map.get(chunk_id, ChunkRecord(chunk_id=chunk_id, source="", text="")).source),
                "text": merged.get(chunk_id, {}).get("text", self.chunk_map.get(chunk_id, ChunkRecord(chunk_id=chunk_id, source="", text="")).text),
                "score": score,
            }
            for chunk_id, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)
        ]

        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates[: max(self.settings.top_k_final * 3, 10)]]
        rerank_scores = self.reranker.predict(pairs)

        reranked = []
        for item, rr_score in zip(candidates, rerank_scores, strict=False):
            item = {**item, "score": float(rr_score)}
            reranked.append(item)

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[: self.settings.top_k_final]
