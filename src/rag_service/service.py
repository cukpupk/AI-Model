from rag_service.config import get_settings
from rag_service.generation import generate_answer
from rag_service.index_store import IndexStore
from rag_service.retrieval import HybridRetriever
from rag_service.schemas import Citation, QueryResponse


class RAGService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_store = IndexStore(self.settings.index_dir)

    def query(self, question: str) -> QueryResponse:
        chunks = self.index_store.load_chunks()
        bm25 = self.index_store.load_bm25()
        retriever = HybridRetriever(chunks=chunks, bm25=bm25)

        hits = retriever.retrieve(question)
        if not hits:
            return QueryResponse(
                answer="No relevant knowledge found. Please ingest documents first.",
                citations=[],
                retrieved_chunks=0,
            )

        answer = generate_answer(question, hits)
        citations = [
            Citation(source=h["source"], chunk_id=h["chunk_id"], score=float(h["score"]))
            for h in hits
        ]
        return QueryResponse(answer=answer, citations=citations, retrieved_chunks=len(hits))
