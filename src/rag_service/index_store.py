import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag_service.schemas import ChunkRecord


class IndexStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunks_file = self.root / "chunks.json"
        self.bm25_file = self.root / "bm25.pkl"

    def save_chunks(self, chunks: list[ChunkRecord]) -> None:
        payload = [chunk.model_dump() for chunk in chunks]
        self.chunks_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load_chunks(self) -> list[ChunkRecord]:
        if not self.chunks_file.exists():
            return []
        payload = json.loads(self.chunks_file.read_text(encoding="utf-8"))
        return [ChunkRecord(**item) for item in payload]

    def save_bm25(self, bm25: BM25Okapi) -> None:
        with self.bm25_file.open("wb") as f:
            pickle.dump(bm25, f)

    def load_bm25(self) -> BM25Okapi | None:
        if not self.bm25_file.exists():
            return None
        with self.bm25_file.open("rb") as f:
            return pickle.load(f)
