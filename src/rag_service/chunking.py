import hashlib
from pathlib import Path

from rag_service.schemas import ChunkRecord


def _hash_chunk(source: str, text: str, index: int) -> str:
    raw = f"{source}::{index}::{text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:24]


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= overlap:
        raise ValueError("chunk size must be larger than overlap")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def chunk_document(path: Path, text: str, size: int, overlap: int) -> list[ChunkRecord]:
    source = str(path.as_posix())
    parts = split_text(text=text, size=size, overlap=overlap)
    return [
        ChunkRecord(chunk_id=_hash_chunk(source, part, i), source=source, text=part)
        for i, part in enumerate(parts)
    ]
