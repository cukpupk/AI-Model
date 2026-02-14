from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_dir: str
    glob: str = "**/*"


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    message: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: int


class ChunkRecord(BaseModel):
    chunk_id: str
    source: str
    text: str
