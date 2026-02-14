from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from rag_service.config import get_settings
from rag_service.ingest import IngestionService
from rag_service.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from rag_service.service import RAGService

app = FastAPI(
    title="Industrial Advanced RAG",
    version="0.1.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/swagger")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    source = Path(req.source_dir)
    if not source.exists() or not source.is_dir():
        raise HTTPException(status_code=400, detail="source_dir must exist and be a directory")

    service = IngestionService()
    docs, chunks = service.ingest(source_dir=source, glob_pattern=req.glob)
    return IngestResponse(documents=docs, chunks=chunks, message="Ingestion completed")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    service = RAGService()
    return service.query(req.query)


def run() -> None:
    settings = get_settings()
    uvicorn.run("rag_service.api:app", host=settings.app_host, port=settings.app_port, reload=False)
