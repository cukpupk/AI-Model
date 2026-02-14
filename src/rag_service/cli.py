from pathlib import Path

import typer

from rag_service.ingest import IngestionService
from rag_service.service import RAGService

app = typer.Typer(help="Industrial RAG CLI")


@app.command()
def ingest(source_dir: str, glob: str = "**/*") -> None:
    service = IngestionService()
    docs, chunks = service.ingest(Path(source_dir), glob)
    typer.echo(f"Ingested {docs} documents, {chunks} chunks")


@app.command()
def query(text: str) -> None:
    service = RAGService()
    result = service.query(text)
    typer.echo(result.answer)
    typer.echo("\nCitations:")
    for c in result.citations:
        typer.echo(f"- {c.source} [{c.chunk_id}] score={c.score:.4f}")
