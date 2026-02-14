# Industrial Advanced RAG

Production-style Retrieval-Augmented Generation (RAG) service with:

- Hybrid retrieval (Dense + BM25)
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Source-grounded answers with citations
- FastAPI endpoints for ingest + query
- ChromaDB vector database (local persistent)
- Configurable LLM backend through LiteLLM

## Architecture

1. **Ingestion**
   - Load documents from a directory (`.txt`, `.md`, `.json`, `.pdf`)
   - Chunk with overlap
   - Embed chunks using SentenceTransformers
   - Upsert vectors into ChromaDB
   - Build BM25 index for lexical retrieval

2. **Retrieval**
   - Dense search in ChromaDB
   - BM25 lexical search
   - Fuse ranks via RRF
   - Rerank final candidates with cross-encoder

3. **Generation**
   - Prompt with top context snippets
   - Generate with any LiteLLM-compatible model
   - Return answer + citations

## Quick Start

### 1) Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 2) Configure

```bash
copy .env.example .env
```

Set at least:

- `LITELLM_MODEL` (example: `gpt-4.1-mini`, `azure/gpt-4.1-mini`, `ollama/llama3.1`)
- Required provider credentials (`OPENAI_API_KEY`, Azure vars, etc.)

### 3) Vector DB mode

ChromaDB is local and persistent by default:

- `CHROMA_COLLECTION=industrial_rag`
- `CHROMA_PATH=./chroma_db`

### 4) Start API

```bash
rag-api
```

Windows alternative (if PATH has issues):

```powershell
.\start_api.ps1
```

### 5) Ingest docs

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d "{\"source_dir\":\"./data\"}"
```

### 6) Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What are the warranty constraints?\"}"
```

## API

- `GET /health`
- `POST /ingest`
- `POST /query`

Interactive API GUI (Swagger):

- `GET /swagger`

## Notes

- For best quality, use domain-tuned embeddings/reranker and curated chunking rules.
- Add auth, audit logging, and policy filters before production deployment.

## Local LLM mode (no cloud key)

This project already supports local models through LiteLLM + Ollama.

Example `.env`:

- `LITELLM_MODEL=ollama/llama3.1:8b`
- `LITELLM_API_BASE=http://127.0.0.1:11434`

Then run locally:

1. Start Ollama service
2. Pull model: `ollama pull llama3.1:8b`
3. Start this API and call `/query`

Note: ChromaDB is a vector database, not a text generation model. You can use ChromaDB for retrieval storage, but generation still needs an LLM (e.g., Ollama local model).
