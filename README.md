# Document RAG Chatbot

A Retrieval-Augmented Generation chatbot with two-tier conversational memory, built to understand and extend an existing RAG architecture end-to-end (FastAPI + WebSocket streaming backend, Streamlit frontend, local LLM via Ollama).

This project was built by fully reverse-engineering and reimplementing [the original design](https://github.com/vyaduvanshi/responsive) file-by-file, then fixing several correctness and robustness issues found along the way (see below).

![System design](assets/rag-chatbot-system-design.png)

## Run it

### Option A — Docker

```
docker compose up --build
```
Wait for Ollama to finish pulling `llama3.1:8b` on first boot, then visit:
- Streamlit UI → http://localhost:8501
- FastAPI backend → http://localhost:8000

### Option B — Native (no Docker)

Requires [Ollama](https://ollama.com/download) and Redis installed locally.

```
ollama pull llama3.1:8b

# backend
cd backend
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.backend.txt
set REDIS_URL=redis://localhost:6379
set OLLAMA_URL=http://localhost:11434
uvicorn app:app --reload --port 8000

# frontend (separate terminal)
cd frontend
python -m venv venv && venv\Scripts\activate
pip install -r requirements.frontend.txt
set BACKEND_URL=http://localhost:8000
streamlit run app.py
```

## Architecture

```
Streamlit (8501) → REST/WebSocket → FastAPI (8000) → Redis (STM) + SQLite (durable state) + ChromaDB (vectors) + Ollama (11434, LLM)
```

**Prompt = Query + Retrieved Document Context + Short-Term Memory (STM) + Retrieved Long-Term Memory (LTM)**

- **Short-term memory**: every turn is written to Redis (hot path) and mirrored to SQLite (durable backup). If Redis is empty on read (restart, session switch), it's transparently rehydrated from SQLite.
- **Long-term memory**: once STM exceeds a token threshold, it's summarized by the LLM, the summary is embedded and stored in ChromaDB, and STM is trimmed — keeping the active context small without losing earlier conversation.
- **Retrieval**: each query is embedded once and used to search both the document-chunk collection (top-k) and the LTM-summary collection (top-1) in ChromaDB, scoped by session via metadata filtering.
- **Graceful prompt degradation**: if the assembled prompt exceeds the token budget, it degrades in stages — shrink chunks → keep 1 chunk → keep last 2 STM turns → truncate the user message — rather than failing outright.

| Component | Storage |
|---|---|
| Sessions, documents, chunks, chat history, STM, LTM summaries | SQLite |
| Chunk & LTM embeddings | ChromaDB |
| Short-term memory (hot path) | Redis |

## Notable fixes made during the rebuild

Reimplementing the original surfaced a handful of real issues, fixed here:

- **Redundant embedding calls**: the retrieval step embedded the same query twice (once per vector search). Now embedded once and reused — halves embedding cost per turn.
- **Silent failure on bad uploads**: unsupported/empty files used to create a "successful" session with zero retrievable content. Now raises a clear error surfaced all the way to the UI.
- **Type mismatch**: a foreign-key-like column was declared `String` but stored `Integer` IDs — silently worked on SQLite, would break on Postgres.
- **Dead configuration**: `REDIS_URL`/`OLLAMA_URL`/`BACKEND_URL` were defined in `docker-compose.yml` but never actually read by the code (hardcoded instead). Now wired through properly.
- **Fixed-sleep startup race**: Ollama's init script waited a flat 10 seconds before pulling the model. Now polls until the server is actually ready.

## Stack

`FastAPI` · `WebSockets` · `Streamlit` · `SQLAlchemy (async)` · `SQLite` · `Redis` · `ChromaDB` · `Ollama (Llama 3.1 8B)` · `sentence-transformers` · `LangChain text splitters` · `Docker Compose`

## Possible extensions

- Hybrid search (BM25 + dense vector, fused via reciprocal rank fusion)
- Retrieval evaluation harness (recall@k, MRR) + LLM-as-judge generation scoring
- Cross-encoder reranking on top of hybrid retrieval
- Multi-document sessions
