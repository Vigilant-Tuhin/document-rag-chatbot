# Document RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot with two-tier conversational memory, hybrid retrieval (dense vector + keyword search), and a custom evaluation harness for measuring retrieval and generation quality.

Backend: FastAPI + WebSocket streaming. Frontend: Streamlit. LLM: Llama 3.1 8B served locally via Ollama. Vector store: ChromaDB. Relational store: SQLite. Cache: Redis.

![System design](assets/rag-chatbot-system-design.png)

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Memory management](#memory-management)
- [Retrieval pipeline](#retrieval-pipeline)
- [Evaluation](#evaluation)
- [Tech stack](#tech-stack)
- [Run it](#run-it)
- [Project structure](#project-structure)
- [Engineering decisions & fixes](#engineering-decisions--fixes)
- [Possible extensions](#possible-extensions)

---

## What it does

Upload a document (`.txt`, `.md`, `.docx`, `.pdf`), then ask questions about it in a chat interface. The system:

1. Splits the document into overlapping chunks, embeds them, and stores them in a vector database alongside a keyword index.
2. On each question, retrieves relevant chunks using **hybrid search** (dense vector similarity + BM25 keyword matching, fused via Reciprocal Rank Fusion).
3. Maintains conversational memory across the session with a two-tier system: fast short-term memory that gets automatically summarized into long-term memory once it grows too large, so the model can "remember" earlier parts of a long conversation without ballooning the prompt.
4. Streams the LLM's response back token-by-token over a WebSocket.

Each chat session is tied to exactly one uploaded document.

## Architecture

```
┌────────────┐   REST (upload, sessions)   ┌─────────────┐
│  Streamlit  │ ──────────────────────────▶ │   FastAPI    │
│  (8501)     │ ◀── WebSocket (streaming) ─ │   (8000)     │
└────────────┘                              └──────┬──────┘
                                                     │
                     ┌───────────────┬───────────────┼───────────────┬──────────────┐
                     ▼               ▼               ▼               ▼              ▼
                 ┌───────┐     ┌──────────┐    ┌───────────┐   ┌──────────┐   ┌───────────┐
                 │ Redis │     │  SQLite  │    │  ChromaDB  │   │  SQLite   │   │  Ollama   │
                 │ (STM) │     │(durable  │    │ (chunk +   │   │  FTS5     │   │ (11434)   │
                 │       │     │ state)   │    │ LTM vectors)│   │(keyword   │   │  LLM +    │
                 │       │     │          │    │            │   │  index)   │   │  judge    │
                 └───────┘     └──────────┘    └───────────┘   └──────────┘   └───────────┘
```

Embeddings for retrieval are computed locally with `sentence-transformers` (not through Ollama). Ollama is used only for text generation, summarization, session-title generation, and (in the eval harness) as an LLM judge.

## Memory management

**Prompt = Query + Retrieved Document Context + Short-Term Memory (STM) + Retrieved Long-Term Memory (LTM)**

**Short-term memory (STM)**
- Every message (user and assistant) is written to Redis (hot path, fast reads) and mirrored to SQLite (durable backup) in the same operation.
- If Redis is empty when read — after a restart, or when switching to a session that wasn't recently active — it's transparently rehydrated from SQLite. Callers never need to know which storage tier actually served the data.

**Long-term memory (LTM)**
- STM length is tracked in estimated tokens. Once it crosses a threshold (2000 tokens), the full STM is summarized by the LLM.
- The summary is stored as a row in SQLite **and** embedded into its own ChromaDB collection, so future queries can semantically recall it.
- STM is then trimmed (Redis keeps the last 4 turns, SQLite keeps the last 3) rather than wiped entirely, so recent context isn't lost the instant a summarization fires.

**Retrieval per query**
- The query is embedded once.
- That embedding is used to search two ChromaDB collections scoped to the current session: the single most relevant LTM summary, and the top-k document chunks (via hybrid search — see below).
- Everything is assembled into one prompt via a template.

**Graceful prompt degradation**
If the assembled prompt exceeds the token budget, it degrades in stages rather than failing outright: shrink each chunk to ~80 words → keep only the single most relevant chunk → keep only the last 2 STM turns → truncate the user's own message as a last resort.

**Session deletion**
Deleting a session clears its Redis STM immediately (so it stops "existing" for chat purposes right away), while a background task cleans up everything else: SQLite STM/LTM rows, ChromaDB vectors (both collections), the FTS5 keyword index, chat history, document chunks, and the document/session rows themselves.

| Data | Storage |
|---|---|
| Sessions, documents, chunk text, chat history, STM rows, LTM summaries | SQLite |
| Chunk embeddings, LTM summary embeddings | ChromaDB |
| Chunk keyword index | SQLite FTS5 |
| Short-term memory (hot path) | Redis |

## Retrieval pipeline

Two retrieval signals are combined for document chunk search:

- **Dense vector search** (ChromaDB, cosine similarity) — good at semantic/paraphrased matches, weaker at exact terms.
- **Keyword search** (SQLite FTS5, BM25 ranking) — good at exact terms, dates, names, numbers; weaker at paraphrase.

They're fused with **Reciprocal Rank Fusion (RRF)**: each chunk's score is the sum of `1 / (k + rank)` across every list it appears in (k=60). A chunk that both search methods agree on ranks highest of all.

This was built specifically to address a measured weakness: in evaluation, pure vector search often ranked the *correct* chunk 2nd or 3rd behind a semantically similar but wrong neighboring chunk — especially for date/fact/number questions where the literal term matters more than semantic similarity. See [Evaluation](#evaluation) below for the measured impact.

## Evaluation

Rather than relying on manual spot-checking, this project includes a two-part evaluation harness (`backend/eval/`) that measures retrieval and generation quality independently, against a hand-labeled 37-question test set built from a Wikipedia-length article.

### Retrieval evaluation

Measures whether the right chunk is actually found, independent of generation quality — using **Recall@k** (was the correct chunk in the top-k results?) and **MRR / Mean Reciprocal Rank** (how highly was it ranked, on average?).

| Metric | Vector-only | Hybrid (RRF) | Change |
|---|---|---|---|
| MRR | 0.650 | 0.801 | **+0.151** |
| Recall@1 | 48.6% | 70.3% | **+21.6 pts** |
| Recall@3 | 81.1% | 89.2% | **+8.1 pts** |
| Recall@5 | 83.8% | 94.6% | **+10.8 pts** |

Across the 37 test questions, hybrid search improved 13, regressed 0, and left 24 unchanged. The improvements concentrated almost entirely in exact-fact questions (dates, named events, statistics) — precisely the failure mode hybrid search targets, confirming the improvement is mechanistic rather than incidental.

### Generation evaluation

Even with correct retrieval, an LLM can still ignore its context or answer poorly. This is measured separately using an **LLM-as-judge** approach: for each question, the full RAG prompt (real retrieved context + the same prompt template the app uses) is sent to the model, and a second judge call scores the resulting answer 1-5 on:
- **Faithfulness** — is the answer grounded in the retrieved context, with no fabricated claims?
- **Relevance** — does it actually answer the question asked?

A crude word-overlap ratio against a human-written reference answer is also tracked as a cheap sanity check independent of the judge.

| Metric | Vector-only | Hybrid | Change |
|---|---|---|---|
| Avg faithfulness | 4.68 / 5 | 4.87 / 5 | +0.19 |
| Avg relevance | 4.89 / 5 | 5.00 / 5 | +0.11 |
| Word overlap (sanity) | 0.647 | 0.669 | +0.022 |

The two lowest-scoring answers under vector-only retrieval (both scoring 3/5 on relevance) disappeared entirely under hybrid retrieval, which reached a perfect 5.00 average relevance across all 37 questions.

**Known limitation, stated rather than hidden:** the judge is the same model (Llama 3.1 8B) used for generation. Self-judging can bias scores toward the model's own phrasing/style. A stronger or different judge model would reduce this — noted as a natural next step rather than treated as settled.

Both harnesses are re-runnable (`python -m eval.run_retrieval_eval --mode {vector,hybrid}` / `python -m eval.run_generation_eval --mode {vector,hybrid}`), and every run is saved as a timestamped JSON in `backend/eval/results/`, so results are reproducible and auditable rather than just reported numbers.

## Tech stack

`FastAPI` · `WebSockets` · `Streamlit` · `SQLAlchemy (async)` · `SQLite` + `FTS5` · `Redis` · `ChromaDB` · `Ollama (Llama 3.1 8B)` · `sentence-transformers (all-mpnet-base-v2)` · `LangChain text splitters` · `Docker Compose`

## Run it

### Option A — Docker

```
docker compose up --build
```
First boot pulls the `llama3.1:8b` model and downloads the embedding model — allow a few minutes. Then:
- Streamlit UI → http://localhost:8501
- FastAPI backend → http://localhost:8000

### Option B — Native (no Docker)

Requires [Ollama](https://ollama.com/download) and Redis installed locally.

```
ollama pull llama3.1:8b
```

**Backend** (PowerShell):
```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.backend.txt
$env:REDIS_URL = "redis://localhost:6379"
$env:OLLAMA_URL = "http://localhost:11434"
uvicorn app:app --reload --port 8000
```

**Frontend** (separate terminal):
```powershell
cd frontend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.frontend.txt
$env:BACKEND_URL = "http://localhost:8000"
streamlit run app.py
```

> Note: in Command Prompt (not PowerShell), use `set VAR=value` instead of `$env:VAR = "value"`.

### Running the evaluation harness

```powershell
cd backend
python -m eval.run_retrieval_eval --session-id <session_id> --mode hybrid
python -m eval.run_generation_eval --session-id <session_id> --mode hybrid
```

Get `<session_id>` from the Streamlit sidebar, `GET /sessions/`, or by querying `sessions` in `backend/db_files/chat.db`. Test questions live in `backend/eval/test_set.json`.

## Project structure

```
backend/
  app.py                    # FastAPI entrypoint
  api/                      # HTTP + WebSocket routes
    chat.py                 # WebSocket streaming endpoint
    documents.py            # Upload endpoint
    sessions.py             # List/delete/history endpoints
    schemas.py               # Pydantic response models
  services/
    chat_orchestrator.py    # Per-message RAG pipeline (retrieval -> prompt -> stream -> memory)
    memory_service.py       # STM/LTM read/write/summarize logic
    ingestion_service.py    # File parsing, chunking, embedding
    hybrid_search.py        # Keyword search + Reciprocal Rank Fusion
    llm_service.py          # Ollama client + local embedding model
    session_service.py      # Session CRUD + cascading delete
  db/
    db_models.py            # SQLAlchemy schema
    database.py             # Async engine + FTS5 setup
    vectordb.py              # ChromaDB wrapper
    redis_client.py           # Redis connection
  eval/
    test_set.json            # Hand-labeled Q&A test set
    run_retrieval_eval.py     # Recall@k / MRR harness
    run_generation_eval.py    # LLM-as-judge harness
    judge_prompt.txt          # Judge scoring instructions
    backfill_fts.py           # One-off FTS backfill for pre-existing chunks
    results/                  # Timestamped eval run outputs
  utils/
    rag_prompt.txt / summarize_prompt.txt / title_prompt.txt
    tokenizer.py, prompt_utils.py, logger.py
frontend/
  app.py                     # Streamlit UI
  utils/api_client.py        # REST client
  utils/websocket_client.py  # Streaming client
docker-compose.yml
init-ollama.sh
```

## Engineering decisions & fixes

This project was built by fully reimplementing an existing RAG architecture from scratch, then improving on several issues found during that process:

- **Redundant embedding calls**: the original retrieval step embedded the same query twice per turn (once per vector search target). Fixed to embed once and reuse — halves embedding cost per message.
- **Hybrid retrieval added**: dense vector search alone under-ranked exact-fact questions (dates, names, statistics). Adding a BM25 keyword index fused via Reciprocal Rank Fusion improved retrieval MRR by +23% with zero regressions (see [Evaluation](#evaluation)).
- **Silent failure on bad uploads**: unsupported or empty files used to create a "successful" session with zero retrievable content. Now raises a clear, user-facing error at the point of ingestion.
- **Type mismatch**: a chunk's foreign-key column was declared `String` while storing `Integer` IDs — worked by accident on SQLite, would break on Postgres. Fixed at the schema level.
- **Dead configuration**: `REDIS_URL` / `OLLAMA_URL` / `BACKEND_URL` were defined in `docker-compose.yml` but never actually read by the application code (hardcoded instead). Now properly wired through `os.getenv`, with defensive `.strip()` calls to tolerate shell quoting quirks.
- **Fixed-sleep startup race**: Ollama's init script waited a flat 10 seconds before pulling the model, which is fragile on both slow and fast machines. Now polls until the server is actually ready.
- **Evaluation-driven development**: rather than eyeballing chat responses, retrieval and generation quality are measured with reproducible, timestamped, re-runnable scripts — the basis for every "improved X by Y%" claim in this README.

## Possible extensions

- Cross-encoder reranking on top of the current hybrid retrieval
- Multi-document sessions (currently one document per session by design)
- A stronger/independent judge model for generation eval, to remove the self-judging bias noted above
- Real tokenizer (current token budgeting is a `len(text) // 4` heuristic)
- Postgres for multi-user scale
