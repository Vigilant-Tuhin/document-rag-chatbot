import logging
import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.vectordb import vectordb

logger = logging.getLogger(__name__)

RRF_K = 60  # standard smoothing constant from the original RRF paper (Cormack et al.)


def _sanitize_fts_query(query: str) -> str:
    """
    Turn a free-text question into a safe FTS5 MATCH expression.

    FTS5 MATCH syntax treats characters like quotes/colons/parens specially,
    so raw user/LLM-adjacent text can throw a syntax error. Extracting plain
    alphanumeric tokens and OR-ing them together sidesteps that entirely, and
    OR (rather than the implicit AND between bare tokens) favors recall —
    appropriate here since this is one signal in a fused ranking, not the
    only one.
    """
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)


async def keyword_search(db: AsyncSession, session_id: str, query: str, n: int = 5) -> list[dict]:
    """Returns up to n chunks ranked by BM25, best match first."""
    fts_query = _sanitize_fts_query(query)

    result = await db.execute(
        text(
            """
            SELECT chunk_id, text
            FROM chunks_fts
            WHERE chunks_fts MATCH :fts_query AND session_id = :session_id
            ORDER BY rank
            LIMIT :n
            """
        ),
        {"fts_query": fts_query, "session_id": session_id, "n": n},
    )
    rows = result.fetchall()
    return [{"chunk_id": row[0], "text": row[1]} for row in rows]


def reciprocal_rank_fusion(*ranked_lists: list[dict], k: int = RRF_K) -> list[dict]:
    """
    Merges multiple ranked lists (each a list of dicts with a "chunk_id" key,
    best match first) into one fused ranking.

    RRF score for a chunk = sum over every list it appears in of 1 / (k + rank).
    A chunk ranked highly in either list scores well; a chunk appearing in
    BOTH lists (agreement between keyword and vector search) scores best of
    all — which is exactly the signal we want for the "correct chunk ranked
    2nd instead of 1st" cases seen in the vector-only baseline eval.
    """
    scores: dict[int, float] = {}
    chunk_text_by_id: dict[int, str] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_text_by_id.setdefault(chunk_id, item.get("text", ""))

    fused = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [{"chunk_id": chunk_id, "text": chunk_text_by_id[chunk_id], "rrf_score": score} for chunk_id, score in fused]


async def hybrid_retrieve(db: AsyncSession, session_id: str, query: str, query_embedding, n: int = 5) -> list[dict]:
    """Combines vector search + keyword search results via RRF, returns top n fused chunks."""
    vector_results = vectordb.search(collection_name="chunks", embedding=query_embedding, session_id=session_id, n=n)
    vector_metas = vector_results.get("metadatas", [[]])[0]
    vector_ranked = [{"chunk_id": md.get("chunk_id"), "text": md.get("text", "")} for md in vector_metas]

    keyword_ranked = await keyword_search(db, session_id, query, n=n)

    fused = reciprocal_rank_fusion(vector_ranked, keyword_ranked)
    return fused[:n]
