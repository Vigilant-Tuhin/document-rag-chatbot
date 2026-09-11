"""
Retrieval evaluation harness.

Measures whether the vector search step (chat_orchestrator's retrieval logic)
actually surfaces the right document chunks for a set of known questions —
independent of the LLM generation step, so results aren't muddied by
generation quality.

Usage:
    python -m eval.run_retrieval_eval --session-id <session_id>

The session_id is whatever session your test document was uploaded into
(visible in the Streamlit sidebar, or via GET /sessions/). Chunk IDs in
test_set.json must correspond to document_chunks.id rows for that session —
you can browse backend/db_files/chat.db with any SQLite tool to find them
after uploading your test document.

Metrics:
    Recall@k — for each question, did any expected chunk appear in the
               top-k retrieved results? Averaged across all questions.
    MRR      — Mean Reciprocal Rank: for each question, 1/rank of the first
               expected chunk found (0 if none found in the searched range).
               Rewards ranking the right chunk higher, not just present.
"""

import argparse
import asyncio
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from db.database import AsyncSessionLocal
from db.vectordb import vectordb
from services.hybrid_search import hybrid_retrieve
from services.llm_service import llm_service

EVAL_DIR = Path(__file__).resolve().parent
TEST_SET_PATH = EVAL_DIR / "test_set.json"
RESULTS_DIR = EVAL_DIR / "results"
RECALL_K_VALUES = [1, 3, 5]
MAX_K = max(RECALL_K_VALUES)


def load_test_set():
    with open(TEST_SET_PATH) as f:
        data = json.load(f)
    # drop template/comment-only entries
    return [item for item in data if "_comment" not in item]


async def evaluate_question(question: str, expected_chunk_ids: list[int], session_id: str, mode: str, db) -> dict:
    embedding = await llm_service.embed(question)

    if mode == "hybrid":
        fused = await hybrid_retrieve(db, session_id, question, embedding, n=MAX_K)
        retrieved_chunk_ids = [c["chunk_id"] for c in fused]
    else:
        results = vectordb.search(collection_name="chunks", embedding=embedding, session_id=session_id, n=MAX_K)
        metadatas = results.get("metadatas", [[]])[0]
        retrieved_chunk_ids = [md.get("chunk_id") for md in metadatas]

    expected_set = set(expected_chunk_ids)

    # Recall@k for each k we care about
    recall_at_k = {}
    for k in RECALL_K_VALUES:
        top_k_ids = set(retrieved_chunk_ids[:k])
        recall_at_k[k] = 1.0 if (top_k_ids & expected_set) else 0.0

    # Reciprocal rank of the first correct hit anywhere in the retrieved list
    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "question": question,
        "expected_chunk_ids": expected_chunk_ids,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "recall_at_k": recall_at_k,
        "reciprocal_rank": reciprocal_rank,
    }


async def run(session_id: str, mode: str):
    test_set = load_test_set()
    if not test_set:
        print("No real entries in test_set.json — remove the template's '_comment' entry and add your own.")
        return

    print(f"Running retrieval eval ({mode} mode): {len(test_set)} questions against session {session_id}\n")

    per_question_results = []
    async with AsyncSessionLocal() as db:
        for item in test_set:
            result = await evaluate_question(item["question"], item["expected_chunk_ids"], session_id, mode, db)
            per_question_results.append(result)

            hit_marker = "✓" if result["reciprocal_rank"] > 0 else "✗"
            print(f"  {hit_marker} [{result['reciprocal_rank']:.2f} RR] {result['question'][:70]}")

    # Aggregate
    summary = {
        "session_id": session_id,
        "mode": mode,
        "num_questions": len(test_set),
        "mrr": round(statistics.mean(r["reciprocal_rank"] for r in per_question_results), 4),
        "recall_at_k": {
            k: round(statistics.mean(r["recall_at_k"][k] for r in per_question_results), 4)
            for k in RECALL_K_VALUES
        },
    }

    print("\n--- Summary ---")
    print(f"Mode: {mode}")
    print(f"MRR: {summary['mrr']}")
    for k in RECALL_K_VALUES:
        print(f"Recall@{k}: {summary['recall_at_k'][k]}")

    # Save full results (per-question + summary) for later before/after comparison
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"retrieval_{mode}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_question": per_question_results}, f, indent=2)

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True, help="Session ID the test document was uploaded into")
    parser.add_argument("--mode", choices=["vector", "hybrid"], default="vector", help="Retrieval mode to evaluate")
    args = parser.parse_args()

    asyncio.run(run(args.session_id, args.mode))
