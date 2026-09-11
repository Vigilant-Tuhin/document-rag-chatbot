"""
Generation evaluation harness — LLM-as-judge.

Unlike run_retrieval_eval.py (which only checks whether the right chunks were
found), this measures the final answer quality: given what was actually
retrieved, did the LLM produce a faithful, relevant answer?

Uses the same Ollama model as the app itself as the judge. Worth being
upfront about the limitation this implies: a model judging its own outputs
can be biased toward its own phrasing/style. A stronger/different judge model
would reduce that bias — noted here rather than silently ignored.

Usage:
    python -m eval.run_generation_eval --session-id <session_id> --mode hybrid
"""

import argparse
import asyncio
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from db.database import AsyncSessionLocal
from db.vectordb import vectordb
from services.chat_orchestrator import chat_orchestrator
from services.hybrid_search import hybrid_retrieve
from services.llm_service import llm_service

EVAL_DIR = Path(__file__).resolve().parent
TEST_SET_PATH = EVAL_DIR / "test_set.json"
JUDGE_PROMPT_PATH = EVAL_DIR / "judge_prompt.txt"
RESULTS_DIR = EVAL_DIR / "results"


def load_test_set():
    with open(TEST_SET_PATH) as f:
        data = json.load(f)
    return [item for item in data if "_comment" not in item]


def load_judge_prompt() -> str:
    return JUDGE_PROMPT_PATH.read_text()


def word_overlap_ratio(generated: str, expected: str) -> float:
    """Crude sanity metric alongside the LLM judge: fraction of expected-answer
    words that also appear in the generated answer. Cheap, no LLM call, useful
    for catching totally-off-base answers even if the judge call fails."""
    gen_words = set(re.findall(r"[a-z0-9]+", generated.lower()))
    exp_words = set(re.findall(r"[a-z0-9]+", expected.lower()))
    if not exp_words:
        return 0.0
    return len(gen_words & exp_words) / len(exp_words)


async def retrieve_context(db, session_id: str, question: str, mode: str) -> list[str]:
    embedding = await llm_service.embed(question)
    if mode == "hybrid":
        fused = await hybrid_retrieve(db, session_id, question, embedding, n=3)
        return [c["text"] for c in fused]
    results = vectordb.search(collection_name="chunks", embedding=embedding, session_id=session_id, n=3)
    metadatas = results.get("metadatas", [[]])[0]
    return [md.get("text", "") for md in metadatas]


async def judge_answer(question: str, context: list[str], expected_answer: str, generated_answer: str) -> dict:
    template = load_judge_prompt()
    prompt = template.format(
        question=question,
        context="\n\n".join(context) if context else "(no context retrieved)",
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )

    raw = await llm_service.chat(prompt)
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
        return {
            "faithfulness": int(parsed.get("faithfulness", 0)),
            "relevance": int(parsed.get("relevance", 0)),
            "reasoning": parsed.get("reasoning", ""),
            "parse_ok": True,
        }
    except (json.JSONDecodeError, ValueError):
        # Judge didn't return clean JSON — don't crash the whole eval run over one bad parse
        return {"faithfulness": 0, "relevance": 0, "reasoning": f"JUDGE PARSE FAILED: {raw[:200]}", "parse_ok": False}


async def evaluate_question(item: dict, session_id: str, mode: str, db) -> dict:
    question = item["question"]
    expected_answer = item["expected_answer"]

    context = await retrieve_context(db, session_id, question, mode)
    prompt = chat_orchestrator._build_prompt(question, short_memory=[], long_memory=[], doc_contexts=context)
    generated_answer = await llm_service.chat(prompt)

    scores = await judge_answer(question, context, expected_answer, generated_answer)
    overlap = word_overlap_ratio(generated_answer, expected_answer)

    return {
        "question": question,
        "generated_answer": generated_answer,
        "expected_answer": expected_answer,
        "word_overlap": round(overlap, 3),
        **scores,
    }


async def run(session_id: str, mode: str):
    test_set = load_test_set()
    if not test_set:
        print("No real entries in test_set.json.")
        return

    print(f"Running generation eval ({mode} mode): {len(test_set)} questions against session {session_id}")
    print("(this calls the LLM twice per question — generation + judging — so it's slower than the retrieval eval)\n")

    per_question_results = []
    async with AsyncSessionLocal() as db:
        for i, item in enumerate(test_set, start=1):
            result = await evaluate_question(item, session_id, mode, db)
            per_question_results.append(result)
            print(f"  [{i}/{len(test_set)}] faithfulness={result['faithfulness']} relevance={result['relevance']} | {item['question'][:60]}")

    parsed_ok = [r for r in per_question_results if r["parse_ok"]]
    summary = {
        "session_id": session_id,
        "mode": mode,
        "num_questions": len(test_set),
        "num_judge_parse_failures": len(test_set) - len(parsed_ok),
        "avg_faithfulness": round(statistics.mean(r["faithfulness"] for r in parsed_ok), 3) if parsed_ok else None,
        "avg_relevance": round(statistics.mean(r["relevance"] for r in parsed_ok), 3) if parsed_ok else None,
        "avg_word_overlap": round(statistics.mean(r["word_overlap"] for r in per_question_results), 3),
    }

    print("\n--- Summary ---")
    print(f"Mode: {mode}")
    print(f"Avg faithfulness: {summary['avg_faithfulness']}/5")
    print(f"Avg relevance: {summary['avg_relevance']}/5")
    print(f"Avg word overlap (sanity metric): {summary['avg_word_overlap']}")
    if summary["num_judge_parse_failures"]:
        print(f"WARNING: {summary['num_judge_parse_failures']} judge responses failed to parse as JSON")

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"generation_{mode}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_question": per_question_results}, f, indent=2)

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--mode", choices=["vector", "hybrid"], default="hybrid")
    args = parser.parse_args()

    asyncio.run(run(args.session_id, args.mode))
