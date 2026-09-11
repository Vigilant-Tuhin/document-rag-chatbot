import logging
from typing import List

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from db.vectordb import vectordb
from services.llm_service import llm_service
from services.memory_service import memory_service
from utils.prompt_utils import load_prompt
from utils.tokenizer import estimate_tokens

logger = logging.getLogger(__name__)


class ChatOrchestrator:

    def __init__(self, short_term_token_limit: int = 2000, response_token_limit: int = 4000, k_retrieval: int = 3):
        """
        short_term_token_limit: when to summarize short-term memory (kept here for reference;
                                 the actual threshold lives in memory_service.SHORT_TERM_TOKEN_LIMIT)
        response_token_limit:   max tokens to send to the LLM for the final prompt
        k_retrieval:            top-k document chunks retrieved per query
        """
        self.short_term_token_limit = short_term_token_limit
        self.response_token_limit = response_token_limit
        self.k = k_retrieval

    async def process_message(self, session_id: str, user_message: str, db: AsyncSession):
        logger.info(f"[{session_id}] Received message: {user_message}")

        # 1. Append user message to short-term memory
        await memory_service.add_short_term(session_id=session_id, role="user", content=user_message, db=db)

        # 2. Summarize STM into LTM if it has crossed the threshold
        await memory_service.maybe_summarize(session_id, db)

        # 3. Fetch short-term memory
        short_memory = await memory_service.get_short_term(session_id, db)

        # 4. Embed the query ONCE, reuse for both LTM recall and chunk retrieval
        #    (the original repo calls llm_service.embed(user_message) twice here — once per
        #    search — which is a redundant model inference per turn for identical input)
        query_emb = await llm_service.embed(user_message)
        if isinstance(query_emb, list):
            query_emb = np.array(query_emb, dtype="float32")
        logger.info(f"[{session_id}] Obtained query embedding (dim={query_emb.shape})")

        # 5. Selective LTM recall — single most relevant summary for this session
        ltm_results = vectordb.search(collection_name="ltm", embedding=query_emb, session_id=session_id, n=1)
        ltm_metas = ltm_results.get("metadatas", [[]])[0]
        long_memory = [ltm_metas[0]["summary"]] if ltm_metas else []

        # 6. Top-k document chunk retrieval
        chunk_results = vectordb.search(collection_name="chunks", embedding=query_emb, session_id=session_id, n=self.k)
        chunk_metas = chunk_results.get("metadatas", [[]])[0]
        doc_contexts = [md.get("text", "") for md in chunk_metas]

        logger.info(
            f"[{session_id}] Loaded short-term ({len(short_memory)} turns), "
            f"long-term ({len(long_memory)}), doc chunks ({len(doc_contexts)})"
        )

        # 7. Build the final prompt, trimming if it exceeds the token budget
        prompt = self._build_prompt(user_message, short_memory, long_memory, doc_contexts)
        used_tokens = estimate_tokens(prompt)
        logger.info(f"[{session_id}] Built prompt (approx tokens={used_tokens})")

        if used_tokens > self.response_token_limit:
            prompt = self._trim_prompt(prompt, long_memory, doc_contexts, short_memory, user_message)
            logger.info(f"[{session_id}] Prompt exceeded budget, trimmed to approx tokens={estimate_tokens(prompt)}")

        # 8. Stream the LLM response back to the caller
        logger.info(f"[{session_id}] Streaming prompt to LLM")
        full_response = ""
        async for token in llm_service.chat_stream(prompt):
            full_response += token
            yield token

        logger.info(f"[{session_id}] LLM completed (len={len(full_response)} chars)")

        # 9. Append the completed assistant response to short-term memory
        await memory_service.add_short_term(session_id=session_id, role="assistant", content=full_response, db=db)

    def _build_prompt(self, user_message: str, short_memory: List[dict], long_memory: List[str], doc_contexts: List[str]) -> str:
        long_text = "\n".join(f"- {s}" for s in long_memory) if long_memory else "No long-term memory."
        doc_text = (
            "\n\n".join(f"CHUNK {i + 1}:\n{c}" for i, c in enumerate(doc_contexts))
            if doc_contexts
            else "No document context found."
        )
        short_text = "\n".join(f"{m['role']}: {m['content']}" for m in short_memory) if short_memory else "No short-term memory."

        template = load_prompt("rag_prompt.txt")
        prompt = template.format(
            long_text=long_text,
            doc_text=doc_text,
            doc_count=len(doc_contexts),
            short_text=short_text,
            user_message=user_message,
        )
        return prompt.strip()

    def _trim_prompt(self, prompt: str, long_memory: List[str], doc_contexts: List[str], short_memory: List[dict], user_message: str) -> str:
        """Graceful degradation ladder when the prompt is over budget, each step trying a
        progressively more aggressive cut until the token budget is met."""
        budget = self.response_token_limit
        if estimate_tokens(prompt) <= budget:
            return prompt

        # Step 1: shorten each document chunk to its first 80 words
        short_docs = []
        for c in doc_contexts:
            words = c.split()
            short_docs.append(" ".join(words[:80]) + (" ..." if len(words) > 80 else ""))
        prompt = self._build_prompt(user_message, short_memory, long_memory, short_docs)
        if estimate_tokens(prompt) <= budget:
            return prompt

        # Step 2: keep only the single most relevant chunk
        top1_docs = short_docs[:1]
        prompt = self._build_prompt(user_message, short_memory, long_memory, top1_docs)
        if estimate_tokens(prompt) <= budget:
            return prompt

        # Step 3: keep only the last 2 short-term turns
        last_short = short_memory[-2:] if len(short_memory) > 2 else short_memory
        prompt = self._build_prompt(user_message, last_short, long_memory, top1_docs)
        if estimate_tokens(prompt) <= budget:
            return prompt

        # Step 4: last resort — truncate the user's own message
        user_words = user_message.split()
        half_user = " ".join(user_words[:150]) + "..."
        return self._build_prompt(half_user, last_short, long_memory, top1_docs)


chat_orchestrator = ChatOrchestrator()
