import json
import logging

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from db.db_models import SessionLongTermMemory, SessionShortTermMemory, SessionChatHistory
from db.redis_client import redis_client
from db.vectordb import vectordb
from services.llm_service import llm_service
from utils.prompt_utils import load_prompt
from utils.tokenizer import estimate_tokens

logger = logging.getLogger(__name__)

SHORT_TERM_TOKEN_LIMIT = 2000


class MemoryService:

    SHORT_KEY_TEMPLATE = "session:{session_id}:short_memory"

    # ---- Short-term memory (Redis, mirrored to SQLite) ----

    def _redis_key(self, session_id: str) -> str:
        return self.SHORT_KEY_TEMPLATE.format(session_id=session_id)

    def add_short_term_to_redis(self, session_id: str, role: str, content: str):
        key = self._redis_key(session_id)
        redis_client.rpush(key, json.dumps({"role": role, "content": content}))

    async def add_short_term(self, session_id: str, role: str, content: str, db: AsyncSession):
        """Write a turn to both Redis (fast path) and SQLite (durable STM + permanent chat history)."""
        self.add_short_term_to_redis(session_id, role, content)
        logger.info(f"[{session_id}] Added short-term memory message ({role}) to redis")

        db.add(SessionShortTermMemory(session_id=session_id, role=role, content=content))
        db.add(SessionChatHistory(session_id=session_id, role=role, content=content))
        await db.commit()
        logger.info(f"[{session_id}] Persisted short-term memory + chat history message ({role})")

    async def restore_short_term(self, session_id: str, db: AsyncSession):
        """Rebuild Redis STM from SQLite. Used after restart or when switching to a session not in Redis."""
        key = self._redis_key(session_id)
        logger.info(f"Restoring short-term memory into Redis for session id: {session_id}")

        redis_client.delete(key)

        result = await db.execute(
            select(SessionShortTermMemory).where(SessionShortTermMemory.session_id == session_id)
        )
        rows = result.scalars().all()
        for row in rows:
            redis_client.rpush(key, json.dumps({"role": row.role, "content": row.content}))

        logger.info(f"Restored {len(rows)} messages into Redis for session id: {session_id}")

    async def get_short_term(self, session_id: str, db: AsyncSession):
        key = self._redis_key(session_id)
        items = redis_client.lrange(key, 0, -1)

        if not items:
            logger.info(f"[{session_id}] Redis STM empty, restoring from SQLite")
            await self.restore_short_term(session_id, db)
            items = redis_client.lrange(key, 0, -1)

        return [json.loads(x) for x in items]

    def keep_last_n_short_term(self, session_id: str, n: int = 4):
        redis_client.ltrim(self._redis_key(session_id), -n, -1)

    def clear_redis_short_term(self, session_id: str):
        redis_client.delete(self._redis_key(session_id))
        logger.info(f"[{session_id}] Cleared Redis short-term memory")

    async def trim_short_term_sqlite(self, session_id: str, db: AsyncSession, n: int = 3):
        """Keep only the most recent `n` SQLite STM rows (called right after a summarization pass)."""
        result = await db.execute(
            select(SessionShortTermMemory.id)
            .where(SessionShortTermMemory.session_id == session_id)
            .order_by(SessionShortTermMemory.id.desc())
            .limit(n)
        )
        keep_ids = [row[0] for row in result.fetchall()]
        if not keep_ids:
            return

        await db.execute(
            delete(SessionShortTermMemory)
            .where(SessionShortTermMemory.session_id == session_id)
            .where(SessionShortTermMemory.id.not_in(keep_ids))
        )
        await db.commit()

    async def delete_all_session_memory(self, session_id: str, db: AsyncSession):
        logger.info(f"[{session_id}] Deleting all SQLite STM + LTM rows")
        await db.execute(delete(SessionShortTermMemory).where(SessionShortTermMemory.session_id == session_id))
        await db.execute(delete(SessionLongTermMemory).where(SessionLongTermMemory.session_id == session_id))
        await db.commit()

    # ---- Long-term memory (SQLite summaries + ChromaDB embeddings) ----

    async def append_long_term(self, session_id: str, summary: str, db: AsyncSession):
        mem = SessionLongTermMemory(session_id=session_id, summary=summary)
        db.add(mem)
        await db.commit()

        emb = await llm_service.embed(summary)
        vectordb.add_vector(
            collection_name="ltm",
            embedding=emb,
            metadata={"session_id": session_id, "summary": summary},
            vector_id=None,
        )

    async def get_long_term(self, session_id: str, db: AsyncSession):
        result = await db.execute(select(SessionLongTermMemory).where(SessionLongTermMemory.session_id == session_id))
        return result.scalars().all()

    # ---- Summarization: STM -> LTM once STM crosses a token threshold ----

    async def maybe_summarize(self, session_id: str, db: AsyncSession):
        short_memory = await self.get_short_term(session_id, db)
        if not short_memory:
            return

        text = " ".join(x["content"] for x in short_memory)
        tokens = estimate_tokens(text)
        if tokens < SHORT_TERM_TOKEN_LIMIT:
            return

        logger.info(f"[{session_id}] STM ~{tokens} tokens >= {SHORT_TERM_TOKEN_LIMIT}, summarizing into LTM")
        template = load_prompt("summarize_prompt.txt")
        prompt = template.format(text=text)
        summary = await llm_service.summarize(prompt)

        await self.append_long_term(session_id, summary, db)

        # Trim STM down to the last few turns so recent context isn't lost mid-conversation
        self.keep_last_n_short_term(session_id, n=4)
        await self.trim_short_term_sqlite(session_id, db, n=3)

        logger.info(f"[{session_id}] Summarized STM into LTM and trimmed STM")


memory_service = MemoryService()
