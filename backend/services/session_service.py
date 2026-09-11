import asyncio
import logging
import uuid

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import AsyncSessionLocal
from db.db_models import Document, DocumentChunk, Session, SessionChatHistory
from db.vectordb import vectordb
from services.memory_service import memory_service

logger = logging.getLogger(__name__)


class SessionService:

    async def create_session(self, db: AsyncSession) -> str:
        session_id = str(uuid.uuid4())
        logger.info(f"Creating session {session_id}")

        db.add(Session(id=session_id, session_name=None))
        await db.commit()
        return session_id

    async def list_sessions(self, db: AsyncSession):
        result = await db.execute(select(Session).order_by(Session.created_at.desc()))
        return result.scalars().all()

    async def get_chat_history(self, session_id: str, db: AsyncSession):
        result = await db.execute(
            select(SessionChatHistory)
            .where(SessionChatHistory.session_id == session_id)
            .order_by(SessionChatHistory.created_at.asc())
        )
        rows = result.scalars().all()
        return [{"role": r.role, "content": r.content, "timestamp": r.created_at.isoformat()} for r in rows]

    async def delete_session(self, session_id: str, db: AsyncSession):
        logger.info(f"[{session_id}] Delete triggered")

        # Instant: clear the hot-path Redis STM so the session stops "existing" for chat purposes immediately
        memory_service.clear_redis_short_term(session_id)

        # Slow: everything else happens in the background so the API can respond right away
        asyncio.create_task(self._background_cleanup(session_id))

    async def _background_cleanup(self, session_id: str):
        """
        Deletes, in order:
        - SQLite SessionShortTermMemory / SessionLongTermMemory
        - ChromaDB embeddings (chunks + ltm collections)
        - SQLite SessionChatHistory
        - SQLite DocumentChunk rows
        - SQLite Document rows
        - SQLite Session row
        """
        logger.info(f"[{session_id}] Background cleanup started")

        # Own DB session since the request-scoped one may already be closed by the time this runs
        async with AsyncSessionLocal() as db:
            try:
                await memory_service.delete_all_session_memory(session_id, db)

                vectordb.delete_session_embeddings("chunks", session_id)
                vectordb.delete_session_embeddings("ltm", session_id)

                await db.execute(delete(SessionChatHistory).where(SessionChatHistory.session_id == session_id))
                await db.execute(text("DELETE FROM chunks_fts WHERE session_id = :session_id"), {"session_id": session_id})
                await db.execute(delete(DocumentChunk).where(DocumentChunk.session_id == session_id))
                await db.execute(delete(Document).where(Document.session_id == session_id))
                await db.execute(delete(Session).where(Session.id == session_id))

                await db.commit()
                logger.info(f"[{session_id}] Background cleanup complete")
            except Exception as e:
                logger.error(f"[{session_id}] Cleanup failed: {e}")


session_service = SessionService()
