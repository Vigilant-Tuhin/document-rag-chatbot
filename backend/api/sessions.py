import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import ChatHistoryResponse, DeleteSessionResponse, ListSessionsResponse
from db.database import get_db
from services.session_service import session_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", summary="Fetch all sessions", response_model=ListSessionsResponse)
async def list_sessions(db: AsyncSession = Depends(get_db)):
    try:
        sessions = await session_service.list_sessions(db)
        return ListSessionsResponse(sessions=[{"id": s.id, "name": s.session_name or s.id[:8]} for s in sessions])
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail="Could not list sessions")


@router.delete("/{session_id}", summary="Delete session", response_model=DeleteSessionResponse)
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    try:
        await session_service.delete_session(session_id, db)
        return DeleteSessionResponse(deleted=True)
    except Exception as e:
        logger.error(f"[{session_id}] Failed to trigger delete: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


@router.get("/{session_id}/history", summary="Fetch session chat history", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str, db: AsyncSession = Depends(get_db)):
    try:
        history = await session_service.get_chat_history(session_id, db)
        return ChatHistoryResponse(history=history)
    except Exception as e:
        logger.error(f"[{session_id}] Failed to fetch history: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch history")
