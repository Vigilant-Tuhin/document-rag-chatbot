import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from services.chat_orchestrator import chat_orchestrator

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/{session_id}")
async def chat_ws(websocket: WebSocket, session_id: str, db: AsyncSession = Depends(get_db)):
    await websocket.accept()
    logger.info(f"[{session_id}] WebSocket connected")

    try:
        while True:
            msg = await websocket.receive_text()
            logger.info(f"[{session_id}] Received message: {msg}")

            async for token in chat_orchestrator.process_message(session_id, msg, db):
                await websocket.send_text(token)

            await websocket.send_text("[DONE]")

    except WebSocketDisconnect:
        # Normal client-initiated close (tab closed, page refreshed) — not an error.
        logger.info(f"[{session_id}] WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket error: {e}")
    finally:
        logger.info(f"[{session_id}] WebSocket closed")
