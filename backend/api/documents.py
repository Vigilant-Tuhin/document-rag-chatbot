import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import DocumentUploadResponse
from db.database import get_db
from db.db_models import Session
from services.ingestion_service import ingestion_service
from services.session_service import session_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile, db: AsyncSession = Depends(get_db)):
    logger.info("Document upload received, creating session")

    session_id = await session_service.create_session(db)
    logger.info(f"[{session_id}] Session created")

    try:
        await ingestion_service.ingest(file, db, session_id=session_id)
    except ValueError as e:
        # Bad/unsupported/empty file — surface a clean 4xx instead of a raw 500.
        # The orphaned Session row is harmless; it just has no documents/chunks attached.
        logger.warning(f"[{session_id}] Ingestion rejected: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    logger.info(f"[{session_id}] Document ingested")

    session_row = await db.get(Session, session_id)
    return DocumentUploadResponse(
        session_id=session_id,
        session_name=session_row.session_name,
        status="ingested",
    )
