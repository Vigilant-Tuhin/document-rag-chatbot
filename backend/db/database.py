import logging
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from db.db_models import Base

logger = logging.getLogger(__name__)

DB_DIR = "./db_files"
os.makedirs(DB_DIR, exist_ok=True)

DATABASE_URL = f"sqlite+aiosqlite:///{DB_DIR}/chat.db"

engine = create_async_engine(DATABASE_URL, echo=False)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    """FastAPI dependency: yields a request-scoped DB session."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    logger.info("Initialising SQLite database")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
