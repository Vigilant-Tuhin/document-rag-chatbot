"""
One-off backfill: populates chunks_fts for DocumentChunk rows that were
ingested before the FTS5 keyword index existed. Safe to re-run (skips
chunks already present in chunks_fts).

Usage:
    python -m eval.backfill_fts
"""

import asyncio

from sqlalchemy import select, text

from db.database import AsyncSessionLocal, init_db
from db.db_models import DocumentChunk


async def backfill():
    await init_db()  # ensures chunks_fts exists if this is run before the app ever has

    async with AsyncSessionLocal() as db:
        existing = await db.execute(text("SELECT chunk_id FROM chunks_fts"))
        already_indexed = {row[0] for row in existing.fetchall()}

        result = await db.execute(select(DocumentChunk))
        all_chunks = result.scalars().all()

        to_insert = [c for c in all_chunks if c.id not in already_indexed]
        print(f"Found {len(all_chunks)} total chunks, {len(to_insert)} not yet in chunks_fts")

        for chunk in to_insert:
            await db.execute(
                text("INSERT INTO chunks_fts (chunk_id, session_id, text) VALUES (:chunk_id, :session_id, :text)"),
                {"chunk_id": chunk.id, "session_id": chunk.session_id, "text": chunk.text},
            )

        await db.commit()
        print(f"Backfilled {len(to_insert)} chunks into chunks_fts")


if __name__ == "__main__":
    asyncio.run(backfill())
