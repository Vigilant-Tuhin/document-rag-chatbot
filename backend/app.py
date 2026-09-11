from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.chat import router as chat_router
from api.documents import router as documents_router
from api.sessions import router as sessions_router
from db.database import init_db
from utils.logger import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await init_db()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(sessions_router, prefix="/sessions")
app.include_router(documents_router)
app.include_router(chat_router, prefix="/chat")


@app.get("/")
def root():
    return {"status": "running"}
