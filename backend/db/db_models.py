from sqlalchemy import Column, String, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


def utcnow():
    """Callable default so each row gets its own timestamp, not one frozen at import time."""
    return datetime.now(timezone.utc)


class Session(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)
    session_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    filename = Column(String)
    content_type = Column(String)
    created_at = Column(DateTime, default=utcnow)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, index=True)
    session_id = Column(String, index=True)
    chunk_index = Column(Integer)
    text = Column(Text)


class SessionShortTermMemory(Base):
    __tablename__ = "session_short_term_memory"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    timestamp = Column(DateTime, default=utcnow)


class SessionLongTermMemory(Base):
    __tablename__ = "session_long_term_memory"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    summary = Column(Text)
    created_at = Column(DateTime, default=utcnow)


class SessionChatHistory(Base):
    __tablename__ = "session_chat_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=utcnow)
