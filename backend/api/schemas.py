from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ---- Document endpoints ----
class DocumentUploadResponse(BaseModel):
    session_id: str
    session_name: Optional[str]
    status: str


# ---- Session endpoints ----
class ListSessionsResponse(BaseModel):
    sessions: List[Dict[str, str]]


class DeleteSessionResponse(BaseModel):
    deleted: bool


class ChatHistoryResponse(BaseModel):
    history: List[Dict[str, Any]]
