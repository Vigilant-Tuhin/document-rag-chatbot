import os

import requests

# docker-compose sets BACKEND_URL=http://backend:8000; same env-var pattern as the backend services
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")


def upload_file(file):
    files = {"file": (file.name, file, file.type)}
    r = requests.post(f"{BACKEND_URL}/upload", files=files)
    if not r.ok:
        # e.g. 422 for an unsupported/empty file — surface the backend's message
        # instead of letting a KeyError on a missing "session_id" crash the UI
        detail = r.json().get("detail", "Upload failed") if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise RuntimeError(detail)
    return r.json()


def list_sessions():
    r = requests.get(f"{BACKEND_URL}/sessions/")
    r.raise_for_status()
    return r.json()


def delete_session(session_id: str):
    r = requests.delete(f"{BACKEND_URL}/sessions/{session_id}")
    r.raise_for_status()
    return r.json()


def get_history(session_id: str):
    r = requests.get(f"{BACKEND_URL}/sessions/{session_id}/history")
    r.raise_for_status()
    return r.json()
