import json
import logging
import os

import httpx
from sentence_transformers import SentenceTransformer

from utils.prompt_utils import load_prompt

logger = logging.getLogger(__name__)

# docker-compose sets OLLAMA_URL=http://ollama:11434; same fallback pattern as redis_client.py
# OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
# LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b").strip()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class LLMService:

    def __init__(self):
        # Embeddings are computed locally (not via Ollama) using sentence-transformers.
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.llm_model = LLM_MODEL
        self.http_client = httpx.AsyncClient(timeout=120.0)

    async def embed(self, text: str):
        vector = self.embedding_model.encode(text, show_progress_bar=False)
        return vector.astype("float32")

    async def generate_session_title(self, text: str) -> str:
        template = load_prompt("title_prompt.txt")
        title_prompt = template.format(text=text)
        raw_title = await self.chat(title_prompt)

        title = raw_title.strip().replace('"', "").replace("Title:", "").strip()
        if not title:
            title = "Untitled Document"
        return title

    async def chat_stream(self, prompt: str):
        """Stream tokens from Ollama one at a time as they're generated."""
        async with self.http_client.stream(
            "POST",
            f"{OLLAMA_URL}/api/generate",
            json={"model": self.llm_model, "prompt": prompt, "stream": True},
        ) as response:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "response" in data:
                    yield data["response"]

    async def chat(self, prompt: str) -> str:
        """Non-streaming call, used for summarization and title generation."""
        response = await self.http_client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": self.llm_model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json()["response"]

    async def summarize(self, prompt: str) -> str:
        return await self.chat(prompt)


llm_service = LLMService()
