import logging
import uuid
import chromadb
import numpy as np

logger = logging.getLogger(__name__)


class VectorDB:
    def __init__(self, persist_path: str = "./db_files/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_path)

        self.chunk_collection = self.client.get_or_create_collection(
            name="chunks", metadata={"hnsw:space": "cosine"}
        )
        self.ltm_collection = self.client.get_or_create_collection(
            name="ltm_summaries", metadata={"hnsw:space": "cosine"}
        )

        # internal short names used throughout the codebase
        self.collections = {
            "chunks": self.chunk_collection,
            "ltm": self.ltm_collection,
        }

    def add_vector(self, collection_name: str, embedding, metadata: dict, vector_id: str | None = None):
        if vector_id is None:
            vector_id = str(uuid.uuid4())
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        col = self.collections[collection_name]
        col.add(embeddings=[embedding], metadatas=[metadata], ids=[vector_id])

    def search(self, collection_name: str, embedding, session_id: str, n: int = 3):
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        col = self.collections[collection_name]
        return col.query(
            query_embeddings=[embedding],
            where={"session_id": session_id},
            n_results=n,
        )

    def delete_session_embeddings(self, collection_name: str, session_id: str):
        col = self.collections[collection_name]
        col.delete(where={"session_id": session_id})


vectordb = VectorDB()
