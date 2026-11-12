"""Persistencia y búsqueda por similitud coseno con ChromaDB."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Passage:
    text: str
    source: str
    page: int
    chunk: int
    score: float


class ChromaStore:
    def __init__(self, folder: Path, collection_name: str):
        import chromadb

        folder.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(folder))
        self.collection_name = collection_name
        self.collection = self._get_collection()

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # Los embeddings los generamos nosotros.
        )

    def count(self) -> int:
        return self.collection.count()

    def replace(self, records: list[dict], embeddings: list[list[float]]) -> None:
        """Reconstruye el índice completo, evitando duplicados y archivos obsoletos."""
        if not records or len(records) != len(embeddings):
            raise ValueError("Se requiere una lista no vacía de registros y embeddings alineados")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_collection()
        for start in range(0, len(records), 100):
            rec_batch = records[start:start + 100]
            emb_batch = embeddings[start:start + 100]
            self.collection.upsert(
                ids=[row["id"] for row in rec_batch],
                documents=[row["text"] for row in rec_batch],
                metadatas=[{
                    "source": row["source"],
                    "page": row["page"],
                    "chunk": row["chunk"],
                } for row in rec_batch],
                embeddings=emb_batch,
            )

    def search(self, embedding: list[float], k: int = 5) -> list[Passage]:
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        passages: list[Passage] = []
        for text, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            passages.append(Passage(
                text=text, source=str(meta["source"]),
                page=int(meta["page"]), chunk=int(meta["chunk"]),
                score=max(-1.0, min(1.0, 1.0 - float(dist))),
            ))
        return passages
