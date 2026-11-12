"""Representaciones numéricas de las preguntas y los documentos."""
from __future__ import annotations


class LocalEmbedder:
    def __init__(self, model_name: str):
        # Importación diferida: las pruebas de lectura no descargan modelos.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts, batch_size=16, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return vectors.tolist()
