"""Configuración centralizada del proyecto."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT_DIR / "documentos"
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
MANIFEST_PATH = DATA_DIR / "indice.json"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class Settings:
    embedding_model: str = os.getenv(
        "DOCUCHAT_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    llm_model: str = os.getenv("DOCUCHAT_LLM_MODEL", "llama3.2")
    ollama_url: str = os.getenv("DOCUCHAT_OLLAMA_URL", "http://localhost:11434")
    chunk_size: int = 600
    chunk_overlap: int = 100
    collection_name: str = "documentos_rag"
    max_upload_mb: int = 15


SETTINGS = Settings()
