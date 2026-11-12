"""Organiza la ingesta: documentos -> chunks -> embeddings -> ChromaDB."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag_core.chunking import split_text
from rag_core.config import MANIFEST_PATH, Settings
from rag_core.loaders import list_documents, read_document


@dataclass(frozen=True)
class IndexResult:
    documents: int
    chunks: int
    errors: list[str]


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def document_fingerprints(folder: Path) -> dict[str, str]:
    return {p.name: file_digest(p) for p in list_documents(folder)}


def load_manifest(path: Path = MANIFEST_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def index_is_current(settings: Settings, documents_dir: Path, manifest_path: Path = MANIFEST_PATH) -> bool:
    manifest = load_manifest(manifest_path)
    return bool(
        manifest
        and manifest.get("embedding_model") == settings.embedding_model
        and manifest.get("chunk_size") == settings.chunk_size
        and manifest.get("chunk_overlap") == settings.chunk_overlap
        and manifest.get("fingerprints") == document_fingerprints(documents_dir)
    )


def build_index(
    documents_dir: Path,
    embedder,
    store,
    settings: Settings,
    manifest_path: Path = MANIFEST_PATH,
) -> IndexResult:
    paths = list_documents(documents_dir)
    if not paths:
        raise ValueError("No hay archivos PDF, TXT o MD en la carpeta documentos")

    records: list[dict] = []
    errors: list[str] = []
    successful = 0
    for path in paths:
        try:
            pages = read_document(path)
            if not pages:
                raise ValueError("No se pudo extraer texto (¿PDF escaneado?)")
            chunks_for_file = []
            for page in pages:
                for i, piece in enumerate(split_text(
                    page.text, chunk_size=settings.chunk_size,
                    overlap=settings.chunk_overlap,
                ), start=1):
                    unique = f"{path.name}|{page.page}|{i}|{piece}"
                    record_id = hashlib.sha256(unique.encode("utf-8")).hexdigest()
                    chunks_for_file.append({
                        "id": record_id, "source": path.name, "page": page.page,
                        "chunk": i, "text": piece,
                    })
            if not chunks_for_file:
                raise ValueError("El documento está vacío")
            records.extend(chunks_for_file)
            successful += 1
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if not records:
        raise ValueError("No se pudo obtener texto de ningún documento. " + "; ".join(errors))

    embeddings = embedder.encode([record["text"] for record in records])
    if len(embeddings) != len(records):
        raise RuntimeError("El modelo de embeddings produjo un número incorrecto de vectores")

    # Si falla la reconstrucción, el manifiesto no debe quedar marcado como válido.
    manifest_path.unlink(missing_ok=True)
    store.replace(records, embeddings)

    # Solo archivos correctamente indexados: si hubo errores, reindexar tras corregirlos.
    manifest = {
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "fingerprints": document_fingerprints(documents_dir) if not errors else {},
        "documents": successful,
        "chunks": len(records),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(manifest_path)
    return IndexResult(documents=successful, chunks=len(records), errors=errors)
