"""Lectura de PDF, TXT y Markdown, conservando la página del PDF."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_core.chunking import normalize_text
from rag_core.config import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class TextPage:
    text: str
    page: int  # 1-indexado en PDF; 0 para TXT/MD.


def list_documents(folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda p: p.name.casefold(),
    )


def read_document(path: Path) -> list[TextPage]:
    if not path.is_file():
        raise FileNotFoundError(path)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Formato no admitido: {extension}")

    if extension == ".pdf":
        import pymupdf

        result: list[TextPage] = []
        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                content = normalize_text(page.get_text("text", sort=True))
                if content:
                    result.append(TextPage(content, page_number))
        return result

    # UTF-8 es el formato recomendado; cp1252 permite textos creados en Windows.
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = path.read_text(encoding="cp1252")
    content = normalize_text(content)
    return [TextPage(content, 0)] if content else []
