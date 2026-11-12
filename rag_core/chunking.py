"""Normalización y fragmentación de texto con solapamiento."""
from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Elimina espacios redundantes sin perder los saltos entre párrafos."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\u00a0 ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """Separa un texto en ventanas de caracteres, preferentemente en límites de palabras.

    El siguiente fragmento repite aproximadamente `overlap` caracteres del anterior.
    Para entradas enormes sin espacios también progresa y nunca se cicla.
    """
    if chunk_size < 80:
        raise ValueError("chunk_size debe ser al menos 80")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap debe estar entre 0 y chunk_size - 1")

    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Buscar espacio o salto de línea próximo al final, sin producir trozos diminutos.
            boundary = max(text.rfind(" ", start + chunk_size // 2, end),
                           text.rfind("\n", start + chunk_size // 2, end))
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(0, end - overlap)
        # Evitar arrancar en mitad de una palabra cuando sea posible.
        if next_start > 0 and next_start < len(text) and not text[next_start - 1].isspace():
            next_space = text.find(" ", next_start, min(end, next_start + 55))
            if next_space >= 0:
                next_start = next_space + 1
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks
