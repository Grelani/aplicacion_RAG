"""Prueba el recuperador contra una lista de preguntas.

Ejecutar DESPUÉS de indexar:
    python -m scripts.evaluar

Esta prueba evalúa recuperación, no la fidelidad de generación del LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

from rag_core.config import CHROMA_DIR, DOCUMENTS_DIR, SETTINGS
from rag_core.embeddings import LocalEmbedder
from rag_core.indexing import index_is_current
from rag_core.pipeline import retrieve
from rag_core.vector_store import ChromaStore

CASES = Path(__file__).resolve().parents[1] / "tests" / "casos_demo.json"


def main() -> None:
    if not index_is_current(SETTINGS, DOCUMENTS_DIR):
        raise SystemExit("Índice inexistente o desactualizado. Abre la interfaz y pulsa Indexar.")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    embedder = LocalEmbedder(SETTINGS.embedding_model)
    store = ChromaStore(CHROMA_DIR, SETTINGS.collection_name)
    hits = 0
    for number, case in enumerate(cases, 1):
        candidates, accepted = retrieve(
            case["pregunta"], embedder, store,
            top_k=5, min_similarity=0.43,
        )
        found_sources = [p.source for p in accepted]
        if case.get("fuente_esperada"):
            passed = case["fuente_esperada"] in found_sources
        else:
            passed = len(accepted) == 0
        hits += int(passed)
        print(f"{'OK' if passed else 'REVISAR'} | Caso {number}: {case['pregunta']}")
        print(f"  Fuentes superando umbral: {found_sources or 'ninguna'}")
        if candidates:
            print(f"  Mayor similitud: {candidates[0].score:.3f}")
    print(f"\nResultado: {hits}/{len(cases)} casos de recuperación cumplieron la expectativa")
    print("Nota: ajusta el umbral o reformula preguntas según los resultados; no es una garantía de abstención.")


if __name__ == "__main__":
    main()
