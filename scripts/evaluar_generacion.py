"""Genera evidencia de una evaluación RAG de extremo a extremo.

Requiere el índice y el modelo LLM local. Guarda data/resultados_evaluacion.csv.
Se debe revisar manualmente la fidelidad de cada respuesta.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from rag_core.config import CHROMA_DIR, DATA_DIR, DOCUMENTS_DIR, SETTINGS
from rag_core.embeddings import LocalEmbedder
from rag_core.indexing import index_is_current
from rag_core.pipeline import answer_question
from rag_core.vector_store import ChromaStore

INPUT = Path(__file__).resolve().parents[1] / "tests" / "casos_demo.json"
OUTPUT = DATA_DIR / "resultados_evaluacion.csv"


def main():
    if not index_is_current(SETTINGS, DOCUMENTS_DIR):
        raise SystemExit("Primero indexa los documentos con la interfaz Streamlit")
    cases = json.loads(INPUT.read_text(encoding="utf-8"))
    embedder = LocalEmbedder(SETTINGS.embedding_model)
    store = ChromaStore(CHROMA_DIR, SETTINGS.collection_name)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "pregunta", "fuente_esperada", "fuentes_recuperadas",
            "similitud_maxima", "uso_llm", "respuesta", "evaluacion_manual",
        ])
        writer.writeheader()
        for case in cases:
            result = answer_question(
                case["pregunta"], embedder, store, SETTINGS.llm_model,
                SETTINGS.ollama_url, top_k=5, min_similarity=0.43,
            )
            writer.writerow({
                "pregunta": case["pregunta"],
                "fuente_esperada": case.get("fuente_esperada") or "SIN RESPUESTA",
                "fuentes_recuperadas": "; ".join({p.source for p in result.supported}),
                "similitud_maxima": round(max([p.score for p in result.retrieved], default=0), 3),
                "uso_llm": "SI" if result.called_llm else "NO",
                "respuesta": result.text,
                "evaluacion_manual": "PENDIENTE",
            })
            print(f"Pregunta: {case['pregunta']}\nRespuesta: {result.text}\n")
    print(f"Reporte CSV creado: {OUTPUT}")


if __name__ == "__main__":
    main()
