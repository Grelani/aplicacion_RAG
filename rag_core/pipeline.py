"""Pipeline RAG: pregunta -> búsqueda semántica -> respuesta fundamentada."""
from __future__ import annotations

from dataclasses import dataclass

from rag_core.ollama_client import chat_with_context
from rag_core.vector_store import Passage

NO_EVIDENCE = (
    "No encontré información suficiente en los documentos "
    "proporcionados para responder esta pregunta."
)


@dataclass(frozen=True)
class Answer:
    text: str
    retrieved: list[Passage]
    supported: list[Passage]
    called_llm: bool


def retrieve(question: str, embedder, store, top_k: int, min_similarity: float) -> tuple[list[Passage], list[Passage]]:
    vector = embedder.encode([question])[0]
    retrieved = store.search(vector, k=top_k)
    supported = [passage for passage in retrieved if passage.score >= min_similarity]
    return retrieved, supported


def answer_question(
    question: str, embedder, store, model: str, base_url: str,
    top_k: int = 5, min_similarity: float = 0.43,
) -> Answer:
    if not question.strip():
        raise ValueError("Escribe una pregunta")
    retrieved, supported = retrieve(question, embedder, store, top_k, min_similarity)
    if not supported:
        return Answer(NO_EVIDENCE, retrieved, supported, called_llm=False)
    answer = chat_with_context(question, supported, model=model, base_url=base_url)
    return Answer(answer, retrieved, supported, called_llm=True)
