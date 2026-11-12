from unittest.mock import patch

import pytest

from rag_core.pipeline import NO_EVIDENCE, answer_question
from rag_core.vector_store import Passage


class FakeEmbedder:
    def encode(self, texts):
        return [[0.2, 0.3] for _ in texts]


class FakeStore:
    def __init__(self, matches):
        self.matches = matches

    def search(self, embedding, k):
        return self.matches[:k]


def passage(score):
    return Passage("KNN utiliza vecinos cercanos.", "documento.md", 0, 1, score)


def test_abstains_without_matches_and_does_not_call_llm():
    with patch("rag_core.pipeline.chat_with_context") as llm:
        result = answer_question("¿Quién ganó el Mundial?", FakeEmbedder(), FakeStore([]), "x", "http://localhost")
    assert result.text == NO_EVIDENCE
    assert not result.called_llm
    llm.assert_not_called()


def test_abstains_when_similarity_is_low():
    with patch("rag_core.pipeline.chat_with_context") as llm:
        result = answer_question("¿Cuál es la capital?", FakeEmbedder(), FakeStore([passage(0.2)]), "x", "http://localhost")
    assert result.text == NO_EVIDENCE
    assert len(result.retrieved) == 1
    assert result.supported == []
    llm.assert_not_called()


def test_generates_answer_only_with_evidence():
    with patch("rag_core.pipeline.chat_with_context", return_value="KNN compara vecinos [FUENTE 1].") as llm:
        result = answer_question("¿Qué es KNN?", FakeEmbedder(), FakeStore([passage(0.88)]), "x", "http://localhost")
    assert result.called_llm
    assert "[FUENTE 1]" in result.text
    llm.assert_called_once()


def test_empty_question():
    with pytest.raises(ValueError):
        answer_question(" ", FakeEmbedder(), FakeStore([]), "x", "http://localhost")
