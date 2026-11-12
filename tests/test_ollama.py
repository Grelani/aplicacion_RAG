from unittest.mock import MagicMock, patch

import pytest

from rag_core.ollama_client import OllamaError, chat_with_context, model_installed
from rag_core.vector_store import Passage


def test_model_detected_with_latest_tag():
    assert model_installed("llama3.2", ["llama3.2:latest"])
    assert not model_installed("llama3.2", ["qwen3:4b"])


def test_ollama_chat_payload_has_sources_and_non_streaming():
    result = MagicMock()
    result.json.return_value = {"message": {"content": "Respuesta basada en [FUENTE 1]."}}
    result.raise_for_status.return_value = None
    with patch("rag_core.ollama_client.requests.post", return_value=result) as call:
        response = chat_with_context(
            "¿Qué es ReLU?",
            [Passage("ReLU devuelve max(0, x)", "redes.md", 0, 1, 0.92)],
            "llama3.2", "http://localhost:11434",
        )
    assert "[FUENTE 1]" in response
    payload = call.call_args.kwargs["json"]
    assert payload["stream"] is False
    assert "redes.md" in payload["messages"][1]["content"]
    assert "¿Qué es ReLU?" in payload["messages"][1]["content"]


def test_ollama_rejects_empty_content():
    result = MagicMock()
    result.json.return_value = {"message": {"content": ""}}
    result.raise_for_status.return_value = None
    with patch("rag_core.ollama_client.requests.post", return_value=result):
        with pytest.raises(OllamaError, match="vacía"):
            chat_with_context("¿Qué?", [], "llama3.2", "http://localhost:11434")
