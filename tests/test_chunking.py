import pytest

from rag_core.chunking import normalize_text, split_text


def test_empty_text():
    assert split_text(" \n\t  ") == []


def test_short_text():
    assert split_text("Texto pequeño.") == ["Texto pequeño."]


def test_normalize_text():
    assert normalize_text("  hola   mundo\r\n\r\n\r\n adiós ") == "hola mundo\n\nadiós"


def test_long_text_uses_multiple_chunks_and_preserves_context():
    text = " ".join([f"palabra{i}" for i in range(200)])
    chunks = split_text(text, chunk_size=250, overlap=50)
    assert len(chunks) >= 5
    assert all(0 < len(c) <= 250 for c in chunks)
    assert "palabra0" in chunks[0]
    assert "palabra199" in chunks[-1]


def test_long_unbroken_text_finishes():
    chunks = split_text("a" * 600, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert max(map(len, chunks)) <= 200


def test_invalid_parameters():
    with pytest.raises(ValueError):
        split_text("texto", chunk_size=79)
    with pytest.raises(ValueError):
        split_text("texto", chunk_size=100, overlap=100)
