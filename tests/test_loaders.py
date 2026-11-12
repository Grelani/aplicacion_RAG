from pathlib import Path

import pytest

from rag_core.loaders import list_documents, read_document


def test_txt_and_md(tmp_path):
    txt = tmp_path / "ejemplo.txt"
    md = tmp_path / "temas.md"
    txt.write_text("¡Hola, documentación RAG!", encoding="utf-8")
    md.write_text("# Una sección\nUn texto de prueba", encoding="utf-8")
    assert read_document(txt)[0].page == 0
    assert "¡Hola" in read_document(txt)[0].text
    assert "Una sección" in read_document(md)[0].text
    assert len(list_documents(tmp_path)) == 2


def test_pdf_with_two_pages(tmp_path):
    import pymupdf

    pdf_path = tmp_path / "dos_paginas.pdf"
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Pagina uno, documentos")
    pdf.new_page().insert_text((72, 72), "Pagina dos, embeddings")
    pdf.save(pdf_path)
    pdf.close()
    pages = read_document(pdf_path)
    assert [(p.page, p.text.split(",")[0]) for p in pages] == [
        (1, "Pagina uno"), (2, "Pagina dos")
    ]


def test_bad_extension(tmp_path):
    f = tmp_path / "foo.csv"
    f.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError):
        read_document(f)
