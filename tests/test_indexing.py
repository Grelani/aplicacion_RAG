import json

from rag_core.config import Settings
from rag_core.indexing import build_index, index_is_current


class FakeEmbedder:
    def encode(self, texts):
        return [[0.7, 0.7] for _ in texts]


class FakeStore:
    def replace(self, records, vectors):
        self.records = records
        self.vectors = vectors


def test_index_and_detect_changes(tmp_path):
    folder = tmp_path / "documentos"
    folder.mkdir()
    doc = folder / "a.txt"
    doc.write_text("La regresión predice un valor numérico. " * 45, encoding="utf-8")
    settings = Settings()
    store = FakeStore()
    manifest = tmp_path / "indice.json"
    result = build_index(folder, FakeEmbedder(), store, settings, manifest)
    assert result.documents == 1
    assert result.chunks >= 2
    assert len(store.records) == len(store.vectors)
    assert index_is_current(settings, folder, manifest)
    assert json.loads(manifest.read_text(encoding="utf-8"))["documents"] == 1
    doc.write_text(doc.read_text(encoding="utf-8") + " Más datos.", encoding="utf-8")
    assert not index_is_current(settings, folder, manifest)
