"""Interfaz Streamlit de DocuChat AI.

Ejecutar: python -m streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_core.config import (
    CHROMA_DIR, DOCUMENTS_DIR, MANIFEST_PATH, SETTINGS, SUPPORTED_EXTENSIONS,
)
from rag_core.embeddings import LocalEmbedder
from rag_core.indexing import build_index, index_is_current, load_manifest
from rag_core.loaders import list_documents
from rag_core.ollama_client import OllamaError, list_installed_models, model_installed
from rag_core.pipeline import answer_question
from rag_core.vector_store import ChromaStore

st.set_page_config(page_title="DocuChat AI | RAG local", page_icon="📚", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 2rem;}
[data-testid="stChatMessage"] {border-radius: 12px;}
.small-note {color: #708090; font-size: .89rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cargando el modelo multilingüe de embeddings...")
def get_embedder(model_name: str) -> LocalEmbedder:
    return LocalEmbedder(model_name)


@st.cache_resource(show_spinner=False)
def get_store() -> ChromaStore:
    return ChromaStore(CHROMA_DIR, SETTINGS.collection_name)


@st.cache_data(ttl=10, show_spinner=False)
def ollama_models() -> list[str]:
    return list_installed_models(SETTINGS.ollama_url)


def save_uploads(files) -> tuple[int, list[str]]:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    count, errors = 0, []
    for uploaded in files:
        filename = Path(uploaded.name.replace("\\", "/")).name
        if not filename or filename in {".", ".."} or Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            errors.append(f"Nombre o formato no admitido: {uploaded.name}")
            continue
        data = uploaded.getvalue()
        if not data or len(data) > SETTINGS.max_upload_mb * 1024 * 1024:
            errors.append(f"Archivo vacío o mayor de {SETTINGS.max_upload_mb} MB: {filename}")
            continue
        destination = DOCUMENTS_DIR / filename
        temporary = DOCUMENTS_DIR / f".{filename}.tmp"
        temporary.write_bytes(data)
        temporary.replace(destination)
        count += 1
    return count, errors


def display_sources(passages, title="Fragmentos utilizados", opened=False) -> None:
    with st.expander(f"📄 {title} ({len(passages)})", expanded=opened):
        if not passages:
            st.write("Sin fragmentos que superen el umbral establecido")
            return
        for i, passage in enumerate(passages, 1):
            location = f"Página {passage.page}" if passage.page else f"Fragmento {passage.chunk}"
            st.markdown(f"**[FUENTE {i}] {passage.source}** · {location} · similitud: {passage.score:.2f}")
            st.code(passage.text, language=None, wrap_lines=True)


DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("📂 Mis documentos")
    current_files = list_documents(DOCUMENTS_DIR)
    st.caption(f"{len(current_files)} archivos en la carpeta documentos/")
    for path in current_files:
        st.caption(f"• {path.name}")

    uploaded_files = st.file_uploader(
        "Añadir documentos", type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help=f"PDF con texto, TXT o Markdown; máximo {SETTINGS.max_upload_mb} MB por archivo",
    )
    if st.button("Guardar archivos", disabled=not uploaded_files, use_container_width=True):
        saved, errors = save_uploads(uploaded_files)
        if saved:
            st.success(f"Se guardaron {saved} archivos. Ahora reconstruye el índice")
        for error in errors:
            st.error(error)
        if saved:
            st.rerun()

    if current_files:
        selected = st.selectbox("Eliminar un archivo", options=[p.name for p in current_files])
        if st.button("Eliminar archivo seleccionado", use_container_width=True):
            (DOCUMENTS_DIR / selected).unlink()
            st.session_state["messages"] = []
            st.rerun()

    st.divider()
    if st.button("🔄 Indexar / reconstruir base", type="primary", use_container_width=True):
        try:
            with st.spinner("Extrayendo contenido y calculando embeddings..."):
                result = build_index(
                    DOCUMENTS_DIR, get_embedder(SETTINGS.embedding_model),
                    get_store(), SETTINGS,
                )
            st.session_state["messages"] = []
            st.success(f"Indexados: {result.documents} documentos, {result.chunks} fragmentos")
            if result.errors:
                for error in result.errors:
                    st.warning(error)
            else:
                st.rerun()
        except Exception as exc:
            st.error(f"No se pudo crear el índice: {exc}")

    st.divider()
    st.subheader("⚙️ Búsqueda")
    top_k = st.slider("Fragmentos a recuperar", min_value=1, max_value=10, value=5)
    min_similarity = st.slider(
        "Similitud mínima (coseno)", min_value=0.20, max_value=0.85,
        value=0.43, step=0.01,
        help="Más alto = más estricto. Se recomienda ajustar con las preguntas de prueba.",
    )
    st.caption(f"Embeddings: {SETTINGS.embedding_model.split('/')[-1]}")
    st.caption(f"LLM: {SETTINGS.llm_model} (Ollama local)")
    if st.button("Vaciar historial de chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

st.title("📚 DocuChat AI")
st.write("Pregunta sobre tus documentos y consulta los fragmentos utilizados para responder")
manifest = load_manifest()
ready = index_is_current(SETTINGS, DOCUMENTS_DIR)
indexed_chunks = get_store().count()

col1, col2, col3 = st.columns(3)
col1.metric("Archivos disponibles", len(list_documents(DOCUMENTS_DIR)))
col2.metric("Fragmentos indexados", indexed_chunks)
col3.metric("Índice", "Actualizado" if ready and indexed_chunks > 0 else "Pendiente")

if len(list_documents(DOCUMENTS_DIR)) < 5:
    st.info("Para la entrega escolar, carga al menos cinco documentos. La muestra inicial ya incluye cinco")
if not ready or indexed_chunks == 0:
    st.warning("Debes pulsar **Indexar / reconstruir base** en el panel lateral antes de preguntar")

try:
    installed = ollama_models()
    if model_installed(SETTINGS.llm_model, installed):
        st.caption(f"🟢 Ollama conectado · modelo {SETTINGS.llm_model} disponible")
    else:
        st.info(f"Ollama está funcionando, pero falta el modelo. Ejecuta: `ollama pull {SETTINGS.llm_model}`")
except OllamaError:
    st.info("Ollama no está conectado. Inícialo e instala el modelo con `ollama pull llama3.2`")

with st.expander("💡 Preguntas sugeridas para probar", expanded=not ready):
    st.markdown("""
- ¿Qué diferencia hay entre clasificación y regresión?
- ¿Qué es la fuga de datos y por qué es un problema?
- ¿Cómo funciona KNN y qué significa k?
- ¿Qué hacen la función ReLU y la retropropagación?
- ¿En qué se diferencian precisión y recall?
- ¿En qué año se inauguró el Metro de Morelia? *(prueba fuera de los documentos)*
""")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            display_sources(message.get("sources", []), opened=False)
            if message.get("other_matches"):
                display_sources(message["other_matches"], "Candidatos descartados por el umbral")

question = st.chat_input("Escribe tu pregunta acerca de los documentos...")
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not ready or indexed_chunks == 0:
            st.warning("Primero indexa tus documentos desde el panel izquierdo")
        else:
            try:
                with st.spinner("Buscando evidencia y preparando la respuesta..."):
                    result = answer_question(
                        question=question,
                        embedder=get_embedder(SETTINGS.embedding_model),
                        store=get_store(),
                        model=SETTINGS.llm_model,
                        base_url=SETTINGS.ollama_url,
                        top_k=top_k, min_similarity=min_similarity,
                    )
                st.markdown(result.text)
                display_sources(result.supported, opened=True)
                rejected = [p for p in result.retrieved if p not in result.supported]
                if rejected:
                    display_sources(rejected, "Candidatos descartados por el umbral")
                st.session_state["messages"].append({
                    "role": "assistant", "content": result.text,
                    "sources": result.supported, "other_matches": rejected,
                })
            except OllamaError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Error al procesar la pregunta: {exc}")

st.divider()
st.caption("Proyecto escolar · RAG local · Fuente = fragmento recuperado, no garantía automática de veracidad")
