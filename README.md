# DocuChat AI — aplicación RAG local en Python

**Proyecto escolar funcional**: pregunta sobre una colección de PDF, TXT y Markdown, recupera pasajes por similitud semántica y redacta una respuesta con Ollama, mostrando documentos y páginas/fragmentos que sirvieron como contexto.

## 1. Arquitectura

```text
5+ archivos PDF/TXT/MD
       ↓
Extracción de texto (PyMuPDF / Python)
       ↓
Fragmentación por ventanas con superposición (600 / 100 caracteres)
       ↓
Embeddings con paraphrase-multilingual-MiniLM-L12-v2
       ↓
ChromaDB local persistente (distancia coseno)

Pregunta del usuario → embedding de pregunta → top-k fragmentos
                                             ↓
                                   Filtro por similitud
                                             ↓
                                Ollama + llama3.2 (LLM)
                                             ↓
                           Respuesta y fragmentos con origen
```

No usa LangChain ni servicios de pago. Los embeddings se producen en la máquina y el modelo LLM se ejecuta con Ollama en `localhost:11434`.

## 2. Requisitos previos

- Windows, macOS o Linux, con **Python 3.11 o 3.12**, pip y espacio para instalar dependencias y descargar modelos.
- **Ollama** instalado: https://ollama.com/download
- Conexión a Internet en la **primera instalación** y primera descarga del modelo de embeddings y el LLM.
- Se recomiendan al menos **8 GB de RAM** (la experiencia depende de la computadora).
- Visual Studio Code es opcional.

## 3. Instalación en Windows, paso a paso

Abre esta carpeta en VS Code. Después, en PowerShell desde la raíz del proyecto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Si PowerShell bloquea scripts**, no hace falta cambiar la política de Windows; ejecuta directamente los binarios:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Alternativa desde **Git Bash**:

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

Después, instala Ollama desde el enlace anterior y, en una terminal nueva:

```powershell
ollama pull llama3.2
ollama list
```

Ollama para Windows suele iniciar su servicio al abrir la aplicación. Si no está funcionando, abre Ollama y, si corresponde, usa `ollama serve` en una terminal. No ejecutes otro servidor si el puerto ya está ocupado.

## 4. Ejecutar la aplicación

Con el entorno activo, desde la raíz:

```bash
python -m streamlit run app.py
```

Streamlit mostrará una dirección local, por lo general `http://localhost:8501`.

En la barra lateral:

1. Revisa los **5 documentos de ejemplo** (4 de texto y 1 PDF de 2 páginas).
2. Pulsa **Indexar / reconstruir base** y espera a que aparezca el número de fragmentos.
3. Escribe una pregunta en el chat y observa la respuesta y los textos recuperados.
4. Usa **Añadir documentos** para cargar otros archivos. Después de agregar o eliminar archivos, hay que **reindexar**.

Los fragmentos y el manifiesto se almacenan en `data/`, que se crea automáticamente. Se conservan al cerrar y volver a abrir el programa.

## 5. Preguntas de demostración

| Tipo | Pregunta | Fuente esperada |
|---|---|---|
| Presente | ¿En qué se diferencia la clasificación de la regresión? | `01_fundamentos_aprendizaje_automatico.md` |
| Presente | ¿Qué es la fuga de datos? | `02_preprocesamiento_y_datos.txt` |
| Presente | ¿Cómo funciona KNN y qué significa k? | `03_algoritmos_supervisados.md` |
| Presente | ¿Para qué sirve ReLU? | `04_redes_neuronales.md` |
| Presente | ¿Qué diferencia hay entre precisión y recall? | `05_metricas_y_evaluacion.pdf` |
| Ausente | ¿Quién llegó a la Luna en 1969? | Debe abstenerse **si no se recupera evidencia suficiente** |
| Ausente | ¿En qué año se inauguró el Metro de Morelia? | Debe abstenerse **si no se recupera evidencia suficiente** |

La regla de abstención es doble: un filtro por similitud descarta resultados bajo el umbral y el prompt del LLM le exige responder solo con respaldo documental. **No es una garantía perfecta**: algunas preguntas ajenas pueden superar el umbral; el LLM puede equivocarse. Se debe medir y ajustar el umbral usando preguntas reales y revisar la respuesta.

La barra lateral permite cambiar `top-k` (1 a 10) y el umbral de similitud (0.20 a 0.85). Un valor predeterminado de `0.43` es orientativo, no un valor matemáticamente universal.

## 6. Verificar el proyecto

**Pruebas unitarias** (no necesitan Ollama ni descargar Sentence Transformers):

```bash
python -m pytest -q
```

**Evaluación experimental del recuperador** (necesita haber indexado y descargado embeddings):

```bash
python -m scripts.evaluar
```

El evaluador verifica si se recupera la fuente esperada o si, en preguntas fuera de alcance, no se recuperan fragmentos que superen el umbral. No comprueba que las respuestas del LLM sean correctas. Guarda el resultado como evidencia de pruebas y documenta los errores si los hay.

## 7. Código: dónde está cada requisito

| Requisito | Archivo |
|---|---|
| Documentos 5+ | `documentos/` |
| PDF/TXT/MD | `rag_core/loaders.py` |
| Chunks | `rag_core/chunking.py` |
| Embeddings | `rag_core/embeddings.py` |
| Persistencia Chroma | `rag_core/vector_store.py` |
| Ingesta e IDs de fragmentos | `rag_core/indexing.py` |
| Recuperación y umbral | `rag_core/pipeline.py` |
| Generación con LLM | `rag_core/ollama_client.py` |
| Interfaz con fuentes | `app.py` |
| Pruebas | `tests/`, `scripts/evaluar.py` |

## 8. Limitaciones conocidas

- **PDF escaneados** (solo imágenes) requieren OCR externo; aquí se extrae texto seleccionable.
- Documentos Markdown y TXT indican número de fragmento; un PDF indica la **página real**.
- La interfaz permite preguntas consecutivas, pero cada una se recupera de manera independiente: **no hay memoria conversacional** en el LLM.
- Se reindexa **toda la carpeta**, adecuado para una demostración con pocos documentos. No es un sistema empresarial multiusuario.
- La pantalla muestra fragmentos recuperados y etiquetas `[FUENTE N]`, pero no verifica automáticamente que todas las afirmaciones del LLM coincidan con una cita.
- Los documentos de muestra son **material didáctico original**, no artículos científicos revisados por pares.
- Los modelos consumen RAM, CPU y espacio de disco; la primera ejecución puede tardar por las descargas.
- No subas documentos confidenciales en un entorno compartido sin revisar los permisos y la seguridad del equipo.

## 9. Problemas frecuentes

**`ollama` no se reconoce**: instala Ollama, reinicia la terminal y confirma `ollama --version`.

**`No se puede conectar con Ollama`**: abre la aplicación Ollama, revisa que escuche en `localhost:11434` y prueba `ollama list`.

**`El modelo llama3.2 no está descargado`**: ejecuta `ollama pull llama3.2`.

**`No module named streamlit/chromadb`**: confirma que el entorno virtual está activado y reinstala `requirements.txt` con el Python correcto.

**`No hay texto extraíble`**: el PDF probablemente es una imagen escaneada; prueba con un PDF cuyo texto puedas seleccionar.

**Respuesta vacía o errónea**: reduce o sube el umbral con criterio, ajusta las preguntas, revisa los textos recuperados, comprueba que la carpeta se haya indexado y considera un modelo LLM más capaz si tu equipo lo permite.

**Cambio de modelo embedding**: se debe reconstruir el índice. El proyecto detecta cambios de configuración y de archivos con huellas SHA-256.

## 10. Referencias técnicas

- ChromaDB: https://docs.trychroma.com/
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
- Sentence Transformers: https://sbert.net/
- PyMuPDF: https://pymupdf.readthedocs.io/
- Streamlit: https://docs.streamlit.io/

Licencia: proyecto de ejemplo educativo, disponible para adaptación y extensión.

## 11. Evaluación de respuestas completas (opcional)

Para generar un CSV con preguntas, respuestas reales del LLM, similitudes y fuentes recuperadas, inicia Ollama y ejecuta:

```bash
python -m scripts.evaluar_generacion
```

El archivo se guarda en `data/resultados_evaluacion.csv`. La columna `evaluacion_manual` queda en `PENDIENTE`: analiza cada respuesta y decide si es correcta, si está suficientemente respaldada y si cita las fuentes adecuadas. Esto permite presentar una evaluación honesta sin inventar resultados.
