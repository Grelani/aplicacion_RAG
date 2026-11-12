# Guía breve para presentar el proyecto RAG

## Objetivo

Construir un sistema que responda preguntas sobre una colección de documentos, sin depender únicamente del conocimiento previo del LLM.

## Orden de la demostración (aprox. 5–8 minutos)

1. **Definir RAG**: recuperación de fragmentos y generación de una respuesta basada en ellos.
2. **Mostrar la carpeta** `documentos/`: contiene cinco archivos, incluido un PDF y un TXT.
3. **Explicar la carga**: PyMuPDF extrae el texto PDF por página; Python lee TXT/Markdown.
4. **Explicar el chunking**: separamos el texto en fragmentos de hasta 600 caracteres con superposición de 100 caracteres. La superposición ayuda a mantener contexto entre fragmentos.
5. **Explicar embeddings**: la red Sentence Transformers transforma textos y preguntas en vectores. Los textos con significados cercanos tienden a producir vectores próximos.
6. **Mostrar ChromaDB**: guarda vectores, fragmentos y metadatos (archivo, página, número).
7. **Mostrar la búsqueda**: se convierte la pregunta en vector; se recuperan los k resultados más similares; se descartan los que estén debajo del umbral.
8. **Mostrar Ollama**: envía pregunta y pasajes al LLM. La respuesta debe basarse en ese contexto.
9. **Hacer una pregunta conocida**: “¿Qué es la fuga de datos?”; comprobar que se muestre `02_preprocesamiento_y_datos.txt`.
10. **Hacer una pregunta desconocida**: “¿En qué año se inauguró el Metro de Morelia?”; si falla el rechazo, ajustar el umbral y explicar la limitación.

## Conceptos para defender el proyecto

- **Chunk**: unidad pequeña de texto recuperable.
- **Embedding**: representación numérica de un texto en un espacio vectorial.
- **Base vectorial**: sistema que almacena y recupera esos vectores.
- **Similitud coseno**: medida de parecido entre vectores. En esta configuración, el puntaje es `1 - distancia coseno`.
- **Top-k**: cantidad máxima de fragmentos que consulta la búsqueda.
- **Umbral**: criterio heurístico de relevancia; requiere calibración.
- **LLM**: modelo generativo de lenguaje.
- **Abstención**: admitir que el contexto no basta para responder.

## Preguntas típicas del profesor

**¿Qué diferencia hay entre RAG y entrenar el LLM?**
RAG consulta una colección externa al momento de responder; no modifica los pesos del modelo de lenguaje.

**¿Por qué no enviamos los cinco documentos completos al LLM?**
Para limitar la cantidad de texto y enfocar la respuesta en fragmentos relevantes, con menor consumo de contexto.

**¿Por qué ChromaDB?**
Facilita guardar y consultar vectores localmente, junto con metadatos de procedencia.

**¿Por qué aparecen fuentes?**
Porque cada fragmento se guardó con el nombre de archivo y, en el caso de PDF, con el número de página.

**¿Cómo se impide que invente?**
Se filtran los resultados con poca similitud y se instruye al LLM a no responder sin evidencia. Son protecciones, no garantías.

**¿Qué mejorarías?**
Agregar un reranker, OCR, evaluación con más preguntas, extracción avanzada de tablas, citas verificadas y control de acceso.
