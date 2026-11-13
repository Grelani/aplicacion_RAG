# DocuChat AI — RAG Application

DocuChat AI is a Python application that allows users to ask questions about documents using **Retrieval-Augmented Generation (RAG)**, a technique that combines information retrieval with large language models (LLMs).

Institution: Escuela Nacional de Estudios Superiores (ENES), Morelia — Universidad Nacional Autónoma de México (UNAM).

Developed by: Bialy Calderón and Grecia Arias.
bialycalderonm@gmail.com
greciaariaass@gmail.com

The main goal is to generate answers based on the content of provided documents without retraining the language model.

## Technologies

- **Python 3.11:** Programming language
- **Streamlit:** User interface
- **PyMuPDF:** PDF text extraction
- **Sentence Transformers:** Text embedding generation
- **ChromaDB:** Vector database and semantic search
- **Ollama:** Local language model execution
- **Llama 3.2:** Natural language response generation

## How It Works

The application consists of two main processes.

### 1. Document Processing

1. Documents in PDF, TXT, or Markdown format are loaded
2. Text is extracted from each document
3. The extracted text is divided into smaller fragments (*chunks*)
4. Each chunk is converted into an embedding using `paraphrase-multilingual-MiniLM-L12-v2`
5. Embeddings, original text, and metadata are stored in ChromaDB

### 2. Response Generation

1. The user submits a question
2. The question is converted into an embedding
3. ChromaDB retrieves the most semantically relevant chunks
4. The application filters the retrieved chunks according to a similarity threshold
5. Llama 3.2 receives the question and selected chunks as context
6. The model generates a response based on the retrieved information

If insufficient information is found, the application is designed to indicate that the documents do not provide enough evidence to answer the question.

## Requirements

Before running the application, make sure you have installed:

- Python 3.11
- pip
- Ollama
- Git (optional)

Download Ollama from: https://ollama.com/download

## Installation and Setup

### 1. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/Grelani/aplicacion_RAG.git
cd aplicacion_RAG
```

### 2. Create a Virtual Environment

**macOS / Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows (Git Bash):**

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
```

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

With the virtual environment activated, run:

```bash
python -m pip install -r requirements.txt
```

The initial installation may take a few minutes.

### 4. Download Llama 3.2

Make sure Ollama is installed, then run:

```bash
ollama pull llama3.2
```

Verify that the model is available:

```bash
ollama list
```

Ollama must be running for the application to communicate with the language model.

### 5. Run the Application

From the project directory, with the virtual environment activated, run:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser to access DocuChat AI.

## How to Use

1. Open the Streamlit interface
2. Upload or select the available documents
3. Index the documents when necessary
4. Enter a question related to their content
5. Review the generated response and its supporting sources

The first indexing process may take longer because the embedding model needs to be downloaded.

## Embedding Model

The application uses:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

This model converts text into **384-dimensional vectors**, allowing semantic comparisons between user questions and document chunks.

## Language Model

The application uses **Llama 3.2**, running locally through Ollama.

The model receives the retrieved document fragments and generates natural language responses based on the provided context.

## Limitations

- Response quality depends on the information available in the documents
- High semantic similarity does not guarantee that a chunk contains the correct answer
- Language models may generate inaccurate information, so supporting sources should be reviewed
- Processing and response times depend on the computer's hardware

## Academic Purpose

This project was developed as an academic exercise to explore techniques for extending the knowledge available to artificial intelligence models through external information retrieval.

It demonstrates document processing, text embeddings, vector databases, semantic search, and Retrieval-Augmented Generation (RAG).

## Author & Contact

**Name:** [Your Full Name]

**GitHub:** [Grelani](https://github.com/Grelani)

**Email:** [your.email@example.com]

**Repository:** [aplicacion_RAG](https://github.com/Grelani/aplicacion_RAG)

For questions, suggestions, or feedback, feel free to contact me through GitHub or email.
