# Local Letter RAG

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Offline](https://img.shields.io/badge/inference-100%25%20local-orange)

A fully offline retrieval-augmented generation system for PDF letters and formal documents. All document processing, embedding, and inference runs locally — no cloud API required. The system returns grounded answers with inline citations while preserving the source document's structure.

---

## Table of Contents

- [Demo](#demo)
- [Why This Exists](#why-this-exists)
- [Features](#features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Implementation Notes](#implementation-notes)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Demo

▶ [Watch the demo](https://youtu.be/95QFCS6z438)

---

## Why This Exists

Formal letters and documents (legal notices, official correspondence, government forms) often need exact-answer retrieval — names, dates, clause references — where a hosted LLM API is either overkill, a privacy risk, or unavailable. This project runs the full pipeline (extraction → embedding → retrieval → generation) on local hardware with no data leaving the machine.

---

## Features

- Upload one or more PDFs and index them locally
- Hybrid retrieval: semantic vector search (ChromaDB) combined with BM25 keyword search
- Cross-encoder reranking for improved precision on exact names, dates, and phrases
- OCR fallback via Tesseract for scanned or image-heavy PDFs
- Inline citations enforced in every generated answer
- Streaming and non-streaming response modes
- Structure-preserving system prompt for formal document outputs
- Full persistence under `data/` — state survives restarts
- Runs entirely offline after initial model download

---

## Architecture

```mermaid
flowchart LR
    UI[Vanilla JS UI] -->|HTTP| API[FastAPI app]

    API -->|upload PDF| Ingest[Ingestion pipeline]
    Ingest -->|extract text| PDF[PyMuPDF]
    Ingest -->|OCR fallback| OCR[Tesseract + Pillow]
    Ingest -->|chunk with overlap| Chunker[Paragraph-aware chunker]
    Chunker -->|embed locally| Embedder[SentenceTransformers]
    Embedder -->|persist vectors| Chroma[(ChromaDB)]
    Ingest -->|store PDF + metadata| Docs[(data/docs)]

    API -->|ask / ask/stream| Retriever[Hybrid retriever]
    Retriever -->|semantic search| Chroma
    Retriever -->|keyword search| Docs
    Retriever -->|combine results| Reranker[Cross-encoder reranker]
    Reranker -->|top-k context| Prompt[Prompt builder]
    Prompt -->|chat| Ollama[(Local LLM)]
    Ollama -->|answer| API
    API -->|response| UI
```

---

## Repository Structure

```
README.md
requirements.txt
LICENSE
app/
    __init__.py
    config.py          # environment-driven configuration
    ingest.py           # PDF extraction, OCR fallback, chunking
    llm.py               # Ollama chat + streaming client
    main.py             # FastAPI routes, retrieval, sessions, orchestration
    server.py           # ASGI entrypoint
    ui.html              # browser UI for upload and chat
    vector_store.py    # ChromaDB client and embedding persistence
data/
    chroma/             # persistent vector store
    docs/                # uploaded PDFs, registry, cached chunks
```

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and on your `PATH` (required only for scanned/image-based PDFs)

---

## Setup

```
git clone https://github.com/Anup806/local-letter-RAG.git
cd local-letter-RAG

# Create and activate a virtual environment (Windows)
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull the local LLM (in a separate terminal, Ollama must be running)
ollama pull qwen2.5:1.5b
```

Start the server:

```
.\.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000` in your browser.

---

## Usage

Upload a PDF and ask a question via the browser UI, or hit the API directly:

```
curl -X POST http://127.0.0.1:8000/upload -F "file=@sample_letter.pdf"

curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What is the effective date mentioned in the letter?\"}"
```

---

## API Endpoints

| Method   | Endpoint                | Description                                  |
| -------- | ------------------------ | --------------------------------------------- |
| `GET`    | `/`                       | Browser UI                                     |
| `GET`    | `/health`                | Checks Ollama, ChromaDB, and document count    |
| `GET`    | `/docs`                  | Lists indexed documents and chunk counts       |
| `POST`   | `/upload`                | Upload and index a PDF                        |
| `POST`   | `/ask`                    | Full JSON answer                               |
| `GET`    | `/ask/stream`            | Streaming answer (NDJSON)                      |
| `DELETE` | `/reset`                  | Clear all indexed documents and vector store   |
| `DELETE` | `/session/{session_id}`  | Clear one chat session                         |

---

## Configuration

All settings are environment-driven. Key variables:

| Variable             | Description                                |
| --------------------- | -------------------------------------------- |
| `OLLAMA_URL`          | Local Ollama chat endpoint                   |
| `OLLAMA_MODEL`        | Generation model (default: `qwen2.5:1.5b`)   |
| `EMBED_MODEL_NAME`    | Local embedding model                        |
| `MAX_CHUNK_CHARS`     | Max characters per chunk                     |
| `CHUNK_OVERLAP`       | Overlap between consecutive chunks           |
| `TOP_K`               | Number of vector search results              |
| `KW_TOP_K`            | Number of keyword search results             |
| `MAX_CONTEXT_CHUNKS`  | Max chunks passed to prompt                  |
| `OCR_MIN_TEXT_CHARS`  | Threshold to trigger OCR fallback            |

Defaults are defined in `app/config.py`.

---

## Implementation Notes

- `app/ingest.py` — extracts text with PyMuPDF; falls back to Tesseract when a page has insufficient text
- `app/vector_store.py` — persists normalized embeddings in ChromaDB using cosine similarity
- `app/main.py` — combines vector search with BM25 and cross-encoder reranking; enforces inline citations and a formal-document system prompt; limits session history to recent turns with repetition guards
- `data/docs/registry.json` — tracks uploaded file metadata and document IDs
- `data/docs/<doc_id>/chunks.json` — cached chunk data for keyword search without reprocessing
- Session history is in-memory only; no external state store required

---

## Known Limitations

- Session history resets on server restart (in-memory only, no persistent chat store)
- OCR quality depends on scan resolution — low-DPI scans will degrade retrieval accuracy
- Single-machine only — no multi-user auth or concurrency controls
- Generation quality is bounded by the local model size (`qwen2.5:1.5b` is small; swap in a larger Ollama model if your hardware allows)

---

## License

Released under the [MIT License](LICENSE).
