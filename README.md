# Local Document Assistant (Offline)
<!--
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
-->

Fully local PDF document assistant that runs on your machine with:
- FastAPI backend
- Ollama (Qwen2.5:0.5b) for generation
- ChromaDB for vector search
- Local Python embeddings via `nomic-embed-text`

## Setup

1) Create and activate a virtual environment.
2) Install requirements:

```
pip install -r requirements.txt
```

3) Make sure Ollama is running and the model is available:

```
ollama pull qwen2.5:0.5b
```

4) Start the server:

```
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

5) Open the UI:

```
http://127.0.0.1:8000
```

## Notes

- The embedding model downloads on first run. After that, the app can run offline.
- Uploaded PDFs are stored under `data/docs/` and indexed into `data/chroma/`.
- The app preserves formatting by instructing the model to follow the source layout.

## Environment Variables

- `OLLAMA_URL` (default: http://localhost:11434/api/chat)
- `OLLAMA_MODEL` (default: qwen2.5:0.5b)
- `EMBED_MODEL_NAME` (default: nomic-ai/nomic-embed-text-v1.5)
- `MAX_CHUNK_CHARS` (default: 1200)
- `CHUNK_OVERLAP` (default: 150)
- `TOP_K` (default: 4)
