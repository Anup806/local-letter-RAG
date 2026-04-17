import json
import logging
import os
import re
import shutil
import threading
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Iterable

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from app.config import (
    DOCS_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    OLLAMA_URL,
    OLLAMA_MODEL,
    EMBED_MODEL_NAME,
    MAX_CHUNK_CHARS,
    CHUNK_OVERLAP,
    TOP_K,
    KW_TOP_K,
    MAX_CONTEXT_CHUNKS,
    REQUEST_TIMEOUT,
)
from app.ingest import extract_pdf_pages, build_chunks
from app.vector_store import get_client, get_collection, add_chunks, query_chunks
from app.llm import ollama_chat, ollama_chat_stream

app = FastAPI(title="Local Document Assistant")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

embedder = SentenceTransformer(EMBED_MODEL_NAME)
chroma_client = get_client()

UI_PATH = os.path.join(os.path.dirname(__file__), "ui.html")
REGISTRY_PATH = os.path.join(DOCS_DIR, "registry.json")

logger = logging.getLogger(__name__)

CHUNK_CACHE: Dict[str, List[Dict[str, Any]]] = {}

UPLOAD_STATE_LOCK = threading.Lock()
UPLOAD_IN_PROGRESS = False

SESSION_STORE: Dict[str, List[Dict[str, str]]] = {}
SESSION_LOCK = threading.Lock()

SYSTEM_PROMPT = """
You are a formal document assistant.
The context below is extracted from a letter or formal document template.

Rules you MUST follow:
1. Identify the document structure (header, salutation, body, closing, signature).
2. If the user asks to generate a letter, reproduce that EXACT structure.
3. Fill in placeholders like [DATE], [RECIPIENT], [SUBJECT] using info from the question.
4. Never add sections that don't exist in the source document.
5. Never omit sections that do exist in the source document.
6. Do not repeat any section. Stop after completing the letter once.
7. If the answer is not in the provided context, say: "This information is not in the uploaded document."

Context:
{context}
""".strip()

NOT_FOUND_MESSAGE = "This information is not in the uploaded document."

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "about",
    "what", "when", "where", "which", "would", "could", "should", "their", "there",
    "please",
}

class AskRequest(BaseModel):
    """Request payload for /ask."""
    prompt: str
    doc_id: Optional[str] = ""
    session_id: Optional[str] = ""


def _load_ui() -> str:
    """Load the embedded HTML UI."""
    with open(UI_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _load_registry() -> dict:
    """Load the registry, ensuring required keys are present."""
    base = {"docs": {}, "files": {}, "last_doc_id": ""}
    if not os.path.exists(REGISTRY_PATH):
        return base
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return base
    data.setdefault("docs", {})
    data.setdefault("files", {})
    data.setdefault("last_doc_id", "")

    for doc_id, info in data.get("docs", {}).items():
        filename = (info or {}).get("filename")
        if not filename:
            continue
        if filename not in data["files"]:
            data["files"][filename] = {
                "doc_id": doc_id,
                "chunk_count": (info or {}).get("chunk_count", 0),
                "created_at": (info or {}).get("created_at", ""),
            }
    return data


def _save_registry(registry: dict) -> None:
    """Persist the registry to disk."""
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def _register_doc(doc_id: str, filename: str, chunk_count: int) -> None:
    """Register an indexed document with filename tracking."""
    registry = _load_registry()
    entry = {
        "filename": filename,
        "chunk_count": chunk_count,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    registry["docs"][doc_id] = entry
    registry["files"][filename] = {
        "doc_id": doc_id,
        "chunk_count": chunk_count,
        "created_at": entry["created_at"],
    }
    registry["last_doc_id"] = doc_id
    _save_registry(registry)


def _resolve_doc_id(doc_id: str) -> str:
    """Resolve a doc_id if provided, or raise when missing/unknown."""
    registry = _load_registry()
    if doc_id:
        if doc_id in registry.get("docs", {}):
            return doc_id
        raise HTTPException(status_code=404, detail="doc_id not found")
    raise HTTPException(status_code=404, detail="doc_id is required")


def _build_prompt(question: str, context: str) -> tuple[str, str]:
    """Build the system and user prompts for the LLM."""
    system = SYSTEM_PROMPT.replace("{context}", context)
    return system, question.strip()


def _start_upload() -> None:
    """Mark upload as in progress to guard reset operations."""
    global UPLOAD_IN_PROGRESS
    with UPLOAD_STATE_LOCK:
        if UPLOAD_IN_PROGRESS:
            raise HTTPException(status_code=409, detail="Upload already in progress")
        UPLOAD_IN_PROGRESS = True


def _finish_upload() -> None:
    """Clear the upload-in-progress flag."""
    global UPLOAD_IN_PROGRESS
    with UPLOAD_STATE_LOCK:
        UPLOAD_IN_PROGRESS = False


def _is_upload_in_progress() -> bool:
    """Check whether an upload is currently running."""
    with UPLOAD_STATE_LOCK:
        return UPLOAD_IN_PROGRESS


def _get_doc_ids_for_query(doc_id: Optional[str]) -> List[str]:
    """Return doc ids to search based on an optional filter."""
    registry = _load_registry()
    docs = registry.get("docs", {})
    if doc_id:
        if doc_id not in docs:
            raise HTTPException(status_code=404, detail="doc_id not found")
        return [doc_id]
    last = registry.get("last_doc_id") or ""
    if last and last in docs:
        return [last]
    if docs:
        return [next(iter(docs.keys()))]
    raise HTTPException(status_code=404, detail="no documents uploaded yet")


def _find_doc_by_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Find a document by filename in the registry."""
    registry = _load_registry()
    if filename in registry.get("files", {}):
        info = registry["files"][filename]
        return {
            "doc_id": info.get("doc_id"),
            "chunk_count": info.get("chunk_count", 0),
        }
    for doc_id, info in registry.get("docs", {}).items():
        if (info or {}).get("filename") == filename:
            return {
                "doc_id": doc_id,
                "chunk_count": (info or {}).get("chunk_count", 0),
            }
    return None


def _list_docs_summary() -> List[Dict[str, Any]]:
    """Return a summarized list of indexed documents."""
    registry = _load_registry()
    docs = []
    for doc_id, info in registry.get("docs", {}).items():
        docs.append({
            "doc_id": doc_id,
            "filename": (info or {}).get("filename", ""),
            "chunk_count": (info or {}).get("chunk_count", 0),
            "created_at": (info or {}).get("created_at", ""),
        })
    docs.sort(key=lambda d: d.get("created_at", ""))
    return docs


def _get_or_create_session_id(session_id: Optional[str]) -> str:
    """Return an existing session id or create a new one."""
    value = (session_id or "").strip()
    if not value:
        value = uuid.uuid4().hex
    with SESSION_LOCK:
        SESSION_STORE.setdefault(value, [])
    return value


def _get_session_history(session_id: str, max_pairs: int = 3) -> List[Dict[str, str]]:
    """Return the most recent message pairs for a session."""
    with SESSION_LOCK:
        history = SESSION_STORE.get(session_id, [])
        if not history:
            return []
        return history[-max_pairs * 2 :]


def _append_session(session_id: str, prompt: str, answer: str) -> None:
    """Append a user/assistant turn to session history."""
    with SESSION_LOCK:
        SESSION_STORE.setdefault(session_id, [])
        SESSION_STORE[session_id].extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ])


def _clear_session(session_id: str) -> bool:
    """Clear a session history by id."""
    with SESSION_LOCK:
        return SESSION_STORE.pop(session_id, None) is not None


def _ollama_tags_url() -> str:
    """Resolve the Ollama tags endpoint from the configured URL."""
    if "/api/" in OLLAMA_URL:
        base = OLLAMA_URL.split("/api/")[0]
        return f"{base}/api/tags"
    return OLLAMA_URL.rstrip("/") + "/api/tags"


def _delete_docs_dir_contents() -> int:
    """Delete all files in the docs directory and return the file count."""
    deleted = 0
    if not os.path.exists(DOCS_DIR):
        return deleted
    for entry in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, entry)
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                deleted += len(files)
            shutil.rmtree(path)
        else:
            os.remove(path)
            deleted += 1
    return deleted


def _chunks_path(doc_id: str) -> str:
    """Return the on-disk chunk cache path for a document."""
    return os.path.join(DOCS_DIR, doc_id, "chunks.json")


def _save_chunks(doc_id: str, chunks: List[Dict[str, Any]]) -> None:
    """Persist chunk metadata to disk for keyword search."""
    path = _chunks_path(doc_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def _load_chunks(doc_id: str) -> List[Dict[str, Any]]:
    """Load cached chunks for a document."""
    if doc_id in CHUNK_CACHE:
        return CHUNK_CACHE[doc_id]
    path = _chunks_path(doc_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    CHUNK_CACHE[doc_id] = data
    return data


def _keyword_search(query: str, doc_id: str, top_k: int) -> List[Dict[str, Any]]:
    """Run a lightweight keyword search over cached chunks."""
    chunks = _load_chunks(doc_id)
    if not chunks:
        return []

    def normalize(text: str) -> str:
        lowered = (text or "").lower()
        # join hyphenated line breaks, e.g., "sub-\nlet" -> "sublet"
        lowered = re.sub(r"-\s*\n\s*", "", lowered)
        lowered = re.sub(r"[\r\n]+", " ", lowered)
        lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    q_norm = normalize(query or "")
    tokens = re.findall(r"[a-z0-9]{3,}", q_norm)
    keywords = [t for t in tokens if t not in STOPWORDS]
    if not keywords:
        return []

    results = []
    for chunk in chunks:
        raw = chunk.get("text") or ""
        text = normalize(raw)
        if not text:
            continue
        score = 0
        if q_norm and q_norm in text:
            score += 3
        for kw in keywords:
            if kw in text:
                score += 1
        if score > 0:
            results.append({
                "text": raw,
                "metadata": {
                    "doc_id": doc_id,
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "chunk_index": chunk.get("chunk_index"),
                },
                "distance": None,
                "score": score,
            })

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results[:top_k]


def _keyword_search_multi(query: str, doc_ids: List[str], top_k: int) -> List[Dict[str, Any]]:
    """Run keyword search across multiple documents."""
    results: List[Dict[str, Any]] = []
    for doc_id in doc_ids:
        results.extend(_keyword_search(query, doc_id, top_k))
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results[:top_k]


def _merge_hits(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge hits while removing duplicates across sources."""
    merged: List[Dict[str, Any]] = []
    seen = set()
    for hit in primary + secondary:
        meta = hit.get("metadata") or {}
        key = (
            meta.get("doc_id"),
            meta.get("chunk_index"),
            meta.get("page"),
            meta.get("source"),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    return merged


def _format_sources(hits: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """Return a compact list of source references for UI display."""
    sources: List[Dict[str, Any]] = []
    seen = set()
    for hit in hits:
        meta = hit.get("metadata") or {}
        key = (
            meta.get("doc_id"),
            meta.get("source"),
            meta.get("page"),
            meta.get("chunk_index"),
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "doc_id": meta.get("doc_id"),
            "source": meta.get("source"),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
        })
        if len(sources) >= limit:
            break
    return sources


def _clean_context(context: str) -> str:
    """Remove consecutive duplicate lines and paragraphs from context."""
    if not context:
        return context
    lines = context.splitlines()
    deduped_lines: List[str] = []
    last_line = None
    for line in lines:
        if line == last_line:
            continue
        deduped_lines.append(line)
        last_line = line

    text = "\n".join(deduped_lines)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    deduped_paras: List[str] = []
    last_para = None
    for para in paragraphs:
        if para == last_para:
            continue
        deduped_paras.append(para)
        last_para = para
    return "\n\n".join(deduped_paras)


def _dedupe_answer(answer: str) -> str:
    """Collapse repeated consecutive paragraph blocks in model output."""
    if not answer:
        return answer
    paragraphs = [p for p in answer.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        return answer

    cleaned: List[str] = []
    i = 0
    max_block = min(8, len(paragraphs) // 2)
    while i < len(paragraphs):
        matched = False
        for block_size in range(1, max_block + 1):
            block = paragraphs[i:i + block_size]
            next_block = paragraphs[i + block_size:i + 2 * block_size]
            if block and block == next_block:
                block_text = "\n\n".join(block)
                if len(block_text) >= 200:
                    cleaned.extend(block)
                    i += block_size
                    while paragraphs[i:i + block_size] == block:
                        i += block_size
                    matched = True
                    break
        if not matched:
            cleaned.append(paragraphs[i])
            i += 1
    return "\n\n".join(cleaned)


def _should_stop_repetition(text: str) -> bool:
    """Detect repeated suffix blocks to stop streaming loops."""
    if len(text) < 800:
        return False
    for size in (220, 260, 300):
        if len(text) < size * 2:
            continue
        tail = text[-size:]
        if tail and text.endswith(tail * 2):
            return True
    return False


def _retrieve_context(prompt: str, doc_id: Optional[str]) -> Dict[str, Any]:
    """Query vector and keyword matches to build context and sources."""
    doc_ids = _get_doc_ids_for_query(doc_id)
    resolved_doc_id = doc_id or (doc_ids[0] if doc_ids else None)
    collection = get_collection(chroma_client, COLLECTION_NAME)
    where = {"doc_id": resolved_doc_id} if resolved_doc_id else None
    hits = query_chunks(collection, prompt, embedder, TOP_K, where=where)
    kw_hits = _keyword_search_multi(prompt, doc_ids, KW_TOP_K)
    merged = _merge_hits(hits, kw_hits)
    selected = [h for h in merged if h.get("text")][:MAX_CONTEXT_CHUNKS]
    context = "\n\n".join([h["text"] for h in selected if h.get("text")])
    context = _clean_context(context)
    return {
        "doc_id": resolved_doc_id or "",
        "context": context,
        "merged": merged,
        "sources": _format_sources(merged),
    }


@app.get("/health")
def health():
    ollama_status = "error"
    chroma_status = "error"
    indexed_docs = len(_load_registry().get("docs", {}))

    try:
        resp = requests.get(_ollama_tags_url(), timeout=min(5, REQUEST_TIMEOUT))
        if resp.status_code == 200:
            ollama_status = "ok"
    except Exception as exc:
        logger.warning("Ollama health check failed: %s", exc)

    try:
        collections = chroma_client.list_collections()
        names = {c.name for c in collections}
        if COLLECTION_NAME in names:
            collection = get_collection(chroma_client, COLLECTION_NAME)
            _ = collection.count()
            chroma_status = "ok"
    except Exception as exc:
        logger.warning("ChromaDB health check failed: %s", exc)

    return {
        "ollama": ollama_status,
        "chromadb": chroma_status,
        "indexed_docs": indexed_docs,
    }


@app.get("/", response_class=HTMLResponse)
def root():
    return _load_ui()


@app.get("/docs")
def list_docs():
    """List indexed documents and their chunk counts."""
    docs = _list_docs_summary()
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)
    return {
        "documents": docs,
        "total_documents": len(docs),
        "total_chunks": total_chunks,
    }


@app.delete("/reset")
def reset_documents():
    """Clear the vector store and delete all uploaded documents."""
    if _is_upload_in_progress():
        raise HTTPException(status_code=409, detail="Upload in progress. Try again shortly.")

    deleted_files = _delete_docs_dir_contents()
    CHUNK_CACHE.clear()

    try:
        collection = get_collection(chroma_client, COLLECTION_NAME)
        collection.delete(where={})
    except Exception as exc:
        logger.warning("Failed to clear ChromaDB collection: %s", exc)

    _save_registry({"docs": {}, "files": {}, "last_doc_id": ""})

    return {"status": "cleared", "deleted_files": deleted_files}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear a chat session and its history."""
    if not _clear_session(session_id):
        raise HTTPException(status_code=404, detail="session_id not found")
    return {"status": "cleared", "session_id": session_id}


@app.post("/ingest")
@app.post("/upload")
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload a PDF and append its chunks to the existing collection."""
    _start_upload()
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a .pdf file")

        filename = os.path.basename(file.filename)
        existing = _find_doc_by_filename(filename)
        if existing:
            return {
                "doc_id": existing.get("doc_id", ""),
                "filename": filename,
                "chunk_count": existing.get("chunk_count", 0),
                "duplicate": True,
            }

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        pages = extract_pdf_pages(file_bytes)
        if not pages:
            raise HTTPException(status_code=400, detail="No readable text found in PDF")

        chunks = build_chunks(pages, max_chars=MAX_CHUNK_CHARS, overlap=CHUNK_OVERLAP)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to build chunks from PDF")

        doc_id = uuid.uuid4().hex[:12]
        doc_dir = os.path.join(DOCS_DIR, doc_id)
        os.makedirs(doc_dir, exist_ok=True)

        source_pdf_path = os.path.join(doc_dir, "source.pdf")
        with open(source_pdf_path, "wb") as f:
            f.write(file_bytes)

        for chunk in chunks:
            chunk["source"] = filename
            chunk["doc_id"] = doc_id

        collection = get_collection(chroma_client, COLLECTION_NAME)
        chunk_count = add_chunks(collection, doc_id, chunks, embedder, source=filename)

        _save_chunks(doc_id, chunks)
        CHUNK_CACHE.pop(doc_id, None)

        _register_doc(doc_id, filename, chunk_count)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": chunk_count,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
    finally:
        _finish_upload()


@app.post("/ask")
def ask(req: AskRequest):
    try:
        prompt = (req.prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        doc_id = (req.doc_id or "").strip() or None
        session_id = _get_or_create_session_id(req.session_id)
        history = _get_session_history(session_id)

        retrieval = _retrieve_context(prompt, doc_id)
        context = retrieval["context"]
        sources = retrieval["sources"]
        merged = retrieval["merged"]
        resolved_doc_id = retrieval["doc_id"]

        if not merged or len(context.strip()) < 50:
            answer = NOT_FOUND_MESSAGE
            _append_session(session_id, prompt, answer)
            return {
                "doc_id": resolved_doc_id,
                "answer": answer,
                "session_id": session_id,
                "sources": sources,
            }

        system, user = _build_prompt(prompt, context)
        answer = ollama_chat(system=system, user=user, temperature=0.1, history=history)
        answer = _dedupe_answer(answer)
        _append_session(session_id, prompt, answer)

        return {
            "doc_id": resolved_doc_id,
            "answer": answer,
            "session_id": session_id,
            "sources": sources,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


@app.get("/ask/stream")
def ask_stream(
    prompt: str = Query("", min_length=1),
    doc_id: str = Query(""),
    session_id: str = Query(""),
):
    """Stream an answer token-by-token using Ollama's streaming API."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    doc_id_value = (doc_id or "").strip() or None
    session_value = _get_or_create_session_id(session_id)
    history = _get_session_history(session_value)

    retrieval = _retrieve_context(prompt, doc_id_value)
    context = retrieval["context"]
    sources = retrieval["sources"]
    merged = retrieval["merged"]
    resolved_doc_id = retrieval["doc_id"]

    if not merged or len(context.strip()) < 50:
        answer = NOT_FOUND_MESSAGE
        _append_session(session_value, prompt, answer)

        def stream_static() -> Iterable[str]:
            yield json.dumps({"type": "token", "content": answer}) + "\n"
            yield json.dumps({
                "type": "done",
                "session_id": session_value,
                "doc_id": resolved_doc_id,
                "sources": sources,
            }) + "\n"

        return StreamingResponse(
            stream_static(),
            media_type="application/x-ndjson",
            headers={"X-Session-Id": session_value},
        )

    system, user = _build_prompt(prompt, context)

    def stream_tokens() -> Iterable[str]:
        answer_parts: List[str] = []
        try:
            for token in ollama_chat_stream(
                system=system,
                user=user,
                temperature=0.1,
                history=history,
            ):
                answer_parts.append(token)
                yield json.dumps({"type": "token", "content": token}) + "\n"
                if _should_stop_repetition("".join(answer_parts)):
                    break
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
            return

        answer = "".join(answer_parts).strip()
        answer = _dedupe_answer(answer)
        _append_session(session_value, prompt, answer)
        yield json.dumps({
            "type": "done",
            "session_id": session_value,
            "doc_id": resolved_doc_id,
            "sources": sources,
        }) + "\n"

    return StreamingResponse(
        stream_tokens(),
        media_type="application/x-ndjson",
        headers={"X-Session-Id": session_value},
    )
