import json
import logging
import os
import re
import shutil
import threading
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Optional, List, Dict, Any, Iterable

import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

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
    BM25_TOP_K,
    BM25_MIN_TOKEN_LEN,
    MAX_CONTEXT_CHUNKS,
    REQUEST_TIMEOUT,
    MONTHLY_REPORT_TEMPLATE_PATH,
    RERANK_ENABLED,
    RERANK_MODEL_NAME,
    RERANK_TOP_N,
    RERANK_BATCH_SIZE,
    CITATION_STRICT,
    CITATION_PER_PARAGRAPH,
    STREAM_CHUNK_SIZE,
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
BM25_CACHE: Dict[str, Dict[str, Any]] = {}
BM25_LOCK = threading.Lock()

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
8. Cite every answer sentence with one or more tags like [S1].
9. Use only the citation tags provided in the context. Do not invent tags.
10. Keep citations inline. Do not add new sections for citations.

Context:
<<CONTEXT>>
""".strip()

MONTHLY_REPORT_SYSTEM_PROMPT = """
You are a project reporting assistant.
Use the monthly project report template below as the ONLY output format.

Rules you MUST follow:
1. Output must follow the template text, headings, numbering, and bullet order exactly.
2. Keep all section titles, labels, and line breaks exactly as written in the template.
3. Fill in values using only the provided context from the daily project report.
4. If a value is missing, keep the label and write: "This information is not in the uploaded document."
5. For instruction lines (e.g., "Insert location Map", "S-Curve to be plotted"), keep them verbatim unless the context explicitly provides replacement data.
6. Do not add sections or commentary outside the template.

Monthly project report template:
<<TEMPLATE>>

Context:
<<CONTEXT>>
""".strip()

NOT_FOUND_MESSAGE = "This information is not in the uploaded document."

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "about",
    "what", "when", "where", "which", "would", "could", "should", "their", "there",
    "please",
}


def _tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text for BM25 using a simple alphanumeric filter."""
    lowered = (text or "").lower()
    lowered = re.sub(r"-\s*\n\s*", "", lowered)
    lowered = re.sub(r"[\r\n]+", " ", lowered)
    tokens = re.findall(r"[a-z0-9]{%d,}" % BM25_MIN_TOKEN_LEN, lowered)
    return [t for t in tokens if t not in STOPWORDS]


def _get_bm25_index(doc_id: str) -> Optional[Dict[str, Any]]:
    """Return a cached BM25 index for a document."""
    with BM25_LOCK:
        if doc_id in BM25_CACHE:
            return BM25_CACHE[doc_id]

    chunks = _load_chunks(doc_id)
    if not chunks:
        return None

    tokenized = [_tokenize_for_bm25(c.get("text") or "") for c in chunks]
    if not any(tokenized):
        return None

    bm25 = BM25Okapi(tokenized)
    data = {"bm25": bm25, "chunks": chunks}
    with BM25_LOCK:
        BM25_CACHE[doc_id] = data
    return data


def _bm25_search(query: str, doc_id: str, top_k: int) -> List[Dict[str, Any]]:
    """Run BM25 keyword search over cached chunks."""
    data = _get_bm25_index(doc_id)
    if not data:
        return []

    tokens = _tokenize_for_bm25(query or "")
    if not tokens:
        return []

    bm25 = data["bm25"]
    chunks = data["chunks"]
    scores = bm25.get_scores(tokens)
    if scores is None or len(scores) == 0:
        return []

    ranked = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in ranked:
        chunk = chunks[int(idx)]
        score = float(scores[int(idx)])
        if score <= 0:
            continue
        results.append({
            "text": chunk.get("text"),
            "metadata": {
                "doc_id": doc_id,
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "chunk_index": chunk.get("chunk_index"),
            },
            "distance": None,
            "bm25_score": score,
        })
    return results


def _bm25_search_multi(query: str, doc_ids: List[str], top_k: int) -> List[Dict[str, Any]]:
    """Run BM25 keyword search across multiple documents."""
    results: List[Dict[str, Any]] = []
    for doc_id in doc_ids:
        results.extend(_bm25_search(query, doc_id, top_k))
    results.sort(key=lambda x: x.get("bm25_score", 0.0), reverse=True)
    return results[:top_k]


@lru_cache(maxsize=1)
def _get_reranker() -> Optional[CrossEncoder]:
    """Load the cross-encoder reranker lazily."""
    if not RERANK_ENABLED:
        return None
    try:
        return CrossEncoder(RERANK_MODEL_NAME)
    except Exception as exc:
        logger.warning("Reranker unavailable: %s", exc)
        return None


def _rerank_hits(prompt: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank top hits with a cross-encoder for better precision."""
    if not hits:
        return hits
    reranker = _get_reranker()
    if not reranker:
        return hits

    candidates = [h for h in hits if h.get("text")]
    if not candidates:
        return hits

    top_n = min(RERANK_TOP_N, len(candidates))
    subset = candidates[:top_n]
    pairs = [(prompt, h.get("text") or "") for h in subset]
    try:
        scores = reranker.predict(pairs, batch_size=RERANK_BATCH_SIZE)
    except Exception as exc:
        logger.warning("Rerank failed: %s", exc)
        return hits

    for item, score in zip(subset, scores):
        item["rerank_score"] = float(score)

    reranked = sorted(subset, key=lambda h: h.get("rerank_score", 0.0), reverse=True)
    remainder = candidates[top_n:]
    tail = [h for h in hits if not h.get("text")]
    return reranked + remainder + tail


def _build_cited_context(selected: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """Build a context string with citation tags and return a citation map."""
    parts: List[str] = []
    citations: List[Dict[str, Any]] = []
    for idx, hit in enumerate(selected, start=1):
        meta = hit.get("metadata") or {}
        tag = f"S{idx}"
        citations.append({
            "tag": tag,
            "doc_id": meta.get("doc_id"),
            "source": meta.get("source"),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
        })
        header = "[{tag}] source={source} page={page} chunk={chunk}".format(
            tag=tag,
            source=meta.get("source", ""),
            page=meta.get("page", ""),
            chunk=meta.get("chunk_index", ""),
        )
        parts.append(f"{header}\n{hit.get('text') or ''}")
    return "\n\n".join(parts), citations


def _citation_tag_set(citations: List[Dict[str, Any]]) -> set[str]:
    return {f"[{c.get('tag')}]" for c in citations if c.get("tag")}


def _validate_citations(answer: str, tags: set[str]) -> bool:
    """Ensure the answer uses only valid tags and includes citations."""
    if not tags:
        return False
    used = set(re.findall(r"\[S\d+\]", answer or ""))
    if not used or not used.issubset(tags):
        return False
    if not CITATION_PER_PARAGRAPH:
        return True
    paragraphs = [p for p in (answer or "").split("\n\n") if p.strip()]
    for para in paragraphs:
        if not re.search(r"\[S\d+\]", para):
            return False
    return True


def _enforce_citations(answer: str, citations: List[Dict[str, Any]], is_monthly: bool) -> str:
    """Apply citation enforcement to the model answer."""
    if is_monthly or not CITATION_STRICT:
        return answer
    if not answer or answer.strip() == NOT_FOUND_MESSAGE:
        return answer
    tags = _citation_tag_set(citations)
    if not _validate_citations(answer, tags):
        return NOT_FOUND_MESSAGE
    return answer


def _stream_text_chunks(text: str, chunk_size: int) -> Iterable[str]:
    """Yield text in fixed-size chunks for streaming responses."""
    if not text:
        return
    for i in range(0, len(text), max(1, chunk_size)):
        yield text[i:i + chunk_size]


def _is_monthly_report_request(question: str) -> bool:
    """Check whether the question asks for a monthly report."""
    q = (question or "").lower()
    return any(
        key in q
        for key in (
            "monthly project report",
            "monthly progress report",
            "monthly report",
        )
    )


@lru_cache(maxsize=1)
def _load_monthly_report_template() -> str:
    """Load and cache the monthly report template text from PDF."""
    path = MONTHLY_REPORT_TEMPLATE_PATH
    if not path:
        return ""
    if not os.path.exists(path):
        logger.warning("Monthly report template not found: %s", path)
        return ""
    try:
        with open(path, "rb") as f:
            pages = extract_pdf_pages(f.read())
    except Exception as exc:
        logger.warning("Failed to read monthly report template: %s", exc)
        return ""

    parts = [
        (page.get("text") or "").strip()
        for page in pages
        if (page.get("text") or "").strip()
    ]
    return "\n\n".join(parts).strip()

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
    prompt = (question or "").strip()
    if _is_monthly_report_request(prompt):
        template = _load_monthly_report_template()
        if not template:
            raise HTTPException(
                status_code=500,
                detail="Monthly report template not found or unreadable",
            )
        system = MONTHLY_REPORT_SYSTEM_PROMPT
        system = system.replace("<<TEMPLATE>>", template)
        system = system.replace("<<CONTEXT>>", context)
        return system, prompt

    system = SYSTEM_PROMPT.replace("<<CONTEXT>>", context)
    return system, prompt


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
    with BM25_LOCK:
        BM25_CACHE.pop(doc_id, None)


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
    """Run BM25 keyword search over cached chunks."""
    return _bm25_search(query, doc_id, top_k)


def _keyword_search_multi(query: str, doc_ids: List[str], top_k: int) -> List[Dict[str, Any]]:
    """Run BM25 keyword search across multiple documents."""
    return _bm25_search_multi(query, doc_ids, top_k)


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


def _format_sources(
    hits: List[Dict[str, Any]],
    limit: int = 5,
    citation_map: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return a compact list of source references for UI display."""
    sources: List[Dict[str, Any]] = []
    seen = set()
    tag_lookup: Dict[tuple, str] = {}
    if citation_map:
        for item in citation_map:
            key = (
                item.get("doc_id"),
                item.get("source"),
                item.get("page"),
                item.get("chunk_index"),
            )
            tag = item.get("tag")
            if tag:
                tag_lookup[key] = tag

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
        entry = {
            "doc_id": meta.get("doc_id"),
            "source": meta.get("source"),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
        }
        tag = tag_lookup.get(key)
        if tag:
            entry["tag"] = tag
        sources.append(entry)
        if len(sources) >= limit:
            break
    return sources


def _log_selected_chunks(selected: List[Dict[str, Any]]) -> None:
    """Print the chunks that were selected for the final context."""
    print("\n=== Selected context chunks ===", flush=True)
    if not selected:
        print("(none)", flush=True)
        return

    for item in selected:
        meta = item.get("metadata") or {}
        text = (item.get("text") or "").strip()
        print(
            "--- chunk_index={chunk_index} page={page} source={source} doc_id={doc_id} distance={distance} ---".format(
                chunk_index=meta.get("chunk_index", ""),
                page=meta.get("page", ""),
                source=meta.get("source", ""),
                doc_id=meta.get("doc_id", ""),
                distance=item.get("distance", ""),
            ),
            flush=True,
        )
        print(text, flush=True)
    print("=== End selected context chunks ===\n", flush=True)


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


def _retrieve_context(
    prompt: str,
    doc_id: Optional[str],
    include_citations: bool = True,
) -> Dict[str, Any]:
    """Query vector and BM25 matches to build context and sources."""
    doc_ids = _get_doc_ids_for_query(doc_id)
    resolved_doc_id = doc_id or (doc_ids[0] if doc_ids else None)
    collection = get_collection(chroma_client, COLLECTION_NAME)
    where = {"doc_id": resolved_doc_id} if resolved_doc_id else None
    hits = query_chunks(collection, prompt, embedder, TOP_K, where=where)
    bm25_hits = _bm25_search_multi(prompt, doc_ids, BM25_TOP_K)
    merged = _merge_hits(hits, bm25_hits)
    merged = _rerank_hits(prompt, merged)
    selected = [h for h in merged if h.get("text")][:MAX_CONTEXT_CHUNKS]
    citations: List[Dict[str, Any]] = []
    if include_citations:
        context, citations = _build_cited_context(selected)
    else:
        context = "\n\n".join([h["text"] for h in selected if h.get("text")])
    context = _clean_context(context)
    return {
        "doc_id": resolved_doc_id or "",
        "context": context,
        "merged": merged,
        "sources": _format_sources(merged, citation_map=citations),
        "citations": citations,
        "selected": selected,
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
    BM25_CACHE.clear()

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
        with BM25_LOCK:
            BM25_CACHE.pop(doc_id, None)

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
        is_monthly = _is_monthly_report_request(prompt)
        doc_id = (req.doc_id or "").strip() or None
        session_id = _get_or_create_session_id(req.session_id)
        history = _get_session_history(session_id)

        retrieval = _retrieve_context(prompt, doc_id, include_citations=not is_monthly)
        context = retrieval["context"]
        sources = retrieval["sources"]
        merged = retrieval["merged"]
        citations = retrieval["citations"]
        resolved_doc_id = retrieval["doc_id"]
        selected = retrieval["selected"]

        _log_selected_chunks(selected)

        if (not merged or len(context.strip()) < 50) and not is_monthly:
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
        answer = _enforce_citations(answer, citations, is_monthly)
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

    is_monthly = _is_monthly_report_request(prompt)

    doc_id_value = (doc_id or "").strip() or None
    session_value = _get_or_create_session_id(session_id)
    history = _get_session_history(session_value)

    retrieval = _retrieve_context(prompt, doc_id_value, include_citations=not is_monthly)
    context = retrieval["context"]
    sources = retrieval["sources"]
    merged = retrieval["merged"]
    citations = retrieval["citations"]
    resolved_doc_id = retrieval["doc_id"]
    selected = retrieval["selected"]

    _log_selected_chunks(selected)

    if (not merged or len(context.strip()) < 50) and not is_monthly:
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

    if CITATION_STRICT and not is_monthly:
        answer = ollama_chat(system=system, user=user, temperature=0.1, history=history)
        answer = _dedupe_answer(answer)
        answer = _enforce_citations(answer, citations, is_monthly)
        _append_session(session_value, prompt, answer)

        def stream_static() -> Iterable[str]:
            for chunk in _stream_text_chunks(answer, STREAM_CHUNK_SIZE):
                yield json.dumps({"type": "token", "content": chunk}) + "\n"
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
        if CITATION_STRICT:
            answer = _enforce_citations(answer, citations, is_monthly)
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
