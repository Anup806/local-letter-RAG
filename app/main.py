import json
import os
import re
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from app.config import (
    DOCS_DIR,
    CHROMA_DIR,
    OLLAMA_MODEL,
    EMBED_MODEL_NAME,
    MAX_CHUNK_CHARS,
    CHUNK_OVERLAP,
    TOP_K,
    KW_TOP_K,
    MAX_CONTEXT_CHUNKS,
)
from app.ingest import extract_pdf_pages, build_chunks
from app.vector_store import get_client, get_collection, add_chunks, query_chunks
from app.llm import ollama_chat

app = FastAPI(title="Local Document Assistant")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

embedder = SentenceTransformer(EMBED_MODEL_NAME)
chroma_client = get_client()

UI_PATH = os.path.join(os.path.dirname(__file__), "ui.html")
REGISTRY_PATH = os.path.join(DOCS_DIR, "registry.json")

CHUNK_CACHE: Dict[str, List[Dict[str, Any]]] = {}

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "about",
    "what", "when", "where", "which", "would", "could", "should", "their", "there",
    "please",
}

SUMMARY_HINTS = {
    "what is the pdf about",
    "what is this pdf about",
    "what is the document about",
    "summary",
    "summarize",
    "overview",
    "describe the document",
    "describe this document",
}

TEMPLATE_HINTS = {
    "format",
    "template",
    "letter",
    "request",
    "draft",
}

FIELD_ALIASES = {
    "company name": "company",
    "company": "company",
    "contractor name": "contractor",
    "contractor": "contractor",
    "engineer name": "engineer",
    "engineer": "engineer",
    "recipient": "recipient",
    "date": "date",
}


class AskRequest(BaseModel):
    prompt: str
    doc_id: Optional[str] = ""


def _load_ui() -> str:
    with open(UI_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _load_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {"docs": {}, "last_doc_id": ""}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: dict) -> None:
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def _register_doc(doc_id: str, filename: str, chunk_count: int) -> None:
    registry = _load_registry()
    registry["docs"][doc_id] = {
        "filename": filename,
        "chunk_count": chunk_count,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    registry["last_doc_id"] = doc_id
    _save_registry(registry)


def _resolve_doc_id(doc_id: str) -> str:
    registry = _load_registry()
    if doc_id:
        if doc_id in registry.get("docs", {}):
            return doc_id
        raise HTTPException(status_code=404, detail="doc_id not found")

    last = registry.get("last_doc_id") or ""
    if last:
        return last
    raise HTTPException(status_code=404, detail="no documents uploaded yet")


def _build_prompt(question: str, context: str, mode: str) -> tuple[str, str]:
    if mode == "summary":
        system = (
            "You are a local document assistant. Summarize the document using ONLY the context. "
            "If headings exist, include them in the summary. "
            "If the context does not contain relevant information, reply exactly: Not found in the document."
        )
    elif mode == "template":
        system = (
            "You are a local document assistant. Use ONLY the provided context. "
            "If the request asks to write a letter in the format of a template and that "
            "template appears in the context, reproduce the template structure and wording, "
            "filling missing details with placeholders like [Name], [Date], [Address]. "
            "Preserve line breaks, numbering, and headings from the context. "
            "If the context does not contain relevant information, reply exactly: Not found in the document."
        )
    else:
        system = (
            "You are a local document assistant. Use ONLY the provided context. "
            "Answer the question directly from the context and keep the original formatting. "
            "If the context does not contain relevant information, reply exactly: Not found in the document."
        )

    user = f"""QUESTION:
{question}

CONTEXT:
{context}
""".strip()
    return system, user


def _normalize_question(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _is_summary_question(text: str) -> bool:
    q = _normalize_question(text)
    return any(hint in q for hint in SUMMARY_HINTS)


def _is_template_question(text: str) -> bool:
    q = _normalize_question(text)
    return any(hint in q for hint in TEMPLATE_HINTS)


def _first_chunks(doc_id: str, count: int) -> List[Dict[str, Any]]:
    chunks = _load_chunks(doc_id)
    if not chunks:
        return []
    ordered = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
    return ordered[:count]


def _is_not_found_answer(answer: str) -> bool:
    text = (answer or "").strip().lower()
    return text == "not found in the document." or text == "not found in the document"


def _extract_key_values(prompt: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in (prompt or "").splitlines():
        if ":" not in raw_line:
            continue
        key, val = raw_line.split(":", 1)
        key_norm = re.sub(r"[^a-z0-9 ]+", " ", key.lower()).strip()
        key_norm = re.sub(r"\s+", " ", key_norm)
        alias = FIELD_ALIASES.get(key_norm)
        if not alias:
            continue
        val_clean = val.strip()
        if val_clean:
            values[alias] = val_clean
    return values


def _apply_field_replacements(template: str, values: Dict[str, str]) -> str:
    if not template or not values:
        return template

    lines = template.splitlines()

    def replace_if_match(targets: List[str], value: str) -> None:
        if not value:
            return
        for i, line in enumerate(lines):
            if line.strip().lower() in targets:
                lines[i] = value

    replace_if_match(["company", "company name", "company ltd", "company ltd."], values.get("company", ""))
    replace_if_match(["contractor", "contractor name", "contractor ltd", "contractor ltd."], values.get("contractor", ""))
    replace_if_match(["the engineer", "engineer", "engineer name"], values.get("engineer", ""))
    replace_if_match(["recipient"], values.get("recipient", ""))

    date_value = values.get("date", "")
    if date_value:
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("date"):
                lines[i] = f"Date: {date_value}"

    return "\n".join(lines)


def _template_from_hit(doc_id: str, hit: Dict[str, Any], radius: int = 1) -> str:
    meta = hit.get("metadata") or {}
    hit_index = meta.get("chunk_index")
    if hit_index is None:
        return hit.get("text", "")
    chunks = _load_chunks(doc_id)
    if not chunks:
        return hit.get("text", "")
    start = max(0, hit_index - radius)
    end = hit_index + radius
    selected = [c for c in chunks if start <= c.get("chunk_index", -1) <= end]
    selected.sort(key=lambda c: c.get("chunk_index", 0))
    return "\n\n".join([c.get("text", "") for c in selected if c.get("text")])


def _chunks_path(doc_id: str) -> str:
    return os.path.join(DOCS_DIR, doc_id, "chunks.json")


def _save_chunks(doc_id: str, chunks: List[Dict[str, Any]]) -> None:
    path = _chunks_path(doc_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def _load_chunks(doc_id: str) -> List[Dict[str, Any]]:
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
                    "page": chunk.get("page"),
                    "chunk_index": chunk.get("chunk_index"),
                },
                "distance": None,
                "score": score,
            })

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results[:top_k]


@app.get("/health")
def health():
    registry = _load_registry()
    return {
        "status": "ok",
        "model": OLLAMA_MODEL,
        "embed_model": EMBED_MODEL_NAME,
        "doc_count": len(registry.get("docs", {})),
    }


@app.get("/", response_class=HTMLResponse)
def root():
    return _load_ui()


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a .pdf file")

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

        collection = get_collection(chroma_client, f"doc_{doc_id}")
        chunk_count = add_chunks(collection, doc_id, chunks, embedder)

        _save_chunks(doc_id, chunks)
        CHUNK_CACHE.pop(doc_id, None)

        _register_doc(doc_id, file.filename, chunk_count)

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "chunk_count": chunk_count,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


@app.post("/ask")
def ask(req: AskRequest):
    try:
        prompt = (req.prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        doc_id = _resolve_doc_id((req.doc_id or "").strip())
        collection = get_collection(chroma_client, f"doc_{doc_id}")

        hits = query_chunks(collection, prompt, embedder, TOP_K)
        kw_hits = _keyword_search(prompt, doc_id, KW_TOP_K)

        is_summary = _is_summary_question(prompt)
        is_template = _is_template_question(prompt)

        if is_template and kw_hits:
            values = _extract_key_values(prompt)
            template_text = _template_from_hit(doc_id, kw_hits[0], radius=2)
            template_text = _apply_field_replacements(template_text, values)
            if template_text.strip():
                return {"doc_id": doc_id, "answer": template_text}

        primary_hits = kw_hits if (is_template and kw_hits) else hits
        secondary_hits = hits if primary_hits is kw_hits else kw_hits

        merged = []
        seen = set()
        for h in primary_hits + secondary_hits:
            meta = h.get("metadata") or {}
            key = (meta.get("page"), meta.get("chunk_index"))
            if key not in seen:
                merged.append(h)
                seen.add(key)

        summary_chunks = _first_chunks(doc_id, MAX_CONTEXT_CHUNKS) if is_summary else []
        if not merged and not summary_chunks:
            raise HTTPException(status_code=404, detail="No relevant chunks found")

        if summary_chunks:
            selected = summary_chunks
            mode = "summary"
        else:
            selected = [h for h in merged if h.get("text")][:MAX_CONTEXT_CHUNKS]
            mode = "template" if is_template else "general"

        context = "\n\n".join([h["text"] for h in selected])
        if len(context.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Extracted text is too small. The PDF may be scanned or mostly images.",
            )
        system, user = _build_prompt(prompt, context, mode)

        answer = ollama_chat(system=system, user=user, temperature=0.1)

        if is_template and _is_not_found_answer(answer) and kw_hits:
            answer = kw_hits[0].get("text") or answer

        return {"doc_id": doc_id, "answer": answer}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
