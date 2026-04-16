import re
from typing import List, Dict

import fitz  # PyMuPDF


def extract_pdf_pages(pdf_bytes: bytes) -> List[Dict[str, str]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        text = text.replace("\u00a0", " ").rstrip()
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def _split_paragraphs(text: str) -> List[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n{2,}", normalized)
    paras = [p.strip() for p in parts if p.strip()]
    return paras if paras else [normalized.strip()]


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    paras = _split_paragraphs(text)
    chunks = []
    current = []
    current_len = 0

    def split_long_block(block: str) -> List[str]:
        if len(block) <= max_chars:
            return [block]
        step = max(1, max_chars - max(0, overlap))
        parts = []
        start = 0
        while start < len(block):
            end = min(len(block), start + max_chars)
            parts.append(block[start:end].strip())
            if end >= len(block):
                break
            start += step
        return [p for p in parts if p]

    for para in paras:
        para_len = len(para)
        if para_len > max_chars:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_len = 0
            chunks.extend(split_long_block(para))
            continue
        if current and (current_len + para_len + 2) > max_chars:
            chunks.append("\n\n".join(current).strip())

            if overlap > 0:
                overlap_paras = []
                overlap_len = 0
                for p in reversed(current):
                    overlap_paras.insert(0, p)
                    overlap_len += len(p) + 2
                    if overlap_len >= overlap:
                        break
                current = overlap_paras
                current_len = sum(len(p) + 2 for p in current)
            else:
                current = []
                current_len = 0

        current.append(para)
        current_len += para_len + 2

    if current:
        chunks.append("\n\n".join(current).strip())

    return [c for c in chunks if c]


def build_chunks(pages: List[Dict[str, str]], max_chars: int, overlap: int) -> List[Dict[str, str]]:
    chunks = []
    idx = 0
    for page in pages:
        page_chunks = chunk_text(page["text"], max_chars=max_chars, overlap=overlap)
        for text in page_chunks:
            chunks.append({
                "chunk_index": idx,
                "page": page["page"],
                "text": text,
            })
            idx += 1
    return chunks
