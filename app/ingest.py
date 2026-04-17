import logging
import re
from typing import List, Dict

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from app.config import OCR_MIN_TEXT_CHARS

logger = logging.getLogger(__name__)


def _pixmap_to_image(pix: fitz.Pixmap) -> Image.Image:
    """Convert a PyMuPDF pixmap to a Pillow image."""
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _ocr_page(page: fitz.Page) -> str:
    """Run OCR on a single PDF page and return the extracted text."""
    pix = page.get_pixmap(dpi=200)
    img = _pixmap_to_image(pix)
    return pytesseract.image_to_string(img)


def extract_pdf_pages(pdf_bytes: bytes) -> List[Dict[str, str]]:
    """Extract text per page, falling back to OCR for scanned pages."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        text = text.replace("\u00a0", " ").rstrip()
        if len(text.strip()) < OCR_MIN_TEXT_CHARS:
            try:
                ocr_text = _ocr_page(page)
                if ocr_text and ocr_text.strip():
                    logger.info("OCR fallback used for page %s", i + 1)
                    text = ocr_text.replace("\u00a0", " ").rstrip()
            except Exception as exc:
                logger.warning("OCR failed for page %s: %s", i + 1, exc)
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
