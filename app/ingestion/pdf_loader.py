import re
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image

from app import config
from app.ingestion.page_renderer import render_page
from app.vectorstore.metadata import page_image_path


@dataclass
class PageData:
    doc_id: str
    page_num: int   # 1-based
    image_path: Path
    text: str       # native text layer (may be empty for scans)

    def image(self) -> Image.Image:
        return Image.open(self.image_path)


def load_pdf(pdf_path: Path, dpi: int = config.DPI) -> list[PageData]:
    doc_id = pdf_path.stem
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            img_path = page_image_path(doc_id, i)
            img_path.parent.mkdir(parents=True, exist_ok=True)
            if not img_path.exists():
                render_page(page, dpi).save(img_path)
            pages.append(PageData(doc_id, i, img_path, page.get_text("text")))
    return pages


def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    """Paragraph-aware chunking."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}".strip()
    if cur:
        chunks.append(cur)
    return chunks