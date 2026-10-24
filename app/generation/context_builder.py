import json

from PIL import Image

from app import config
from app.generation.vlm_client import image_block, text_block
from app.vectorstore.metadata import page_image_path, parsed_path


def load_parsed(doc_id: str, page: int) -> dict | None:
    p = parsed_path(doc_id, page)
    return json.loads(p.read_text()) if p.exists() else None


def build_context(pages: list[dict]) -> tuple[list[dict], list[str]]:
    """Return (content blocks for the VLM, reference texts for verification)."""
    blocks, refs = [], []
    for p in pages[:config.MAX_IMAGES]:
        doc_id, page = p["doc_id"], p["page"]
        blocks.append(text_block(f"[Source: {doc_id}, page {page}]"))
        blocks.append(image_block(Image.open(page_image_path(doc_id, page))))

        parsed = load_parsed(doc_id, page)
        if parsed:
            extra = [f"Extracted {e['type']}:\n{e['content']}" for e in parsed["elements"]]
            native = parsed["text"][:3000]
            blocks.append(text_block(f"Text layer:\n{native}\n\n" + "\n\n".join(extra)))
            refs.extend(e["content"] for e in parsed["elements"])
            refs.append(native)
    return blocks, refs