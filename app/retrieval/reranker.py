from PIL import Image

from app.generation.prompt_templates import RERANK_PROMPT
from app.generation.vlm_client import ask_json, image_block, text_block
from app.vectorstore.metadata import page_image_path


def rerank(query: str, candidates: list[dict], top_n: int = 8) -> list[dict]:
    """Optional: let the VLM reorder candidate pages. Falls back to the input order on failure."""
    cands = candidates[:top_n]
    blocks = []
    for i, c in enumerate(cands, start=1):
        blocks.append(text_block(f"Candidate {i} ({c['doc_id']}, p.{c['page']}):"))
        blocks.append(image_block(Image.open(page_image_path(c["doc_id"], c["page"]))))
    blocks.append(text_block(RERANK_PROMPT.format(question=query, n=len(cands))))

    try:
        data = ask_json(blocks, max_tokens=300, tag="rerank")
        order = [int(i) - 1 for i in data["ranking"]]
    except (ValueError, KeyError, TypeError):
        return candidates

    order = [i for i in dict.fromkeys(order) if 0 <= i < len(cands)]
    ordered = [cands[i] for i in order]
    ordered += [c for i, c in enumerate(cands) if i not in order]
    return ordered + candidates[top_n:]