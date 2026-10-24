from app import config
from app.generation.vlm_client import ask_json, image_block, text_block
from app.parsing.prompt_templates import LAYOUT_PROMPT

VALID_TYPES = {"table", "chart", "diagram", "figure"}


def detect_regions(img) -> list[dict]:
    """Ask a vision model for table/chart/diagram/figure regions (normalized boxes)."""
    try:
        data = ask_json([image_block(img), text_block(LAYOUT_PROMPT)],
                        model=config.PARSE_MODEL, max_tokens=1000, tag="parse_layout")
    except ValueError:
        return []
    items = data.get("regions", []) if isinstance(data, dict) else data

    regions = []
    for r in items:
        try:
            rtype = r["type"]
            x0, y0, x1, y1 = [min(max(float(v), 0.0), 1.0) for v in r["bbox"]]
        except (KeyError, TypeError, ValueError):
            continue
        if rtype in VALID_TYPES and x1 > x0 and y1 > y0:
            regions.append({"type": rtype, "bbox": [x0, y0, x1, y1]})
    return regions