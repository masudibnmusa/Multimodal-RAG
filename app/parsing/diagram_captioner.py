from app import config
from app.generation.vlm_client import ask_json, image_block, text_block
from app.parsing.prompt_templates import DIAGRAM_PROMPT


def caption_diagram(img) -> dict:
    return ask_json([image_block(img), text_block(DIAGRAM_PROMPT)],
                    model=config.PARSE_MODEL, max_tokens=2000, tag="parse_diagram")


def diagram_to_text(d: dict) -> str:
    lines = [f"Diagram: {d.get('caption', '')}",
             "Components: " + ", ".join(d.get("entities", []))]
    for r in d.get("relationships", []):
        label = f" ({r['label']})" if r.get("label") else ""
        lines.append(f"- {r.get('source', '?')} -> {r.get('target', '?')}{label}")
    return "\n".join(lines)