from app import config
from app.generation.vlm_client import ask_json, image_block, text_block
from app.parsing.prompt_templates import CHART_PROMPT


def interpret_chart(img) -> dict:
    return ask_json([image_block(img), text_block(CHART_PROMPT)],
                    model=config.PARSE_MODEL, max_tokens=3000, tag="parse_chart")


def chart_to_text(c: dict) -> str:
    axes = c.get("axes", {})
    lines = [f"Chart ({c.get('chart_type', 'unknown')}): {c.get('title', '')}",
             c.get("description", ""),
             f"X axis: {axes.get('x', '')}; Y axis: {axes.get('y', '')}"]
    for p in c.get("data_points", []):
        est = " (estimated)" if p.get("estimated") else ""
        lines.append(f"- {p.get('series', '')} | {p.get('label', '')}: {p.get('value', '')}{est}")
    return "\n".join(lines)