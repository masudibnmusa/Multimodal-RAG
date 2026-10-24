from app import config
from app.generation.context_builder import build_context
from app.generation.prompt_templates import ANSWER_PROMPT, ANSWER_SYSTEM
from app.generation.vlm_client import ask, extract_json, text_block
from app.verification.value_checker import check_values


def generate_answer(query: str, pages: list[dict]) -> dict:
    blocks, refs = build_context(pages)
    blocks.append(text_block(ANSWER_PROMPT.format(question=query)))
    raw = ask(blocks, system=ANSWER_SYSTEM, max_tokens=1500, tag="answer")

    try:
        data = extract_json(raw)
    except ValueError:
        data = {"answer": raw}

    result = {
        "answer": data.get("answer", raw),
        "cited_pages": data.get("pages", []),
        "quoted_values": data.get("quoted_values", []),
        "sources": pages,
    }
    if config.USE_VERIFICATION and result["quoted_values"]:
        result["verification"] = check_values(result["quoted_values"], refs)
    return result