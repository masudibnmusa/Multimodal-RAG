from app import config
from app.generation.vlm_client import ask, image_block, text_block
from app.parsing.prompt_templates import OCR_PROMPT


def needs_ocr(native_text: str, min_chars: int = 30) -> bool:
    return len(native_text.strip()) < min_chars


def transcribe_page(img) -> str:
    return ask([image_block(img), text_block(OCR_PROMPT)],
               model=config.PARSE_MODEL, max_tokens=4000, tag="parse_ocr")