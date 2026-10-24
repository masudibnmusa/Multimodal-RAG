import re

from app import config
from app.generation.vlm_client import ask, image_block, text_block
from app.parsing.prompt_templates import TABLE_PROMPT


def clean_markdown(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_table(img) -> str:
    raw = ask([image_block(img), text_block(TABLE_PROMPT)],
              model=config.PARSE_MODEL, max_tokens=3000, tag="parse_table")
    return clean_markdown(raw)