import base64
import io
import json
import re

from anthropic import Anthropic
from PIL import Image

from app import config
from app.utils.cost_tracker import tracker

_client = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def image_block(img: Image.Image) -> dict:
    img = img.convert("RGB")
    img.thumbnail((config.MAX_IMAGE_SIDE, config.MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    data = base64.standard_b64encode(buf.getvalue()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def ask(content, system: str = "", model: str | None = None,
        max_tokens: int = 2000, tag: str = "query") -> str:
    if isinstance(content, str):
        content = [text_block(content)]
    resp = _get_client().messages.create(
        model=model or config.VLM_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    tracker.record(tag, resp.usage.input_tokens, resp.usage.output_tokens)
    return "".join(b.text for b in resp.content if b.type == "text")


def extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    candidates = []
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = text.find(open_c), text.rfind(close_c)
        if s != -1 and e > s:
            candidates.append((s, e))
    for s, e in sorted(candidates):  # earliest opening bracket first
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON found in model output: {text[:200]!r}")


def ask_json(content, **kwargs):
    return extract_json(ask(content, **kwargs))