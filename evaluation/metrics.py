import re

from app.verification.value_checker import extract_numbers


def page_recall_at_k(retrieved: list[dict], doc: str, gold_pages: list[int], k: int) -> float:
    """Fraction of gold pages found in the top-k retrieved pages of the right document."""
    got = {p["page"] for p in retrieved[:k] if p["doc_id"] == doc}
    return sum(1 for g in gold_pages if g in got) / len(gold_pages)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9.]+", " ", str(s).lower()).strip()


def has_number(text: str) -> bool:
    return bool(extract_numbers(text))


def numeric_correct(pred: str, gold: str, rel_tol: float = 0.01) -> bool:
    """Every number in the gold answer appears in the prediction (within 1%)."""
    pred_nums = extract_numbers(pred)
    return all(any(abs(p - g) <= rel_tol * max(1.0, abs(g)) for p in pred_nums)
               for g in extract_numbers(gold))


def answer_correct(pred: str, gold: str) -> bool:
    """Simple containment check. Swap in an LLM judge for free-text answers."""
    if has_number(gold):
        return numeric_correct(pred, gold)
    return _norm(gold) in _norm(pred)