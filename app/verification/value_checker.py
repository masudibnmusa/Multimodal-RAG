import re

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_numbers(text: str) -> set[float]:
    nums = set()
    for m in NUM_RE.findall(str(text)):
        m = m.replace(",", "").rstrip(".")
        try:
            nums.add(float(m))
        except ValueError:
            pass
    return nums


def _close(a: float, b: float, rel_tol: float = 1e-6) -> bool:
    return abs(a - b) <= rel_tol * max(1.0, abs(a), abs(b))


def check_values(quoted: list, reference_texts: list[str]) -> list[dict]:
    """Cross-check numbers the model says it read against extracted tables/charts.
    Note: does not convert units (4.2M vs 4,200,000)."""
    ref_nums: set[float] = set()
    for t in reference_texts:
        ref_nums |= extract_numbers(t)

    report = []
    for v in quoted:
        nums = extract_numbers(v)
        if not nums:
            status = "no_number"
        elif not ref_nums:
            status = "no_reference"
        elif any(_close(n, r) for n in nums for r in ref_nums):
            status = "confirmed"
        else:
            status = "unverified"
        report.append({"value": v, "status": status})
    return report