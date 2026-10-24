from collections import defaultdict

from app import config


class CostTracker:
    """Tracks token usage per tag (parse_table, answer, ...)."""

    def __init__(self):
        self.usage = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})

    def record(self, tag: str, input_tokens: int, output_tokens: int):
        u = self.usage[tag]
        u["calls"] += 1
        u["input_tokens"] += input_tokens
        u["output_tokens"] += output_tokens

    def cost(self, tag: str | None = None) -> float:
        tags = [tag] if tag else list(self.usage)
        total = 0.0
        for t in tags:
            u = self.usage[t]
            total += u["input_tokens"] / 1e6 * config.PRICE_INPUT_PER_MTOK
            total += u["output_tokens"] / 1e6 * config.PRICE_OUTPUT_PER_MTOK
        return total

    def summary(self) -> dict:
        out = {t: {**u, "cost_usd": round(self.cost(t), 4)} for t, u in self.usage.items()}
        out["total_cost_usd"] = round(self.cost(), 4)
        return out


tracker = CostTracker()