from collections import defaultdict


def reciprocal_rank_fusion(rankings: list[list], weights: list[float] | None = None, k: int = 60):
    """Fuse several ranked lists of hashable keys. Returns [(key, score)] best first."""
    weights = weights or [1.0] * len(rankings)
    scores = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        for rank, key in enumerate(ranking, start=1):
            scores[key] += w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])