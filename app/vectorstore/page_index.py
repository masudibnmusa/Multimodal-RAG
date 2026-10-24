import torch

from app import config


class PageIndex:
    """In-memory multi-vector index with MaxSim scoring, persisted to disk.
    Fine for a few hundred pages; swap for Qdrant/Vespa when you outgrow it."""

    def __init__(self, path=None):
        self.path = path or config.INDEX_DIR / "pages.pt"
        self.keys: list[tuple[str, int]] = []
        self.embs: list[torch.Tensor] = []
        if self.path.exists():
            data = torch.load(self.path)
            self.keys = [tuple(k) for k in data["keys"]]
            self.embs = data["embs"]
        self._seen = set(self.keys)

    def has(self, key) -> bool:
        return tuple(key) in self._seen

    def add(self, keys, embeddings):
        for key, emb in zip(keys, embeddings):
            key = tuple(key)
            if key in self._seen:
                continue
            self.keys.append(key)
            self.embs.append(emb.to("cpu").to(torch.float16))
            self._seen.add(key)

    def save(self):
        torch.save({"keys": [list(k) for k in self.keys], "embs": self.embs}, self.path)

    def search(self, query_emb: torch.Tensor, k: int = 10) -> list[tuple[tuple[str, int], float]]:
        q = query_emb.float()
        scored = []
        for key, emb in zip(self.keys, self.embs):
            sim = q @ emb.float().T                      # (n_query_tokens, n_page_patches)
            scored.append((key, sim.max(dim=1).values.sum().item()))  # late interaction
        scored.sort(key=lambda x: -x[1])
        return scored[:k]