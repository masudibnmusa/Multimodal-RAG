import chromadb

from app import config
from app.vectorstore.metadata import ElementMeta


class TextIndex:
    """Chroma collection for text chunks and table/chart/diagram summaries."""

    def __init__(self, name: str = "parsed", path=None):
        self.client = chromadb.PersistentClient(path=str(path or config.INDEX_DIR / "chroma"))
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def add(self, ids, texts, embeddings, metas: list[ElementMeta], batch: int = 500):
        for i in range(0, len(ids), batch):
            s = slice(i, i + batch)
            self.col.upsert(ids=ids[s], documents=texts[s], embeddings=embeddings[s],
                            metadatas=[m.to_chroma() for m in metas[s]])

    def search(self, embedding, k: int = 10) -> list[dict]:
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(query_embeddings=[list(map(float, embedding))], n_results=min(k, n))
        return [{"text": doc, "meta": ElementMeta.from_chroma(meta), "score": 1 - dist}
                for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def count(self) -> int:
        return self.col.count()