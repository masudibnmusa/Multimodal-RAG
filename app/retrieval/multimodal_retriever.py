from collections import defaultdict

from app import config
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.query_router import route_query


class MultimodalRetriever:
    """Queries the text/summary index and/or the page-image index, then fuses with RRF.
    Indexes and embedders can be injected (handy for tests)."""

    def __init__(self, mode: str = config.RETRIEVAL_MODE, text_index=None, page_index=None,
                 text_embedder=None, page_embedder=None):
        self.mode = mode
        if text_index is None and mode in ("parse", "hybrid"):
            from app.vectorstore.text_index import TextIndex
            text_index = TextIndex()
        if page_index is None and mode in ("page", "hybrid"):
            from app.vectorstore.page_index import PageIndex
            page_index = PageIndex()
        self.text_index, self.page_index = text_index, page_index
        self._text_embedder, self._page_embedder = text_embedder, page_embedder

    def _text_emb(self):
        if self._text_embedder is None:
            from app.embeddings.text_embedder import TextEmbedder
            self._text_embedder = TextEmbedder()
        return self._text_embedder

    def _page_emb(self):
        if self._page_embedder is None:
            from app.embeddings.colpali_embedder import ColPaliEmbedder
            self._page_embedder = ColPaliEmbedder()
        return self._page_embedder

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        k = k or config.TOP_K
        visual = route_query(query)["visual"]
        rankings, weights = [], []
        regions = defaultdict(list)

        if self.text_index is not None:
            hits = self.text_index.search(self._text_emb().embed_query(query), config.CANDIDATES_K * 3)
            ranking = []
            for h in hits:
                m = h["meta"]
                key = (m["doc_id"], m["page"])
                if key not in ranking:
                    ranking.append(key)
                if m["element_type"] != "text" and m.get("bbox"):
                    region = {"type": m["element_type"], "bbox": m["bbox"]}
                    if region not in regions[key]:
                        regions[key].append(region)
            rankings.append(ranking)
            weights.append(1.0)

        if self.page_index is not None:
            hits = self.page_index.search(self._page_emb().embed_query(query), config.CANDIDATES_K)
            rankings.append([key for key, _ in hits])
            weights.append(2.0 if visual else 1.0)  # trust page embeddings more for visual questions

        fused = reciprocal_rank_fusion(rankings, weights)[:config.CANDIDATES_K]
        results = [{"doc_id": d, "page": p, "score": s, "regions": regions.get((d, p), [])}
                   for (d, p), s in fused]

        if config.USE_RERANKER and results:
            from app.retrieval.reranker import rerank
            results = rerank(query, results)
        return results[:k]