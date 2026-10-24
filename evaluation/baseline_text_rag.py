"""Text-only RAG baseline: native text layer only, no vision."""
from collections import OrderedDict

from app import config
from app.embeddings.text_embedder import TextEmbedder
from app.generation.vlm_client import ask
from app.ingestion.pdf_loader import chunk_text, load_pdf
from app.vectorstore.metadata import ElementMeta
from app.vectorstore.text_index import TextIndex

SYSTEM = "Answer using only the provided text. Cite pages like (doc.pdf, p.3). If it isn't there, say so."


class BaselineTextRAG:
    def __init__(self):
        self.index = TextIndex(name="baseline_text")
        self.embedder = TextEmbedder()

    def ensure_built(self):
        if self.index.count() > 0:
            return
        ids, texts, metas = [], [], []
        for pdf in sorted(config.PDF_DIR.glob("*.pdf")):
            for page in load_pdf(pdf):
                for i, chunk in enumerate(chunk_text(page.text)):
                    ids.append(f"{page.doc_id}:p{page.page_num}:{i}")
                    texts.append(chunk)
                    metas.append(ElementMeta(page.doc_id, page.page_num, "text"))
        if ids:
            self.index.add(ids, texts, self.embedder.embed(texts).tolist(), metas)

    def retrieve(self, query: str, k: int = config.TOP_K) -> list[dict]:
        hits = self.index.search(self.embedder.embed_query(query), k * 4)
        grouped: "OrderedDict[tuple, list[str]]" = OrderedDict()
        for h in hits:
            key = (h["meta"]["doc_id"], h["meta"]["page"])
            grouped.setdefault(key, []).append(h["text"])
        return [{"doc_id": d, "page": p, "regions": [], "text": "\n\n".join(t)}
                for (d, p), t in list(grouped.items())[:k]]

    def answer(self, query: str, pages: list[dict]) -> dict:
        context = "\n\n".join(f"[{p['doc_id']}, p.{p['page']}]\n{p['text']}" for p in pages)
        ans = ask(f"Context:\n{context}\n\nQuestion: {query}", system=SYSTEM,
                  max_tokens=800, tag="baseline_answer")
        return {"answer": ans}