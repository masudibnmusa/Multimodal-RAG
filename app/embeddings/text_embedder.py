from sentence_transformers import SentenceTransformer

from app import config


class TextEmbedder:
    def __init__(self, name: str = config.TEXT_EMBED_MODEL):
        self.model = SentenceTransformer(name)

    def embed(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def embed_query(self, query: str):
        return self.embed([query])[0]