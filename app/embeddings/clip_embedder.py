"""Optional: figure-level image embeddings (not wired into the pipeline by default)."""
from sentence_transformers import SentenceTransformer


class ClipEmbedder:
    def __init__(self, name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(name)

    def embed_images(self, images):
        return self.model.encode(images, normalize_embeddings=True)

    def embed_text(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)