import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor

from app import config


class ColPaliEmbedder:
    """Page screenshots -> multi-vector patch embeddings (ColQwen2)."""

    def __init__(self, model_name: str = config.COLPALI_MODEL):
        if torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        else:
            device, dtype = "cpu", torch.float32  # works, but slow
        self.model = ColQwen2.from_pretrained(model_name, torch_dtype=dtype, device_map=device).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)

    @torch.no_grad()
    def embed_images(self, images, batch_size: int = 2) -> list[torch.Tensor]:
        out = []
        for i in range(0, len(images), batch_size):
            batch = self.processor.process_images(images[i:i + batch_size]).to(self.model.device)
            emb = self.model(**batch).to("cpu")
            out.extend(torch.unbind(emb))
        return out

    @torch.no_grad()
    def embed_query(self, query: str) -> torch.Tensor:
        batch = self.processor.process_queries([query]).to(self.model.device)
        return self.model(**batch)[0].to("cpu")