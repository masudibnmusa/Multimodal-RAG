"""Central configuration. Override anything via environment variables / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
PAGE_IMG_DIR = DATA_DIR / "page_images"
PARSED_DIR = DATA_DIR / "parsed"
INDEX_DIR = DATA_DIR / "indexes"
for _d in (PDF_DIR, PAGE_IMG_DIR, PARSED_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Models
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
VLM_MODEL = os.getenv("VLM_MODEL", "claude-sonnet-5-5")   # answering
PARSE_MODEL = os.getenv("PARSE_MODEL", VLM_MODEL)         # ingest-time parsing (use a cheaper model if you like)
COLPALI_MODEL = os.getenv("COLPALI_MODEL", "vidore/colqwen2-v1.0")
TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Rendering
DPI = int(os.getenv("DPI", "150"))
MAX_IMAGE_SIDE = 1568  # longest side sent to the vision API

# Retrieval
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # parse | page | hybrid
TOP_K = int(os.getenv("TOP_K", "3"))
CANDIDATES_K = 10
MAX_IMAGES = 5
USE_RERANKER = False
USE_VERIFICATION = True

# Cost tracking (USD per million tokens); fill in from the pricing page
PRICE_INPUT_PER_MTOK = float(os.getenv("PRICE_INPUT_PER_MTOK", "0"))
PRICE_OUTPUT_PER_MTOK = float(os.getenv("PRICE_OUTPUT_PER_MTOK", "0"))