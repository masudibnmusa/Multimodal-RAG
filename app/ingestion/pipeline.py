"""Ingest PDFs: python -m app.ingestion.pipeline --mode hybrid"""
import argparse
import json

from tqdm import tqdm

from app import config
from app.ingestion.layout_detector import detect_regions
from app.ingestion.page_renderer import crop_region
from app.ingestion.pdf_loader import PageData, chunk_text, load_pdf
from app.parsing.chart_interpreter import chart_to_text, interpret_chart
from app.parsing.diagram_captioner import caption_diagram, diagram_to_text
from app.parsing.ocr_fallback import needs_ocr, transcribe_page
from app.parsing.table_extractor import extract_table
from app.utils.cost_tracker import tracker
from app.vectorstore.metadata import ElementMeta, parsed_path


def parse_page(page: PageData) -> dict:
    """Parse one page (cached on disk, so you never pay twice)."""
    cache = parsed_path(page.doc_id, page.page_num)
    if cache.exists():
        return json.loads(cache.read_text())

    img = page.image()
    text = transcribe_page(img) if needs_ocr(page.text) else page.text

    elements = []
    for region in detect_regions(img):
        crop = crop_region(img, region["bbox"])
        try:
            if region["type"] == "table":
                content = extract_table(crop)
            elif region["type"] == "chart":
                content = chart_to_text(interpret_chart(crop))
            else:  # diagram / figure
                content = diagram_to_text(caption_diagram(crop))
        except Exception as exc:  # keep going if one region fails
            print(f"  ! {region['type']} failed on {page.doc_id} p.{page.page_num}: {exc}")
            continue
        elements.append({"type": region["type"], "bbox": region["bbox"], "content": content})

    result = {"doc_id": page.doc_id, "page": page.page_num, "text": text, "elements": elements}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def index_parse_path(pages: list[PageData]):
    from app.embeddings.text_embedder import TextEmbedder
    from app.vectorstore.text_index import TextIndex

    ids, texts, metas = [], [], []
    for page in tqdm(pages, desc="Parsing pages"):
        parsed = parse_page(page)
        base = f"{page.doc_id}:p{page.page_num}"
        for i, chunk in enumerate(chunk_text(parsed["text"])):
            ids.append(f"{base}:text:{i}")
            texts.append(chunk)
            metas.append(ElementMeta(page.doc_id, page.page_num, "text"))
        for j, el in enumerate(parsed["elements"]):
            ids.append(f"{base}:{el['type']}:{j}")
            texts.append(f"[{el['type']}] {el['content']}")
            metas.append(ElementMeta(page.doc_id, page.page_num, el["type"], el["bbox"]))

    if not ids:
        print("Nothing to index on the parse path.")
        return
    embeddings = TextEmbedder().embed(texts).tolist()
    TextIndex().add(ids, texts, embeddings, metas)
    print(f"Indexed {len(ids)} text/element chunks.")


def index_page_path(pages: list[PageData], batch_size: int = 4):
    from app.embeddings.colpali_embedder import ColPaliEmbedder
    from app.vectorstore.page_index import PageIndex

    index = PageIndex()
    todo = [p for p in pages if not index.has((p.doc_id, p.page_num))]
    if not todo:
        print("Page index already up to date.")
        return
    embedder = ColPaliEmbedder()
    for i in tqdm(range(0, len(todo), batch_size), desc="Embedding pages"):
        batch = todo[i:i + batch_size]
        embs = embedder.embed_images([p.image() for p in batch], batch_size=batch_size)
        index.add([(p.doc_id, p.page_num) for p in batch], embs)
    index.save()
    print(f"Indexed {len(todo)} pages.")


def run(mode: str = config.RETRIEVAL_MODE):
    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {config.PDF_DIR}")
        return
    pages = [p for pdf in pdfs for p in load_pdf(pdf)]
    print(f"Loaded {len(pages)} pages from {len(pdfs)} PDFs (mode={mode})")

    if mode in ("parse", "hybrid"):
        index_parse_path(pages)
    if mode in ("page", "hybrid"):
        index_page_path(pages)
    print(json.dumps(tracker.summary(), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["parse", "page", "hybrid"], default=config.RETRIEVAL_MODE)
    run(ap.parse_args().mode)