# Multimodal RAG: Query PDFs with Charts, Tables, and Diagrams

A retrieval-augmented generation (RAG) system that answers questions about the **visual content** of documents, such as:

- "What was Q3 revenue according to the bar chart?"
- "Which component connects to the cache in the architecture diagram?"
- "What is the value in row 4, column 2 of the pricing table?"

Text-only RAG loses this information: PDF text extraction drops charts, mangles tables, and ignores diagrams. This project treats every PDF page as **both text and image**, retrieves the best pages, and sends the page images to a vision LLM that reads the chart or table directly.

---

## Approaches

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| **Parse-then-embed** | A vision model describes each chart, converts tables to markdown/CSV, and captions diagrams. Everything is embedded as text. | Cheap at query time; works with any text retriever; gives bounding boxes for highlighting | Parsing errors are baked into the index; ingest is costly |
| **Page-image embedding (ColPali / ColQwen)** | Each page screenshot is embedded as multi-vector patch embeddings and matched against the query with late interaction (MaxSim). | No OCR or layout parsing; preserves visual layout | Needs a GPU; more storage; vector DB must support multi-vector search |
| **Hybrid** | Run both and fuse the results with reciprocal rank fusion. | Best coverage; lets you compare approaches fairly | Most complex |

The goal of the project is to **measure** where each approach wins (charts vs. tables vs. diagrams vs. plain text), not just to build a demo.

---

## How It Works

1. **Ingestion**: Render each PDF page to an image (about 150 to 200 DPI) and extract the native text layer.
2. **Layout analysis** (parse path): Detect regions (text, table, figure, chart, caption).
3. **Visual element processing**:
   - Tables → markdown / JSON
   - Charts → description + extracted data points
   - Diagrams → caption + entities and relationships
4. **Embedding**: Text chunks, table/chart summaries, and/or full-page ColPali embeddings, each stored with page number, doc ID, and element type.
5. **Query handling**: Optionally rewrite the query and detect whether it targets visual content.
6. **Retrieval**: Search text, summaries, and page embeddings, then fuse and rerank to the top-k pages.
7. **Generation**: Send the retrieved page images plus extracted text to a vision-capable LLM (Claude) and instruct it to cite page numbers.
8. **Verification**: For numeric answers, the model quotes the values it read, optionally cross-checked against extracted table data.
9. **Citations**: Return the page image with the relevant region highlighted.

### Data Flow

```
PDF → pdf_loader.py → page images + text layer
        ↓
 ┌──────┴──────────────────────────────┐
 │ Parse path                          │ Page-embedding path
 │ layout_detector.py                  │ colpali_embedder.py
 │ table/chart/diagram parsers (VLM)   │ (full-page multi-vector)
 │ text_embedder.py                    │         ↓
 │         ↓                           │    page_index.py
 │    text_index.py                    │
 └──────┬──────────────────────────────┘
        ↓
[User question] → query_router.py
        ↓
multimodal_retriever.py → fusion.py → top-k pages
        ↓
context_builder.py (page images + extracted text)
        ↓
vlm_client.py → answer with page citations + highlighted regions
```

---

## Project Structure

```
multimodal-rag/
├── app/
│   ├── main.py                  # Entry point (API / Streamlit)
│   ├── config.py                # Models, DPI, top-k, retrieval mode
│   ├── ingestion/               # PDF loading, page rendering, layout detection, pipeline
│   ├── parsing/                 # Table, chart, diagram parsers; OCR fallback; prompts
│   ├── embeddings/              # Text, ColPali, and optional CLIP embedders
│   ├── vectorstore/             # Text index, multi-vector page index, metadata
│   ├── retrieval/               # Query router, multimodal retriever, fusion, reranker
│   ├── generation/              # VLM client, context builder, answer generator
│   ├── verification/            # Numeric value cross-checking
│   └── utils/                   # Highlighter, cost tracker
├── evaluation/
│   ├── datasets/                # Labeled questions (chart / table / diagram / text)
│   ├── baseline_text_rag.py     # Text-only RAG for comparison
│   ├── benchmark_runner.py
│   └── metrics.py               # Page recall@k, answer accuracy, numeric accuracy
├── data/
│   ├── pdfs/                    # Input documents
│   ├── page_images/             # Rendered pages
│   ├── parsed/                  # Cached parse output (JSON), so you never pay twice
│   └── indexes/                 # Persisted indexes
├── tests/
├── .env.example
├── requirements.txt
└── run.sh
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- An Anthropic API key
- A GPU (recommended) for ColPali / ColQwen embeddings
- Poppler (only if you use `pdf2image` instead of PyMuPDF)

### Installation

```bash
git clone https://github.com/masudibnmusa/Multimodal-RAG.git
cd multimodal-rag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Core dependencies: `pymupdf`, `pillow`, `anthropic`, `colpali-engine`, `torch`.
### Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```
ANTHROPIC_API_KEY=your_key_here
```

Tune retrieval and rendering in `app/config.py`:

| Setting | Description | Example |
|---|---|---|
| `DPI` | Page render resolution | `150` to `200` |
| `RETRIEVAL_MODE` | `parse`, `page`, or `hybrid` | `hybrid` |
| `TOP_K` | Pages sent to the vision LLM | `3` |
| `VLM_MODEL` | Vision model used for answering | configurable |
| `COLPALI_MODEL` | Page-embedding model | configurable |

### Usage

1. Put PDFs in `data/pdfs/`.
2. Ingest and index:

   ```bash
   python -m app.ingestion.pipeline --mode hybrid
   ```

3. Start the app:

   ```bash
   ./run.sh
   ```

4. Ask a question, for example: *"Which quarter had the highest revenue in the bar chart?"* The answer includes page citations and, on the parse path, a highlighted region.

> Parsed output is cached in `data/parsed/`, so re-running ingestion does not re-pay for vision-model calls.

---

## Evaluation

The `evaluation/` folder compares each retrieval mode against a **text-only RAG baseline**.

**Dataset**: Hand-labeled questions with ground-truth page numbers and answers. Aim for 30 to 50 questions across 3 to 5 PDFs, mixing:

- Chart questions
- Table questions
- Diagram questions
- Plain-text questions (control)

**Example entry**:

```json
{
  "id": "q001",
  "doc": "annual_report_2025.pdf",
  "question": "What was Q3 revenue according to the bar chart?",
  "type": "chart",
  "gold_pages": [12],
  "gold_answer": "4.2M"
}
```

**Metrics**

- Page recall@k (did retrieval find the right page?)
- Answer accuracy (LLM-judged or exact match)
- Numeric accuracy (value within tolerance)
- Cost and latency per query, and ingest cost per page

Run:

```bash
python -m evaluation.benchmark_runner --modes text parse page hybrid
```

---

## Design Notes and Known Trade-offs

- **Highlighting**: Bounding boxes come from the parse path (layout detector). ColPali is page-level, though its patch-similarity maps can give rough highlights.
- **Storage**: Multi-vector page embeddings are large. Start with in-memory MaxSim for a few hundred pages, then move to Qdrant or Vespa.
- **Cost**: Vision parsing at ingest is the main expense. Use `cost_tracker.py` and the parse cache.
- **Image limits**: `context_builder.py` must respect the vision API's per-request image count and size limits.
- **Scanned PDFs**: `ocr_fallback.py` handles pages with no native text layer.

---

## Testing

```bash
pytest tests/
```

---

## License

MIT.