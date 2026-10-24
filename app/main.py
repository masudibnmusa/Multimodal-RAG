"""Streamlit entry point. Run with: ./run.sh"""
import streamlit as st
from PIL import Image

from app import config
from app.generation.answer_generator import generate_answer
from app.ingestion.pipeline import run as run_ingestion
from app.retrieval.multimodal_retriever import MultimodalRetriever
from app.utils.cost_tracker import tracker
from app.utils.highlighter import draw_regions
from app.vectorstore.metadata import page_image_path

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("Multimodal RAG: ask about charts, tables and diagrams")


@st.cache_resource
def get_retriever(mode: str) -> MultimodalRetriever:
    return MultimodalRetriever(mode)


modes = ["hybrid", "parse", "page"]
mode = st.sidebar.selectbox("Retrieval mode", modes, index=modes.index(config.RETRIEVAL_MODE))
top_k = st.sidebar.slider("Pages to read (top-k)", 1, 5, config.TOP_K)

if st.sidebar.button("Ingest PDFs in data/pdfs"):
    with st.spinner("Ingesting... this can take a while the first time."):
        run_ingestion(mode)
        st.cache_resource.clear()
    st.sidebar.success("Done")

query = st.text_input("Ask a question about your documents")
if query:
    with st.spinner("Retrieving and reading pages..."):
        pages = get_retriever(mode).retrieve(query, top_k)
        result = generate_answer(query, pages)

    st.subheader("Answer")
    st.write(result["answer"])

    if result.get("verification"):
        with st.expander("Numeric verification"):
            st.table(result["verification"])

    st.subheader("Source pages")
    cols = st.columns(len(pages) or 1)
    for col, p in zip(cols, pages):
        img = Image.open(page_image_path(p["doc_id"], p["page"]))
        col.image(draw_regions(img, p["regions"]), caption=f"{p['doc_id']}, p.{p['page']}")

    with st.expander("Ingest + query cost"):
        st.json(tracker.summary())