ANSWER_SYSTEM = (
    "You answer questions about documents using the page images and extracted text provided. "
    "Read charts, tables and diagrams carefully. Use only the provided pages. "
    "If the answer is not in the pages, say so instead of guessing."
)

ANSWER_PROMPT = """Question: {question}

Respond with ONLY a JSON object:
{{
  "answer": "your answer, mentioning page numbers like (doc.pdf, p.3)",
  "pages": [{{"doc_id": "...", "page": 1}}],
  "quoted_values": ["every number you read from a chart or table, exactly as shown"]
}}"""

RERANK_PROMPT = """Question: {question}

Above are {n} candidate pages. Rank them from most to least likely to contain the answer.
Respond with ONLY JSON: {{"ranking": [candidate numbers, best first]}}"""