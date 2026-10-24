"""Run: python -m evaluation.benchmark_runner --modes text parse page hybrid --k 3"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.generation.answer_generator import generate_answer
from app.retrieval.multimodal_retriever import MultimodalRetriever
from app.utils.cost_tracker import tracker
from evaluation.baseline_text_rag import BaselineTextRAG
from evaluation.metrics import answer_correct, has_number, numeric_correct, page_recall_at_k

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "questions.json"


def run_mode(mode: str, questions: list[dict], k: int) -> list[dict]:
    if mode == "text":
        rag = BaselineTextRAG()
        rag.ensure_built()
        retrieve, answer = (lambda q: rag.retrieve(q, k)), rag.answer
    else:
        retriever = MultimodalRetriever(mode)
        retrieve, answer = (lambda q: retriever.retrieve(q, k)), generate_answer

    rows = []
    for q in questions:
        doc = Path(q["doc"]).stem
        pages = retrieve(q["question"])
        pred = answer(q["question"], pages)["answer"]
        rows.append({
            "id": q["id"], "mode": mode, "type": q["type"], "pred": pred,
            "recall": page_recall_at_k(pages, doc, q["gold_pages"], k),
            "correct": answer_correct(pred, q["gold_answer"]),
            "numeric": numeric_correct(pred, q["gold_answer"]) if has_number(q["gold_answer"]) else None,
        })
    return rows


def summarize(rows: list[dict]):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["mode"], r["type"])].append(r)
        groups[(r["mode"], "ALL")].append(r)
    print(f"\n{'mode':8} {'type':9} {'n':>3} {'recall@k':>9} {'answer':>7} {'numeric':>8}")
    for (mode, qtype), rs in sorted(groups.items()):
        nums = [r["numeric"] for r in rs if r["numeric"] is not None]
        num_acc = f"{sum(nums) / len(nums):.2f}" if nums else "  -"
        print(f"{mode:8} {qtype:9} {len(rs):>3} {sum(r['recall'] for r in rs) / len(rs):>9.2f} "
              f"{sum(r['correct'] for r in rs) / len(rs):>7.2f} {num_acc:>8}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--modes", nargs="+", default=["text", "parse", "page", "hybrid"])
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    questions = json.loads(Path(args.dataset).read_text())
    all_rows = []
    for m in args.modes:
        rows = run_mode(m, questions, args.k)
        Path(f"evaluation/results_{m}.json").write_text(json.dumps(rows, indent=2))
        all_rows += rows
    summarize(all_rows)
    print("\nCost:", json.dumps(tracker.summary(), indent=2))