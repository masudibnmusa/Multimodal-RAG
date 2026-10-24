from app.retrieval.multimodal_retriever import MultimodalRetriever
from app.retrieval.query_router import route_query
from app.verification.value_checker import check_values


class FakeTextIndex:
    def search(self, emb, k):
        return [
            {"text": "t", "score": 0.9, "meta": {"doc_id": "d", "page": 2, "element_type": "table",
                                                 "bbox": [0.1, 0.1, 0.5, 0.5]}},
            {"text": "t", "score": 0.8, "meta": {"doc_id": "d", "page": 7, "element_type": "text", "bbox": None}},
        ]


class FakePageIndex:
    def search(self, emb, k):
        return [(("d", 2), 1.0), (("d", 5), 0.5)]


class FakeEmbedder:
    def embed_query(self, q):
        return [0.0]


def make_retriever():
    return MultimodalRetriever("hybrid", FakeTextIndex(), FakePageIndex(), FakeEmbedder(), FakeEmbedder())


def test_router_detects_visual_questions():
    assert route_query("What was Q3 revenue according to the bar chart?")["visual"]
    assert not route_query("Who is the CEO?")["visual"]


def test_hybrid_fuses_and_keeps_regions():
    results = make_retriever().retrieve("What does the table show?", k=3)
    assert (results[0]["doc_id"], results[0]["page"]) == ("d", 2)
    assert results[0]["regions"] == [{"type": "table", "bbox": [0.1, 0.1, 0.5, 0.5]}]
    assert {r["page"] for r in results} == {2, 5, 7}


def test_value_checker():
    report = check_values(["4.2", "9.9"], ["Q3 revenue: 4.2 million"])
    assert [r["status"] for r in report] == ["confirmed", "unverified"]