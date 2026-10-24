from app.retrieval.fusion import reciprocal_rank_fusion


def test_items_in_both_lists_win():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])
    assert [k for k, _ in fused][:2] == ["a", "c"]


def test_weights_change_the_winner():
    fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 3.0])
    assert fused[0][0] == "b"


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []