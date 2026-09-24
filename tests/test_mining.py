import numpy as np
import pytest

from mine import mine_negatives


def test_teacher_filter_excludes_false_negative_and_all_known_positives():
    pairs = [{"query": "q", "positive": "gold", "positives": ["also gold"]}]
    corpus = ["gold", "false negative", "hard negative", "easy negative", "also gold"]
    q = np.array([[1.]])
    c = np.array([[1.], [.99], [.8], [.1], [.7]])
    gq = np.array([[1., 0.]])  # Teacher and miner dimensions may differ.
    gc = np.array([[.9, 0.], [.98, 0.], [.6, 0.], [.05, 0.], [.8, 0.]])
    rows, stats = mine_negatives(pairs, corpus, q, c, num_negatives=3,
                                guide_query_embeddings=gq, guide_corpus_embeddings=gc,
                                corpus_chunk_size=2)
    assert rows[0]["negatives"] == ["hard negative", "easy negative"]
    assert stats["rows_with_shortfall"] == 1
    assert stats["filtered_margin"] == 1
    assert stats["filtered_known_positive"] == 2
    assert rows[0]["mining"]["threshold"] == pytest.approx(.76)


def test_negative_positive_score_margin_uses_absolute_magnitude():
    pairs = [{"query": "q", "positive": "p"}]
    rows, _ = mine_negatives(pairs, ["p", "too close", "valid"], [[1]],
                             [[-.5], [-.51], [-.7]], relative_margin=.1)
    assert rows[0]["negatives"] == ["valid"]


def test_rank_bounds_are_applied_before_filtering():
    pairs = [{"query": "q", "positive": "p"}]
    rows, _ = mine_negatives(pairs, ["p", "a", "b", "c"], [[1]],
                             [[1], [.8], [.7], [.6]], range_min=2, range_max=3)
    assert rows[0]["negatives"] == ["b"]


def test_shared_query_positives_never_mined():
    pairs = [{"query": "q", "positive": "a"}, {"query": "q", "positive": "b"}]
    rows, _ = mine_negatives(pairs, ["a", "b", "c"], [[1], [1]], [[1], [.9], [.5]])
    assert all(row["negatives"] == ["c"] for row in rows)


def test_partial_guide_rejected():
    with pytest.raises(ValueError, match="both guide"):
        mine_negatives([{"query": "q", "positive": "p"}], ["p"], [[1]], [[1]],
                       guide_query_embeddings=[[1]])
