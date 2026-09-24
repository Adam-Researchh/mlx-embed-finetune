import json

import numpy as np
import pytest

from retrieval import (known_positives, load_corpus, load_pairs, pooled_corpus,
                       retrieval_metrics, search_top_k)


def test_chunked_search_matches_dense_including_ties():
    rng = np.random.default_rng(7)
    q = rng.normal(size=(9, 6)).astype(np.float32)
    c = rng.normal(size=(29, 6)).astype(np.float32)
    c[3] = c[2]  # Tie-breaking must not depend on corpus chunk boundaries.
    scores = q @ c.T
    expected = np.lexsort((np.broadcast_to(np.arange(len(c)), scores.shape), -scores), axis=1)
    for query_chunk in (1, 4, 64):
        for corpus_chunk in (1, 3, 4096):
            batches = list(search_top_k(q, c, 10, query_chunk, corpus_chunk))
            indices = np.concatenate([b[1] for b in batches])
            actual_scores = np.concatenate([b[2] for b in batches])
            np.testing.assert_array_equal(indices, expected[:, :10])
            np.testing.assert_allclose(actual_scores, np.take_along_axis(scores, indices, axis=1), atol=2e-6)


def test_multi_positive_metrics_hand_computed():
    q = np.array([[1., 0.], [0., 1.]])
    c = np.array([[3., 0.], [2., 1.], [1., 2.], [0., 3.]])
    result = retrieval_metrics(q, c, [{0, 2}, {2}], query_chunk_size=1, corpus_chunk_size=2)
    assert result["recall_at"][1] == pytest.approx(0.25)
    assert result["recall_at"][3] == 1.0
    assert result["mrr_at_10"] == pytest.approx(0.75)
    expected_ndcg = ((1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3)) + 1 / np.log2(3)) / 2
    assert result["ndcg_at_10"] == pytest.approx(expected_ndcg)
    assert len(result["per_query"]) == 2


def test_cutoff_is_ten_even_with_smaller_recall_cutoffs():
    q = np.ones((1, 1))
    c = np.arange(12, 0, -1, dtype=np.float32)[:, None]
    metrics = retrieval_metrics(q, c, [{9}], ks=(1,))
    assert metrics["mrr_at_10"] == 0.1
    assert metrics["ndcg_at_10"] == pytest.approx(1 / np.log2(11))
    metrics = retrieval_metrics(q, c, [{10}])
    assert metrics["mrr_at_10"] == metrics["ndcg_at_10"] == 0


def test_known_positives_union_across_rows():
    pairs = [{"query": "q", "positive": "a", "positives": ["b"], "negatives": ["c"]},
             {"query": "q", "positive": "d"}]
    assert known_positives(pairs) == {"q": {"a", "b", "d"}}
    assert pooled_corpus(pairs, ["c", "a"]) == ["c", "a", "b", "d"]


@pytest.mark.parametrize("row", [[], {"query": "q"}, {"query": "", "positive": "p"},
                                     {"query": "q", "positive": "p", "negatives": "bad"}])
def test_invalid_pairs_fail_with_line_number(tmp_path, row):
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=":1:"):
        load_pairs(path)


def test_corpus_duplicate_id_rejected(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"id":"1","text":"a"}\n{"id":"1","text":"b"}\n')
    with pytest.raises(ValueError, match="unique"):
        load_corpus(path)


def test_invalid_or_nonfinite_search_rejected():
    with pytest.raises(ValueError, match="finite"):
        list(search_top_k([[np.nan]], [[1]], 1))
    with pytest.raises(ValueError, match="positive"):
        list(search_top_k([[1]], [[1]], 0))
    with pytest.raises(ValueError, match="relevant"):
        retrieval_metrics([[1]], [[1]], [set()])
