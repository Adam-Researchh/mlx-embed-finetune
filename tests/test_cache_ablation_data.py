import numpy as np
import pytest

from benchmarks.cache_ablation_data import bm25_negatives, paired_bootstrap, positive_groups, split_rows


def test_grouping_closes_transitive_query_and_positive_links():
    rows = [{"query": "a", "positive": "one"},
            {"query": "a", "positive": "two"},
            {"query": "b", "positive": "two"},
            {"query": "c", "positive": "three"}]
    assert positive_groups(rows) == [[0, 1, 2], [3]]
    with pytest.raises(ValueError, match="two positive groups"):
        split_rows(rows, [{"query": "c"}], validation_fraction=.2)


def test_query_holdout_filter_and_split_do_not_share_positive_groups():
    rows = [{"query_id": str(i), "query": str(i), "positive": str(i // 2)} for i in range(12)]
    rows.append({"query_id": "duplicate", "query": "test", "positive": "unseen"})
    training, validation, removed = split_rows(rows, [{"query": "test"}])
    assert [r["query_id"] for r in removed] == ["duplicate"]
    assert {r["positive"] for r in training}.isdisjoint(r["positive"] for r in validation)
    assert len(training) + len(validation) == 12


def test_lexical_mining_excludes_all_answers_for_repeated_query():
    rows = [{"query": "alpha", "positive": "alpha one"},
            {"query": "alpha", "positive": "alpha two"}]
    result = bm25_negatives(rows, ["alpha one", "alpha two", "alpha unrelated", "beta"], 2)
    assert all(r["negatives"] == ["alpha unrelated", "beta"] for r in result)


def test_bootstrap_preserves_query_weight_and_pairing():
    # A three-query group must not receive the same weight as a singleton.
    values = np.array([[1., 1., 1., -1.], [1., 1., 1., -1.]])
    result = paired_bootstrap(values, [[0, 1, 2], [3]], draws=1000)
    assert result["mean_delta"] == .5
    assert result["per_seed_delta"] == [.5, .5]
    zeros = paired_bootstrap(values-values, [[0, 1, 2], [3]])
    assert zeros["exploratory_95_interval"] == [0., 0.]
    with pytest.raises(ValueError):
        paired_bootstrap(values, [[0, 1], [1, 2, 3]])
