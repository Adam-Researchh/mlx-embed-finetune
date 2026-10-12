import json

import numpy as np
import pytest

from retrieval import known_positives
from training_data import load_teacher_scores, make_batch, teacher_groups
from teacher import score_groups


ROWS = [
    {"query": "q", "positive": "a", "positives": ["b", "a"], "negatives": ["c", "b", "c"]},
    {"query": "q", "positive": "d", "negatives": ["c"]},
    {"query": "other", "positive": "c", "negatives": ["a"]},
]


def test_multi_positive_includes_global_relevance_and_deduplicates():
    batch = make_batch(ROWS[:1], known_positives(ROWS), multi_positive=True)
    assert batch.candidates == ["a", "b", "d", "c"]
    np.testing.assert_array_equal(batch.positive_mask, [[True, True, True, False]])
    assert batch.valid_mask.all()


def test_legacy_target_and_false_negative_mask_preserved():
    batch = make_batch(ROWS, multi_positive=False)
    assert batch.candidates[:3] == ["a", "d", "c"]
    assert np.all(batch.positive_mask.sum(axis=1) == 1)
    assert not batch.valid_mask[0, 1]  # Other row's answer is also relevant.
    assert not batch.valid_mask[0, 4]  # Explicit alternate answer is not negative.


def test_teacher_labels_stay_attached_after_shuffling_and_candidate_dedup():
    teacher = {q: {doc: i / 10 for i, doc in enumerate(docs)} for q, docs in teacher_groups(ROWS).items()}
    for multi in (False, True):
        batch = make_batch(ROWS[::-1], known_positives(ROWS), multi_positive=multi, teacher=teacher)
        for i, query in enumerate(batch.queries):
            scored_docs = [doc for j, doc in enumerate(batch.candidates) if batch.teacher_mask[i, j]]
            assert len(scored_docs) == len(set(scored_docs))
            for j, doc in enumerate(batch.candidates):
                if batch.teacher_mask[i, j]:
                    assert batch.teacher_scores[i, j] == pytest.approx(teacher[query][doc])
                    assert batch.valid_mask[i, j]


def cache_rows():
    return [{"type": "teacher_manifest", "schema_version": 1, "train_sha256": "digest"}, *[
        {"query": q, "scores": [{"document": d, "score": i / 10} for i, d in enumerate(docs)]}
        for q, docs in teacher_groups(ROWS).items()
    ]]


@pytest.mark.parametrize("corrupt", ["digest", "missing_query", "missing_doc", "duplicate_doc",
                                    "duplicate_query", "unknown_doc", "nan", "boolean", "overflow"])
def test_invalid_or_stale_teacher_cache_rejected(tmp_path, corrupt):
    records = cache_rows()
    if corrupt == "digest":
        records[0]["train_sha256"] = "other"
    elif corrupt == "missing_query":
        records.pop()
    elif corrupt == "missing_doc":
        records[1]["scores"].pop()
    elif corrupt == "duplicate_doc":
        records[1]["scores"].append(records[1]["scores"][0])
    elif corrupt == "duplicate_query":
        records.append(records[1])
    elif corrupt == "unknown_doc":
        records[1]["scores"][0]["document"] = "unseen"
    else:
        records[1]["scores"][0]["score"] = {"nan": float("nan"), "boolean": True, "overflow": 1e100}[corrupt]
    path = tmp_path / "teacher.jsonl"
    path.write_text("\n".join(map(json.dumps, records)) + "\n")
    with pytest.raises(ValueError):
        load_teacher_scores(path, ROWS, "digest")


def test_score_generation_and_loading_support_different_student_dimension(tmp_path):
    groups = teacher_groups(ROWS)
    queries = list(groups)
    documents = list(dict.fromkeys(d for docs in groups.values() for d in docs))
    q = np.ones((len(queries), 7), dtype=np.float32)
    c = np.arange(1, len(documents) * 7 + 1, dtype=np.float32).reshape(-1, 7)
    scored = score_groups(groups, queries, documents, q, c)
    records = [cache_rows()[0], *scored]
    path = tmp_path / "teacher.jsonl"
    path.write_text("\n".join(map(json.dumps, records)))
    teacher, _ = load_teacher_scores(path, ROWS, "digest")
    batch = make_batch(ROWS, multi_positive=True, teacher=teacher)
    assert batch.teacher_scores.shape == batch.positive_mask.shape
    # The cache contains scores only; no teacher vector dimensionality enters training.
    assert all(isinstance(v, float) for scores in teacher.values() for v in scores.values())


def test_distillation_requires_two_valid_candidates():
    rows = [{"query": "q", "positive": "a", "positives": ["b"]}]
    with pytest.raises(ValueError, match="two valid"):
        make_batch(rows, teacher={"q": {"a": .9, "b": .8}})
    assert make_batch(rows, multi_positive=True, teacher={"q": {"a": .9, "b": .8}}).teacher_mask.all()
