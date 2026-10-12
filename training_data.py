"""Candidate construction and offline teacher score validation (NumPy only)."""

from dataclasses import dataclass
import math

import numpy as np

from retrieval import known_positives, positive_texts, read_jsonl


def teacher_groups(pairs):
    """Fixed, deduplicated candidate groups for each exact query string."""
    groups = {}
    for row in pairs:
        docs = groups.setdefault(row["query"], {})
        for text in [*positive_texts(row), *row.get("negatives", [])]:
            docs.setdefault(text, None)
    return {query: list(docs) for query, docs in groups.items()}


def load_teacher_scores(path, pairs, train_sha256):
    records = iter(read_jsonl(path))
    first = next(records, None)
    if first is None:
        raise ValueError("Empty teacher score file")
    _, manifest = first
    if manifest.get("type") != "teacher_manifest" or manifest.get("schema_version") != 1:
        raise ValueError("Teacher scores require a schema_version=1 teacher_manifest header")
    if manifest.get("train_sha256") != train_sha256:
        raise ValueError("Teacher scores were generated for a different training file; regenerate them")
    expected = teacher_groups(pairs)
    result = {}
    for number, row in records:
        query = row.get("query")
        if not isinstance(query, str) or query not in expected or query in result:
            raise ValueError(f"{path}:{number}: unknown or duplicate teacher query")
        scores = row.get("scores")
        if not isinstance(scores, list):
            raise ValueError(f"{path}:{number}: teacher scores must be a list")
        by_doc = {}
        for item in scores:
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{number}: invalid teacher score entry")
            doc, score = item.get("document"), item.get("score")
            if not isinstance(doc, str) or doc not in expected[query] or doc in by_doc:
                raise ValueError(f"{path}:{number}: unknown or duplicate teacher document")
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or abs(score) > float(np.finfo(np.float32).max) or not math.isfinite(score)):
                raise ValueError(f"{path}:{number}: teacher scores must be finite float32 numbers")
            by_doc[doc] = float(score)
        if set(by_doc) != set(expected[query]) or len(by_doc) < 2:
            raise ValueError(f"{path}:{number}: score every candidate, with at least two per query")
        result[query] = by_doc
    if set(result) != set(expected):
        raise ValueError("Teacher score file is missing training queries")
    return result, manifest


@dataclass
class ContrastiveBatch:
    queries: list
    candidates: list
    positive_mask: np.ndarray
    valid_mask: np.ndarray
    teacher_scores: np.ndarray | None = None
    teacher_mask: np.ndarray | None = None


def make_batch(rows, relevant=None, multi_positive=False, use_negatives=True, teacher=None):
    if not rows:
        raise ValueError("Cannot build an empty contrastive batch")
    relevant = known_positives(rows) if relevant is None else relevant
    queries = [row["query"] for row in rows]
    primary = [row["positive"] for row in rows]
    candidates = list(primary)
    if multi_positive:
        for query in queries:
            candidates.extend(sorted(relevant[query]))
    if use_negatives:
        candidates.extend(text for row in rows for text in row.get("negatives", []))
    if teacher is not None:
        candidates.extend(text for query in queries for text in teacher[query]
                          if text not in candidates)
    if multi_positive:
        candidates = list(dict.fromkeys(candidates))
    positive_mask = np.zeros((len(rows), len(candidates)), dtype=bool)
    valid_mask = np.ones_like(positive_mask)
    for i, query in enumerate(queries):
        for j, text in enumerate(candidates):
            if multi_positive:
                positive_mask[i, j] = text in relevant[query]
            else:
                positive_mask[i, j] = j == i
                valid_mask[i, j] = j == i or text not in relevant[query]
    if not np.all(positive_mask.any(axis=1)):
        raise ValueError("Every query needs at least one positive candidate")
    batch = ContrastiveBatch(queries, candidates, positive_mask, valid_mask)
    if teacher is not None:
        batch.teacher_scores = np.zeros(positive_mask.shape, dtype=np.float32)
        batch.teacher_mask = np.zeros_like(positive_mask)
        for i, query in enumerate(queries):
            seen = set()
            for j, text in enumerate(candidates):
                # A document receives teacher probability once, even when the
                # legacy single-positive candidate pool contains duplicates.
                if valid_mask[i, j] and text in teacher[query] and text not in seen:
                    batch.teacher_scores[i, j] = teacher[query][text]
                    batch.teacher_mask[i, j] = True
                    seen.add(text)
            if len(seen) < 2:
                raise ValueError("Distillation needs two valid scored candidates per query; "
                                 "mine negatives or enable --multi-positive")
    return batch
