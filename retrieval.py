"""Data validation and bounded-memory retrieval metrics (NumPy only)."""

import hashlib
import json

import numpy as np


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            yield number, item


def load_pairs(path):
    pairs = []
    for number, item in read_jsonl(path):
        for key in ("query", "positive"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"{path}:{number}: {key} must be a nonempty string")
        for key in ("negatives", "positives"):
            values = item.get(key, [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{path}:{number}: {key} must be a list of nonempty strings")
        pairs.append(item)
    if not pairs:
        raise ValueError(f"{path}: no usable pairs")
    return pairs


def positive_texts(pair):
    """The designated training target plus any other known relevant documents."""
    return list(dict.fromkeys([pair["positive"], *pair.get("positives", [])]))


def known_positives(pairs):
    relevant = {}
    for pair in pairs:
        relevant.setdefault(pair["query"], set()).update(positive_texts(pair))
    return relevant


def document_id(text):
    """Stable exact-text identity for mined examples and cache provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_corpus(path):
    texts = []
    seen_ids = set()
    for number, item in read_jsonl(path):
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{number}: text must be a nonempty string")
        doc_id = item.get("id", item.get("_id", document_id(text)))
        if not isinstance(doc_id, str) or not doc_id or doc_id in seen_ids:
            raise ValueError(f"{path}:{number}: document IDs must be unique nonempty strings")
        seen_ids.add(doc_id)
        texts.append(text)
    if not texts:
        raise ValueError(f"{path}: empty corpus")
    # The pair format identifies documents by exact text, so duplicates collapse.
    return list(dict.fromkeys(texts))


def pooled_corpus(pairs, extra_texts=()):
    return list(dict.fromkeys([
        *extra_texts,
        *(text for pair in pairs for text in
          [*positive_texts(pair), *pair.get("negatives", [])]),
    ]))


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_embeddings(embeddings):
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Expected a finite embedding matrix with nonzero dimensions")
    # Scaling first prevents squaring large finite components from overflowing.
    scale = np.max(np.abs(x), axis=-1, keepdims=True)
    scaled = x / np.where(scale > 0, scale, 1)
    norm = np.linalg.norm(scaled, axis=-1, keepdims=True)
    return scaled / np.where(norm > 0, norm, 1)


def validate_embedding_pair(query_embeddings, corpus_embeddings, label="Retrieval"):
    q = np.asarray(query_embeddings, dtype=np.float32)
    c = np.asarray(corpus_embeddings, dtype=np.float32)
    if (q.ndim != 2 or c.ndim != 2 or q.shape[1] != c.shape[1]
            or not q.shape[1] or not len(q) or not len(c)):
        raise ValueError(f"{label}: expected nonempty matrices with matching nonzero dimensions")
    if not np.isfinite(q).all() or not np.isfinite(c).all():
        raise ValueError(f"{label} embeddings must be finite")
    if np.any(~np.any(q != 0, axis=1)) or np.any(~np.any(c != 0, axis=1)):
        raise ValueError(f"{label} embeddings contain a zero vector; check the model and pooling")
    return q, c


def positive_integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value > 0


def search_top_k(query_embeddings, corpus_embeddings, k, query_chunk_size=64,
                 corpus_chunk_size=4096):
    """Yield (query offset, indices, scores), tie-broken by corpus input order.

    Stores embeddings and at most a query-chunk x corpus-chunk score matrix;
    never creates the full query x corpus matrix or Python rankings of all docs.
    """
    q, c = validate_embedding_pair(query_embeddings, corpus_embeddings)
    if not all(positive_integer(value) for value in (k, query_chunk_size, corpus_chunk_size)):
        raise ValueError("k and retrieval chunk sizes must be positive integers")
    k = min(k, len(c))
    for start in range(0, len(q), query_chunk_size):
        window = q[start:start + query_chunk_size]
        best_scores = np.empty((len(window), 0), dtype=np.float32)
        best_ids = np.empty((len(window), 0), dtype=np.int64)
        for offset in range(0, len(c), corpus_chunk_size):
            with np.errstate(over="ignore", invalid="ignore"):
                scores = window @ c[offset:offset + corpus_chunk_size].T
            if not np.isfinite(scores).all():
                raise ValueError("Retrieval similarities must be finite; normalize embeddings before scoring")
            ids = np.broadcast_to(np.arange(offset, offset + scores.shape[1]), scores.shape)
            scores = np.concatenate((best_scores, scores), axis=1)
            ids = np.concatenate((best_ids, ids), axis=1)
            order = np.lexsort((ids, -scores), axis=1)[:, :k]
            best_scores = np.take_along_axis(scores, order, axis=1)
            best_ids = np.take_along_axis(ids, order, axis=1)
        yield start, best_ids, best_scores


def retrieval_metrics(query_embeddings, corpus_embeddings, relevant, ks=(1, 3, 5, 10),
                      query_chunk_size=64, corpus_chunk_size=4096):
    """Macro recall, MRR@10, binary nDCG@10; each query may have many positives."""
    query_embeddings, corpus_embeddings = validate_embedding_pair(query_embeddings, corpus_embeddings)
    if not ks or not all(positive_integer(k) for k in ks):
        raise ValueError("Recall cutoffs must be positive integers")
    if len(relevant) != len(query_embeddings) or any(len(gold) == 0 for gold in relevant):
        raise ValueError("Every query must have at least one relevant document")
    if any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
           or i < 0 or i >= len(corpus_embeddings) for gold in relevant for i in gold):
        raise ValueError("Relevant document index must be an integer inside the corpus")
    rows = []
    for start, indices, _ in search_top_k(
        query_embeddings, corpus_embeddings, max(10, *ks), query_chunk_size, corpus_chunk_size
    ):
        for offset, ranked in enumerate(indices):
            gold = set(relevant[start + offset])
            hits = np.asarray([int(i) in gold for i in ranked], dtype=np.float64)
            positions = np.flatnonzero(hits[:10])
            discount = 1.0 / np.log2(np.arange(2, min(10, len(ranked)) + 2))
            ideal = np.sum(1.0 / np.log2(np.arange(2, min(10, len(gold)) + 2)))
            rows.append({
                "recall_at": {k: float(hits[:k].sum() / len(gold)) for k in ks},
                "mrr_at_10": float(1.0 / (positions[0] + 1)) if len(positions) else 0.0,
                "ndcg_at_10": float(np.dot(hits[:10], discount) / ideal),
            })
    return {
        "n": len(rows), "corpus_size": len(corpus_embeddings), "max_k": max(ks),
        "recall_at": {k: float(np.mean([row["recall_at"][k] for row in rows])) for k in ks},
        "mrr_at_10": float(np.mean([row["mrr_at_10"] for row in rows])),
        "ndcg_at_10": float(np.mean([row["ndcg_at_10"] for row in rows])),
        "per_query": rows,
    }
