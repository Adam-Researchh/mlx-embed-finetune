#!/usr/bin/env python3
"""Score fixed training candidate groups once for offline ranking distillation."""

import argparse
import json
from pathlib import Path

import numpy as np

from retrieval import file_digest, load_pairs, validate_embedding_pair
from training_data import teacher_groups


def score_groups(groups, queries, documents, query_embeddings, document_embeddings):
    q, c = validate_embedding_pair(query_embeddings, document_embeddings, label="Teacher")
    if len(queries) != len(q) or len(documents) != len(c):
        raise ValueError("Teacher embeddings do not match the input text counts")
    index = {text: i for i, text in enumerate(documents)}
    rows = []
    for i, query in enumerate(queries):
        candidates = groups[query]
        if len(candidates) < 2:
            raise ValueError("Teacher ranking needs at least two candidates per query; mine negatives first")
        with np.errstate(over="ignore", invalid="ignore"):
            scores = c[[index[text] for text in candidates]] @ q[i]
        if not np.isfinite(scores).all():
            raise ValueError("Teacher similarity scores must be finite")
        rows.append({"query": query, "scores": [
            {"document": text, "score": float(score)} for text, score in zip(candidates, scores)
        ]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--doc-prefix", default="")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.max_length, args.batch_size) < 1:
        parser.error("Batch size and maximum length must be positive")
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        parser.error("Teacher output already exists; choose a new path")
    digest = file_digest(args.train_pairs)
    pairs = load_pairs(args.train_pairs)
    groups = teacher_groups(pairs)
    if any(len(docs) < 2 for docs in groups.values()):
        parser.error("Teacher ranking needs at least two candidates per query; mine negatives first")
    queries = list(groups)
    documents = list(dict.fromkeys(text for docs in groups.values() for text in docs))

    import mlx.core as mx
    from mlx_embeddings.utils import get_model_path
    from evaluate import apply_prefix, detect_input_kwarg, encode_texts, load_encoder

    model_path = get_model_path(args.model)
    model, tokenizer = load_encoder(str(model_path))
    input_kwarg = detect_input_kwarg(model)
    q = encode_texts(model, tokenizer, apply_prefix(queries, args.query_prefix),
                     args.max_length, args.batch_size, True, input_kwarg)
    c = encode_texts(model, tokenizer, apply_prefix(documents, args.doc_prefix),
                     args.max_length, args.batch_size, True, input_kwarg)
    rows = score_groups(groups, queries, documents,
                        np.array(q.astype(mx.float32)), np.array(c.astype(mx.float32)))
    if file_digest(args.train_pairs) != digest:
        raise ValueError("Training file changed while scoring; rerun on a stable dataset")
    manifest = {"type": "teacher_manifest", "schema_version": 1, "train_sha256": digest,
                "teacher": args.model, "resolved_model_path": str(model_path),
                "score_type": "cosine_similarity", "query_prefix": args.query_prefix,
                "doc_prefix": args.doc_prefix, "max_length": args.max_length}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in [manifest, *rows]:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Saved scores for {len(queries)} queries and {sum(len(d) for d in groups.values())} "
          f"query-document pairs to {output}")


if __name__ == "__main__":
    main()
