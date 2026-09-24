#!/usr/bin/env python3
"""Mine training negatives with optional embedding-teacher margin filtering."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from retrieval import (document_id, file_digest, known_positives, load_corpus,
                       load_pairs, pooled_corpus, positive_integer, search_top_k,
                       validate_embedding_pair)


def mine_negatives(pairs, corpus, query_embeddings, corpus_embeddings, num_negatives=3,
                   range_min=0, range_max=100, relative_margin=0.05, absolute_margin=0.0,
                   guide_query_embeddings=None, guide_corpus_embeddings=None,
                   query_chunk_size=64, corpus_chunk_size=4096):
    """Ranks are zero-based in the miner's unfiltered retrieval results.

    A candidate must be below the least-similar known positive by the configured
    margin in the guide space (or miner space when no guide is supplied).
    Shortfalls are reported; filtering is never relaxed to fill a quota.
    """
    if (not positive_integer(num_negatives) or not isinstance(range_min, (int, np.integer))
            or isinstance(range_min, (bool, np.bool_)) or range_min < 0
            or not positive_integer(range_max) or range_max <= range_min):
        raise ValueError("Require positive num_negatives and 0 <= range_min < range_max")
    if any(not math.isfinite(v) or v < 0 for v in (relative_margin, absolute_margin)):
        raise ValueError("Margins must be finite and nonnegative")
    q, c = validate_embedding_pair(query_embeddings, corpus_embeddings)
    if len(pairs) != len(q) or len(corpus) != len(c) or len(set(corpus)) != len(corpus):
        raise ValueError("Embeddings must match pairs and the deduplicated corpus")
    if (guide_query_embeddings is None) != (guide_corpus_embeddings is None):
        raise ValueError("Supply both guide query and corpus embeddings")
    gq, gc = (q, c) if guide_query_embeddings is None else validate_embedding_pair(
        guide_query_embeddings, guide_corpus_embeddings, label="Guide")
    if len(gq) != len(q) or len(gc) != len(c):
        raise ValueError("Guide embeddings must match the query/corpus row counts")

    def guide_score(query_index, doc_index):
        with np.errstate(over="ignore", invalid="ignore"):
            score = float(gq[query_index] @ gc[doc_index])
        if not math.isfinite(score):
            raise ValueError("Guide similarities must be finite")
        return score
    relevant = known_positives(pairs)
    index = {text: i for i, text in enumerate(corpus)}
    if any(text not in index for values in relevant.values() for text in values):
        raise ValueError("Corpus must include every known positive")
    output = []
    stats = {"rows": len(pairs), "selected": 0, "filtered_known_positive": 0,
             "filtered_margin": 0, "rows_with_shortfall": 0}
    for start, rankings, scores in search_top_k(q, c, range_max, query_chunk_size, corpus_chunk_size):
        for offset, (ranked, row_scores) in enumerate(zip(rankings, scores)):
            i = start + offset
            pair = pairs[i]
            gold = relevant[pair["query"]]
            positive_score = min(guide_score(i, index[text]) for text in gold)
            threshold = positive_score - abs(positive_score) * relative_margin - absolute_margin
            if not math.isfinite(threshold):
                raise ValueError("Guide filtering threshold must be finite")
            negatives, details = [], []
            for rank in range(range_min, len(ranked)):
                doc_index = int(ranked[rank])
                text = corpus[doc_index]
                if text in gold:
                    stats["filtered_known_positive"] += 1
                    continue
                candidate_score = guide_score(i, doc_index)
                if candidate_score > threshold:
                    stats["filtered_margin"] += 1
                    continue
                negatives.append(text)
                details.append({"id": document_id(text), "rank": rank,
                                "score": float(row_scores[rank]), "guide_score": candidate_score})
                if len(negatives) == num_negatives:
                    break
            stats["selected"] += len(negatives)
            stats["rows_with_shortfall"] += int(len(negatives) < num_negatives)
            output.append({**pair, "negatives": negatives, "mining": {
                "positive_reference_score": positive_score, "threshold": threshold,
                "negatives": details,
            }})
    return output, stats


def embed(model_name, pairs, corpus, batch_size, max_length, query_prefix, doc_prefix):
    # Import MLX only at the CLI boundary, allowing mining logic tests on Linux.
    import mlx.core as mx
    from evaluate import apply_prefix, detect_input_kwarg, encode_texts, load_encoder

    model, tokenizer = load_encoder(model_name)
    input_kwarg = detect_input_kwarg(model)
    q = encode_texts(model, tokenizer, apply_prefix([p["query"] for p in pairs], query_prefix),
                     max_length, batch_size, True, input_kwarg)
    c = encode_texts(model, tokenizer, apply_prefix(corpus, doc_prefix),
                     max_length, batch_size, True, input_kwarg)
    arrays = np.array(q.astype(mx.float32)), np.array(c.astype(mx.float32))
    del model, tokenizer, q, c
    mx.clear_cache()
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", required=True)
    parser.add_argument("--corpus", help="JSONL with id/text; positives are added automatically")
    parser.add_argument("--model", required=True, help="MLX embedding model used to retrieve candidates")
    parser.add_argument("--guide-model", help="Optional embedding teacher for filtering")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--doc-prefix", default="")
    parser.add_argument("--guide-query-prefix", default="")
    parser.add_argument("--guide-doc-prefix", default="")
    parser.add_argument("--num-negatives", type=int, default=3)
    parser.add_argument("--range-min", type=int, default=0)
    parser.add_argument("--range-max", type=int, default=100)
    parser.add_argument("--relative-margin", type=float, default=0.05)
    parser.add_argument("--absolute-margin", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--corpus-chunk-size", type=int, default=4096)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.max_length, args.num_negatives,
           args.query_chunk_size, args.corpus_chunk_size) < 1:
        parser.error("Batch, length, negative count and chunk sizes must be positive")
    if args.range_min < 0 or args.range_max <= args.range_min:
        parser.error("Require 0 <= --range-min < --range-max")
    if any(not math.isfinite(v) or v < 0 for v in (args.relative_margin, args.absolute_margin)):
        parser.error("Margins must be finite and nonnegative")
    output_path = Path(args.output)
    metadata_path = Path(str(output_path) + ".metadata.json")
    if output_path.exists() or metadata_path.exists():
        parser.error("Output already exists; choose a new output path")
    pairs = load_pairs(args.train_pairs)
    corpus = pooled_corpus(pairs, load_corpus(args.corpus) if args.corpus else [])
    q, c = embed(args.model, pairs, corpus, args.batch_size, args.max_length,
                 args.query_prefix, args.doc_prefix)
    guide_q, guide_c = None, None
    if args.guide_model:
        guide_q, guide_c = embed(args.guide_model, pairs, corpus, args.batch_size, args.max_length,
                                 args.guide_query_prefix, args.guide_doc_prefix)
    mined, stats = mine_negatives(
        pairs, corpus, q, c, args.num_negatives, args.range_min, args.range_max,
        args.relative_margin, args.absolute_margin, guide_q, guide_c,
        args.query_chunk_size, args.corpus_chunk_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        for pair in mined:
            stream.write(json.dumps(pair, ensure_ascii=False, allow_nan=False) + "\n")
    metadata = {"config": vars(args), "stats": stats, "corpus_size": len(corpus),
                "train_sha256": file_digest(args.train_pairs),
                "corpus_sha256": file_digest(args.corpus) if args.corpus else None,
                "output_sha256": file_digest(output_path)}
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps(stats, indent=2))
    if stats["rows_with_shortfall"]:
        print("Some rows have fewer negatives than requested; filters were not relaxed.")


if __name__ == "__main__":
    main()
