#!/usr/bin/env python3
"""
Compare a base encoder model against a fine-tuned encoder model on a retrieval
style evaluation.

Input data format (JSONL):
    {"query": "...", "positive": "...", "negatives": ["...", "..."]}

Two views are reported:

1. Pairwise discrimination — for each row, does the positive outrank its own
   hard negatives? Reports accuracy and the mean positive-minus-negative
   cosine margin.
2. Corpus retrieval — every positive and every negative from the whole file is
   pooled into one corpus, then each query is ranked against all of it.
   Reports Recall@k and MRR@10. This is much closer to how an embedding model
   is actually used than a two-document comparison, and it is where a
   fine-tune that only memorized easy contrasts will show its weakness.
"""

import argparse
import inspect
import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Sequence

import mlx.core as mx
import numpy as np

from retrieval import (file_digest, known_positives, load_corpus, load_pairs,
                       normalize_embeddings, pooled_corpus, retrieval_metrics)
from mlx_embeddings.tokenizer_utils import load_tokenizer
from mlx_embeddings.utils import get_model_path, load_model

DEFAULT_MAX_LENGTH = 512
DEFAULT_RECALL_KS = (1, 3, 5, 10)


def load_encoder(model_name: str):
    model_path = get_model_path(model_name)
    model = load_model(model_path, lazy=False)
    tokenizer = load_tokenizer(model_path)
    model.eval()
    return model, tokenizer


def detect_input_kwarg(model) -> str:
    """Encoder models take `input_ids`; decoder-style embedders take `inputs`."""
    try:
        params = inspect.signature(model.__call__).parameters
    except (TypeError, ValueError):
        return "input_ids"
    for candidate in ("input_ids", "inputs", "input_tokens"):
        if candidate in params:
            return candidate
    return "input_ids"


def apply_prefix(texts: Sequence[str], prefix: str) -> List[str]:
    """Prepend an instruction prefix. Must match what the model was trained with."""
    if not prefix:
        return list(texts)
    return [prefix + t for t in texts]


def read_trained_prefixes(model_dir: str):
    """Recover the prefixes a fine-tuned export was trained with, if recorded."""
    meta = Path(model_dir) / "training_metadata.json"
    if not meta.exists():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid training metadata at {meta}: {exc}") from exc
    if not isinstance(data, dict) or any(
        key in data and not isinstance(data[key], str) for key in ("query_prefix", "doc_prefix")
    ):
        raise ValueError(f"Invalid training metadata at {meta}: prefixes must be strings")
    if "query_prefix" not in data and "doc_prefix" not in data:
        return None
    return data.get("query_prefix", ""), data.get("doc_prefix", "")


def l2_normalize(x: mx.array, eps: float = 1e-12) -> mx.array:
    x = x.astype(mx.float32)
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), eps)


def encode_texts(
    model,
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    batch_size: int = 32,
    normalize: bool = True,
    input_kwarg: str = "input_ids",
) -> mx.array:
    """Encode texts in batches and return a single stacked array.

    Normalization is explicit so that a dot product IS cosine similarity, for
    any encoder — including converted models that do not normalize their own
    output.
    """
    if batch_size < 1 or max_length < 1 or not texts:
        raise ValueError("Nonempty texts, positive batch size and max length required")
    chunks = []
    for i in range(0, len(texts), batch_size):
        window = list(texts[i : i + batch_size])
        encoded = tokenizer(
            window,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        output = model(
            **{input_kwarg: mx.array(encoded["input_ids"])},
            attention_mask=mx.array(encoded["attention_mask"]),
        )
        embeds = output.text_embeds
        if normalize:
            embeds = l2_normalize(embeds)
        mx.eval(embeds)
        chunks.append(embeds)
    return mx.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]


def pairwise_scores(query_embeds: mx.array, pairs: List[Dict], corpus_index: Dict[str, int], corpus_embeds: mx.array) -> Dict:
    correct = 0
    total = 0
    margins = []
    rows = []
    relevant = known_positives(pairs)
    for i, item in enumerate(pairs):
        negatives = [text for text in item.get("negatives", [])
                     if text not in relevant[item["query"]]]
        if not negatives:
            continue
        q = query_embeds[i : i + 1]
        pos = corpus_embeds[corpus_index[item["positive"]] : corpus_index[item["positive"]] + 1]
        pos_score = float((q @ pos.T).item())
        neg_scores = []
        for neg in negatives:
            n = corpus_embeds[corpus_index[neg] : corpus_index[neg] + 1]
            neg_scores.append(float((q @ n.T).item()))
        hardest = max(neg_scores)
        is_correct = pos_score > hardest
        correct += int(is_correct)
        total += 1
        margins.append(pos_score - hardest)
        rows.append(
            {
                "query": item["query"],
                "positive_score": round(pos_score, 4),
                "hardest_negative_score": round(hardest, 4),
                "margin": round(pos_score - hardest, 4),
                "correct": is_correct,
            }
        )
    return {
        "n": total,
        "accuracy": (correct / total) if total else None,
        "mean_margin": (sum(margins) / len(margins)) if margins else None,
        "rows": rows,
    }


def retrieval_scores(query_embeds, pairs, corpus_index, corpus_embeds,
                     ks=DEFAULT_RECALL_KS, query_chunk_size=64, corpus_chunk_size=4096):
    relevant = known_positives(pairs)
    gold = [{corpus_index[text] for text in relevant[pair["query"]]} for pair in pairs]
    return retrieval_metrics(
        np.asarray(query_embeds, dtype=np.float32),
        np.asarray(corpus_embeds, dtype=np.float32), gold, ks,
        query_chunk_size, corpus_chunk_size,
    )


def evaluate_loaded_model(model, tokenizer, pairs, max_length, batch_size,
                          normalize=True, query_prefix="", doc_prefix="",
                          corpus_texts=None, dims=None, query_chunk_size=64,
                          corpus_chunk_size=4096):
    """Also used during training to select checkpoints by retrieval quality."""
    input_kwarg = detect_input_kwarg(model)
    corpus_texts = pooled_corpus(pairs, corpus_texts or [])
    corpus_index = {text: i for i, text in enumerate(corpus_texts)}
    query_embeds = encode_texts(
        model, tokenizer, apply_prefix([p["query"] for p in pairs], query_prefix),
        max_length, batch_size, normalize, input_kwarg,
    )
    corpus_embeds = encode_texts(
        model, tokenizer, apply_prefix(corpus_texts, doc_prefix),
        max_length, batch_size, normalize, input_kwarg,
    )
    full_dim = query_embeds.shape[1]
    if dims and any(d < 1 or d > full_dim for d in dims):
        raise ValueError(f"Requested dimensions must be between 1 and {full_dim}")
    q = np.array(query_embeds.astype(mx.float32))
    c = np.array(corpus_embeds.astype(mx.float32))
    by_dimension = {}
    for dim in dict.fromkeys([full_dim, *(dims or [])]):
        q_dim, c_dim = q, c
        if dim != full_dim:
            q_dim = normalize_embeddings(q[:, :dim])
            c_dim = normalize_embeddings(c[:, :dim])
        by_dimension[str(dim)] = retrieval_scores(
            q_dim, pairs, corpus_index, c_dim,
            query_chunk_size=query_chunk_size, corpus_chunk_size=corpus_chunk_size,
        )
    return {
        "query_prefix": query_prefix, "doc_prefix": doc_prefix,
        "embedding_dim": full_dim,
        "pairwise": pairwise_scores(query_embeds, pairs, corpus_index, corpus_embeds),
        "retrieval": by_dimension[str(full_dim)], "dimensions": by_dimension,
    }


def evaluate_model(model_name, pairs, max_length, batch_size, normalize,
                   query_prefix="", doc_prefix="", **kwargs):
    start = time.perf_counter()
    model, tokenizer = load_encoder(model_name)
    result = evaluate_loaded_model(model, tokenizer, pairs, max_length, batch_size,
                                   normalize, query_prefix, doc_prefix, **kwargs)
    result.update(model=model_name, elapsed_seconds=time.perf_counter() - start)
    return result


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_delta(new, old, digits=4):
    if new is None or old is None:
        return "n/a"
    return f"{new - old:+.{digits}f}"


def print_report(base: Dict, tuned: Dict, show_rows: bool) -> None:
    print("\n" + "=" * 68)
    print("RETRIEVAL (queries ranked against the full pooled corpus)")
    print("=" * 68)
    br, tr = base["retrieval"], tuned["retrieval"]
    print(f"Queries: {br['n']}   Corpus documents: {br['corpus_size']}")
    print(f"{'metric':<12}{'base':>12}{'tuned':>12}{'delta':>12}")
    for k in sorted(br["recall_at"]):
        b, t = br["recall_at"][k], tr["recall_at"][k]
        print(f"{'recall@' + str(k):<12}{b:>12.4f}{t:>12.4f}{t - b:>+12.4f}")
    print(f"{'mrr@10':<12}{br['mrr_at_10']:>12.4f}{tr['mrr_at_10']:>12.4f}{tr['mrr_at_10'] - br['mrr_at_10']:>+12.4f}")

    print(f"{'ndcg@10':<12}{br['ndcg_at_10']:>12.4f}{tr['ndcg_at_10']:>12.4f}{tr['ndcg_at_10'] - br['ndcg_at_10']:>+12.4f}")
    for dim in sorted(base["dimensions"].keys() & tuned["dimensions"].keys(), key=int):
        if int(dim) == base["embedding_dim"] == tuned["embedding_dim"]:
            continue
        b = base["dimensions"][dim]["ndcg_at_10"]
        t = tuned["dimensions"][dim]["ndcg_at_10"]
        print(f"{dim}-dim nDCG@10: {b:.4f} -> {t:.4f} ({t - b:+.4f})")

    print("\n" + "=" * 68)
    print("PAIRWISE (positive vs its own hardest negative)")
    print("=" * 68)
    bp, tp = base["pairwise"], tuned["pairwise"]
    if bp["n"] == 0:
        print("No rows with negatives — pairwise view skipped.")
    else:
        print(f"Rows with negatives: {bp['n']}")
        print(f"{'metric':<14}{'base':>12}{'tuned':>12}{'delta':>12}")
        print(f"{'accuracy':<14}{fmt(bp['accuracy']):>12}{fmt(tp['accuracy']):>12}"
              f"{fmt_delta(tp['accuracy'], bp['accuracy']):>12}")
        print(f"{'mean margin':<14}{fmt(bp['mean_margin']):>12}{fmt(tp['mean_margin']):>12}"
              f"{fmt_delta(tp['mean_margin'], bp['mean_margin']):>12}")

    if show_rows and bp["n"]:
        print("\nPer-query margins (base -> tuned):")
        for b_row, t_row in zip(bp["rows"], tp["rows"]):
            flag = "" if t_row["correct"] else "  <-- tuned incorrect"
            print(f"  {b_row['query'][:52]:<54} {b_row['margin']:>+8.4f} -> {t_row['margin']:>+8.4f}{flag}")


def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned MLX embedding models")
    parser.add_argument("--base-model", required=True, help="Base model path or repository")
    parser.add_argument("--tuned-model", help="Optional tuned model; omit for a baseline-only run")
    parser.add_argument("--eval-pairs", required=True, help="JSONL evaluation file")
    parser.add_argument("--corpus", help="Additional corpus JSONL with id/text objects")
    parser.add_argument("--dims", help="Comma-separated Matryoshka dimensions to evaluate")
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--corpus-chunk-size", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum token length")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    parser.add_argument("--no-normalize", action="store_true", help="Skip explicit L2 normalization")
    parser.add_argument("--query-prefix", default=None,
                        help="Query instruction prefix. Defaults to whatever the tuned model's "
                             "training_metadata.json recorded, so evaluation matches training.")
    parser.add_argument("--doc-prefix", default=None, help="Document instruction prefix")
    parser.add_argument("--show-rows", action="store_true", help="Print per-query detail")
    parser.add_argument("--json-out", default=None, help="Optional path to write full results as JSON")
    args = parser.parse_args()

    if args.json_out and (Path(args.json_out).exists() or Path(args.json_out).is_symlink()):
        parser.error("--json-out already exists; choose a new report path")
    if min(args.batch_size, args.max_length, args.query_chunk_size, args.corpus_chunk_size) < 1:
        parser.error("Batch size, max length and chunk sizes must be positive")
    try:
        dims = [int(d) for d in args.dims.split(",")] if args.dims else None
        if dims and min(dims) < 1:
            raise ValueError
    except ValueError:
        parser.error("--dims must be comma-separated positive integers")
    pairs = load_pairs(args.eval_pairs)
    normalize = not args.no_normalize
    print(f"Loaded {len(pairs)} evaluation rows from {args.eval_pairs}")

    query_prefix, doc_prefix = args.query_prefix, args.doc_prefix
    if query_prefix is None or doc_prefix is None:
        recorded = read_trained_prefixes(args.tuned_model) if args.tuned_model else None
        if recorded:
            if query_prefix is None:
                query_prefix = recorded[0]
            if doc_prefix is None:
                doc_prefix = recorded[1]
            print(f"Using prefixes recorded by the tuned model: "
                  f"query={query_prefix!r} doc={doc_prefix!r}")
    query_prefix = query_prefix or ""
    doc_prefix = doc_prefix or ""
    if query_prefix or doc_prefix:
        print("Both models are scored with the same prefixes, so the comparison stays fair.")

    options = dict(corpus_texts=load_corpus(args.corpus) if args.corpus else None,
                   dims=dims, query_chunk_size=args.query_chunk_size,
                   corpus_chunk_size=args.corpus_chunk_size)
    print(f"\nEncoding with base model: {args.base_model}")
    base = evaluate_model(args.base_model, pairs, args.max_length, args.batch_size, normalize,
                          query_prefix, doc_prefix, **options)
    report = {"base": base, "config": vars(args), "provenance": {
        "eval_sha256": file_digest(args.eval_pairs),
        "corpus_sha256": file_digest(args.corpus) if args.corpus else None,
        "versions": {package: version(package) for package in
                     ("mlx", "mlx-embeddings", "mlx-lm", "numpy", "transformers")},
    }}
    if args.tuned_model:
        print(f"Encoding with tuned model: {args.tuned_model}")
        tuned = evaluate_model(args.tuned_model, pairs, args.max_length, args.batch_size, normalize,
                               query_prefix, doc_prefix, **options)
        report["tuned"] = tuned
        print_report(base, tuned, args.show_rows)
    else:
        for dim, metrics in base["dimensions"].items():
            print(f"{dim} dimensions: nDCG@10={metrics['ndcg_at_10']:.4f}, "
                  f"MRR@10={metrics['mrr_at_10']:.4f}, recall={metrics['recall_at']}")
    if args.json_out:
        serialized = json.dumps(report, indent=2, allow_nan=False)
        with open(args.json_out, "x", encoding="utf-8") as f:
            f.write(serialized + "\n")
        print(f"\nWrote full results to {args.json_out}")


if __name__ == "__main__":
    main()
