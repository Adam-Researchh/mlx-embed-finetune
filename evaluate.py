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
from pathlib import Path
from typing import Dict, List, Sequence

import mlx.core as mx
from mlx_embeddings.tokenizer_utils import load_tokenizer
from mlx_embeddings.utils import get_model_path, load_model

DEFAULT_MAX_LENGTH = 512
DEFAULT_RECALL_KS = (1, 3, 5, 10)


def load_pairs(path: str) -> List[Dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "query" not in item or "positive" not in item:
                raise ValueError(f"Line {line_num} must include query and positive")
            pairs.append(item)
    return pairs


def load_encoder(model_name: str):
    model_path = get_model_path(model_name)
    model = load_model(model_path, lazy=False)
    tokenizer = load_tokenizer(model_path)
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
    except (OSError, json.JSONDecodeError):
        return None
    if "query_prefix" not in data and "doc_prefix" not in data:
        return None
    return data.get("query_prefix", ""), data.get("doc_prefix", "")


def l2_normalize(x: mx.array, eps: float = 1e-12) -> mx.array:
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
    for i, item in enumerate(pairs):
        negatives = item.get("negatives") or []
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


def retrieval_scores(
    query_embeds: mx.array,
    pairs: List[Dict],
    corpus_index: Dict[str, int],
    corpus_embeds: mx.array,
    ks: Sequence[int] = DEFAULT_RECALL_KS,
) -> Dict:
    """Rank every query against the whole pooled corpus."""
    sims = query_embeds @ corpus_embeds.T
    mx.eval(sims)
    order = mx.argsort(-sims, axis=1)
    mx.eval(order)
    order_list = order.tolist()

    max_k = max(ks)
    hits = {k: 0 for k in ks}
    reciprocal_ranks = []
    for i, item in enumerate(pairs):
        gold = corpus_index[item["positive"]]
        ranked = order_list[i]
        try:
            rank = ranked.index(gold) + 1
        except ValueError:  # pragma: no cover - gold is always in corpus
            rank = len(ranked) + 1
        for k in ks:
            if rank <= k:
                hits[k] += 1
        reciprocal_ranks.append(1.0 / rank if rank <= 10 else 0.0)

    n = len(pairs)
    return {
        "n": n,
        "corpus_size": corpus_embeds.shape[0],
        "recall_at": {k: hits[k] / n for k in ks},
        "mrr_at_10": sum(reciprocal_ranks) / n,
        "max_k": max_k,
    }


def evaluate_model(model_name: str, pairs: List[Dict], max_length: int, batch_size: int,
                   normalize: bool, query_prefix: str = "", doc_prefix: str = "") -> Dict:
    model, tokenizer = load_encoder(model_name)
    input_kwarg = detect_input_kwarg(model)

    corpus_texts: List[str] = []
    corpus_index: Dict[str, int] = {}
    for item in pairs:
        for text in [item["positive"], *(item.get("negatives") or [])]:
            if text not in corpus_index:
                corpus_index[text] = len(corpus_texts)
                corpus_texts.append(text)

    query_embeds = encode_texts(
        model, tokenizer, apply_prefix([p["query"] for p in pairs], query_prefix),
        max_length, batch_size, normalize, input_kwarg,
    )
    corpus_embeds = encode_texts(
        model, tokenizer, apply_prefix(corpus_texts, doc_prefix),
        max_length, batch_size, normalize, input_kwarg,
    )

    return {
        "model": model_name,
        "query_prefix": query_prefix,
        "doc_prefix": doc_prefix,
        "pairwise": pairwise_scores(query_embeds, pairs, corpus_index, corpus_embeds),
        "retrieval": retrieval_scores(query_embeds, pairs, corpus_index, corpus_embeds),
    }


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
    parser.add_argument("--tuned-model", required=True, help="Fine-tuned model directory")
    parser.add_argument("--eval-pairs", required=True, help="JSONL evaluation file")
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

    pairs = load_pairs(args.eval_pairs)
    normalize = not args.no_normalize
    print(f"Loaded {len(pairs)} evaluation rows from {args.eval_pairs}")

    query_prefix, doc_prefix = args.query_prefix, args.doc_prefix
    if query_prefix is None or doc_prefix is None:
        recorded = read_trained_prefixes(args.tuned_model)
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

    print(f"\nEncoding with base model: {args.base_model}")
    base = evaluate_model(args.base_model, pairs, args.max_length, args.batch_size, normalize,
                          query_prefix, doc_prefix)
    print(f"Encoding with tuned model: {args.tuned_model}")
    tuned = evaluate_model(args.tuned_model, pairs, args.max_length, args.batch_size, normalize,
                           query_prefix, doc_prefix)

    print_report(base, tuned, args.show_rows)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"base": base, "tuned": tuned}, f, indent=2)
        print(f"\nWrote full results to {args.json_out}")


if __name__ == "__main__":
    main()
