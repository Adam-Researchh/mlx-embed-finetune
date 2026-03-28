#!/usr/bin/env python3
"""
Compare a base encoder model against a fine-tuned encoder model on a simple
query -> positive vs negative discrimination task.

Input data format (JSONL):
    {"query": "...", "positive": "...", "negatives": ["..."]}

For each example, the script encodes the query, one relevant passage, and one
irrelevant passage, then reports cosine-similarity margins for the base and
fine-tuned models.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
from mlx_embeddings.tokenizer_utils import load_tokenizer
from mlx_embeddings.utils import get_model_path, load_model

DEFAULT_MAX_LENGTH = 512


def load_pairs(path: str) -> List[Dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "query" not in item or "positive" not in item or not item.get("negatives"):
                raise ValueError(f"Line {line_num} must include query, positive, and at least one negative")
            pairs.append(item)
    return pairs


def load_encoder(model_name: str):
    model_path = get_model_path(model_name)
    model = load_model(model_path, lazy=False, path_to_repo=model_name)
    tokenizer = load_tokenizer(model_path)
    return model, tokenizer


def encode_texts(model, tokenizer, texts: List[str], max_length: int) -> mx.array:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    output = model(
        input_ids=mx.array(encoded["input_ids"]),
        attention_mask=mx.array(encoded["attention_mask"]),
    )
    mx.eval(output.text_embeds)
    return output.text_embeds


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    sim = (a @ b.T).item()
    return float(sim)


def score_example(model, tokenizer, item: Dict, max_length: int) -> Dict:
    embeddings = encode_texts(
        model,
        tokenizer,
        [item["query"], item["positive"], item["negatives"][0]],
        max_length=max_length,
    )
    query = embeddings[0:1]
    positive = embeddings[1:2]
    negative = embeddings[2:3]
    positive_score = cosine_similarity(query, positive)
    negative_score = cosine_similarity(query, negative)
    return {
        "positive_score": positive_score,
        "negative_score": negative_score,
        "gap": positive_score - negative_score,
        "correct": positive_score > negative_score,
    }


def evaluate_model(model_name: str, pairs: List[Dict], max_length: int) -> Dict:
    model, tokenizer = load_encoder(model_name)
    rows = []
    for item in pairs:
        result = score_example(model, tokenizer, item, max_length)
        rows.append({
            "query": item["query"],
            **result,
        })
    avg_gap = sum(row["gap"] for row in rows) / len(rows)
    accuracy = sum(1 for row in rows if row["correct"]) / len(rows)
    return {
        "model": model_name,
        "avg_gap": avg_gap,
        "accuracy": accuracy,
        "rows": rows,
    }


def print_report(base_result: Dict, tuned_result: Dict):
    print("=" * 90)
    print("EMBEDDING DISCRIMINATION EVALUATION")
    print("=" * 90)
    print(f"Base model:      {base_result['model']}")
    print(f"Fine-tuned:      {tuned_result['model']}")
    print(f"Base avg gap:    {base_result['avg_gap']:.4f}")
    print(f"Tuned avg gap:   {tuned_result['avg_gap']:.4f}")
    print(f"Gap improvement: {tuned_result['avg_gap'] - base_result['avg_gap']:+.4f}")
    print(f"Base accuracy:   {base_result['accuracy'] * 100:.1f}%")
    print(f"Tuned accuracy:  {tuned_result['accuracy'] * 100:.1f}%")
    print()
    print("Per-query results:")
    print("-" * 90)
    for base_row, tuned_row in zip(base_result["rows"], tuned_result["rows"]):
        print(f"Query: {base_row['query']}")
        print(
            f"  base  -> pos {base_row['positive_score']:.4f} | neg {base_row['negative_score']:.4f} | gap {base_row['gap']:.4f}"
        )
        print(
            f"  tuned -> pos {tuned_row['positive_score']:.4f} | neg {tuned_row['negative_score']:.4f} | gap {tuned_row['gap']:.4f}"
        )
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned embedding models")
    parser.add_argument("--base-model", required=True, help="Base model name or path")
    parser.add_argument("--tuned-model", required=True, help="Fine-tuned model name or path")
    parser.add_argument("--eval-pairs", required=True, help="JSONL file with eval examples")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum token length")
    parser.add_argument("--save-json", default=None, help="Optional path to save the full report as JSON")
    return parser


def main():
    args = build_parser().parse_args()
    pairs = load_pairs(args.eval_pairs)
    base_result = evaluate_model(args.base_model, pairs, args.max_length)
    tuned_result = evaluate_model(args.tuned_model, pairs, args.max_length)
    print_report(base_result, tuned_result)

    if args.save_json:
        payload = {"base": base_result, "tuned": tuned_result}
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON report to {args.save_json}")


if __name__ == "__main__":
    main()
