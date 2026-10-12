"""Compare plain/autodiff cache forwards with a same-chunk AD reference.

Synthetic tokens test numerical agreement, MLX tensor allocation and warmed
step time. This is not a retrieval-quality or whole-process-memory benchmark.
"""
import argparse
import gc
from importlib.metadata import version
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

import train
from grad_cache import cached_value_and_grad, capture_rng, restore_rng
from objectives import embedding_objective
from training_data import make_batch, teacher_groups


TEXT_ROWS = [
    {"query": "short", "positive": "answer one", "positives": ["second answer"], "negatives": ["wrong one"]},
    {"query": "longer question", "positive": "second answer", "negatives": ["another wrong answer"]},
    {"query": "short", "positive": "third answer", "negatives": ["wrong one", "third wrong"]},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/nomicai-modernbert-embed-base-4bit")
    parser.add_argument("--fixture", choices=["synthetic", "ragged"], default="synthetic")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--loss-chunk-size", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.queries, args.sequence_length, args.chunk_size, args.loss_chunk_size, args.repeats) < 1:
        parser.error("Batch, chunk, sequence and repeat counts must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("Dropout must be in [0, 1)")
    destination = Path(args.output)
    if destination.exists() or destination.is_symlink():
        parser.error("Choose a new output path")
    records = []
    for seed in args.seeds:
        mx.random.seed(seed)
        model_path = train.get_model_path(args.model)
        model = train.load_model(model_path, lazy=False)
        train.apply_lora_to_model(model, rank=2, dropout=args.dropout)
        train.freeze_base_and_enable_lora(model)
        for _, module in model.named_modules():
            if isinstance(module, train.LoRALinear):
                module.lora_b = mx.random.normal(module.lora_b.shape) * .002
        model.train()
        kwarg = train.detect_input_kwarg(model)
        rng = np.random.default_rng(seed)
        nq, nc = args.queries, 4 * args.queries
        def tokens(count):
            return {"input_ids": mx.array(rng.integers(100, 1000, (count, args.sequence_length))),
                    "attention_mask": mx.ones((count, args.sequence_length), dtype=mx.int32)}
        def chunks(tokens, count):
            return [{k: v[i:i+args.chunk_size] for k, v in tokens.items()}
                    for i in range(0, count, args.chunk_size)]
        query_chunks, candidate_chunks = chunks(tokens(nq), nq), chunks(tokens(nc), nc)
        positives = np.zeros((nq, nc), bool)
        scores, mask = np.zeros((nq, nc), np.float32), np.zeros((nq, nc), bool)
        for i in range(nq):
            positives[i, [i, nq+i]] = True
            mask[i, [i, nq+i, 2*nq+i, 3*nq+i]] = True
            scores[i, [i, nq+i, 2*nq+i, 3*nq+i]] = [.9, .8, .3, .1]
        if args.fixture == "ragged":
            # Fixed variable-length text fixture for the original discrepancy.
            # --queries controls only the synthetic fixture.
            teacher = {q: {doc: i / 10 for i, doc in enumerate(docs)}
                       for q, docs in teacher_groups(TEXT_ROWS).items()}
            batch = make_batch(TEXT_ROWS, multi_positive=True, teacher=teacher)
            tokenizer = train.load_tokenizer(model_path)
            def text_chunks(texts):
                return [train.tokenize_batch(tokenizer, texts[i:i+args.chunk_size], args.sequence_length)
                        for i in range(0, len(texts), args.chunk_size)]
            query_chunks, candidate_chunks = text_chunks(batch.queries), text_chunks(batch.candidates)
            nq, nc = len(batch.queries), len(batch.candidates)
            positives, scores, mask = batch.positive_mask, batch.teacher_scores, batch.teacher_mask
        positives, scores, mask = map(mx.array, (positives, scores, mask))
        valid = mx.ones((nq, nc), dtype=mx.bool_)
        def encode(m, t):
            return train.encode_texts(m, **t, input_kwarg=kwarg)
        def objective(q, c, start, stop):
            return embedding_objective(q, c, positives[start:stop], valid[start:stop],
                                       temperature=.3, hardness=2, dims=[128, 64],
                                       teacher_scores=scores[start:stop], teacher_mask=mask[start:stop],
                                       distill_weight=.2, distill_temperature=.2)
        def reference(m):
            q = mx.concatenate([encode(m, t) for t in query_chunks])
            c = mx.concatenate([encode(m, t) for t in candidate_chunks])
            return objective(q, c, 0, nq)
        ref_fn = nn.value_and_grad(model, reference)
        mx.eval(model.parameters(), query_chunks, candidate_chunks, positives, valid, scores, mask)
        initial_rng = capture_rng()
        def run(mode):
            restore_rng(initial_rng)
            if mode == "reference":
                value, grads = ref_fn(model)
            else:
                value, grads = cached_value_and_grad(model, encode, query_chunks, candidate_chunks,
                                                    objective, args.loss_chunk_size, mode)
            mx.eval(value, grads)
            return value, grads
        # Warm each mode once. Then discard its graphs before measuring.
        for mode in ("reference", "plain", "autodiff"):
            warm = run(mode)
            del warm
        measured = {}
        order = ("reference", "plain", "autodiff") if seed % 2 else ("autodiff", "plain", "reference")
        for mode in order:
            times, peaks = [], []
            for _ in range(args.repeats):
                gc.collect()
                mx.clear_cache()
                mx.reset_peak_memory()
                started = time.perf_counter()
                value, grads = run(mode)
                times.append(time.perf_counter() - started)
                peaks.append(mx.get_peak_memory())
                # Host copies happen outside the timed MLX step.
                vector = np.concatenate([np.asarray(g, dtype=np.float32).ravel()
                                         for _, g in tree_flatten(grads)]).astype(np.float64)
                loss = value.item()
                end_rng = capture_rng()
                del value, grads
            if not np.isfinite(vector).all() or not np.isfinite(loss):
                raise FloatingPointError("Nonfinite benchmark result")
            measured[mode] = {"loss": loss, "gradient": vector, "end_rng": end_rng,
                              "seconds": times, "peak_mlx_bytes": peaks}
        ref = measured["reference"]
        denom = np.linalg.norm(ref["gradient"])
        if denom == 0:
            raise ValueError("Zero reference gradient makes relative error undefined")
        for mode, result in measured.items():
            vector = result.pop("gradient") if mode != "reference" else result["gradient"]
            record = {"seed": seed, "mode": mode, "loss": result["loss"],
                      "loss_absolute_error": abs(result["loss"] - ref["loss"]),
                      "gradient_relative_l2": float(np.linalg.norm(vector-ref["gradient"]) / denom),
                      "gradient_cosine": float(np.dot(vector,ref["gradient"]) / np.linalg.norm(vector) / denom),
                      "same_rng": result["end_rng"] == ref["end_rng"],
                      "seconds": result["seconds"], "peak_mlx_bytes": result["peak_mlx_bytes"]}
            records.append(record)
            print(json.dumps(record), flush=True)
        # Drop last loop references as well as model closures before next seed.
        del model, ref_fn, measured, ref, result, vector
        gc.collect()
        mx.clear_cache()
    report = {"config": vars(args), "records": records,
              "model_revision": model_path.name if model_path.parent.name == "snapshots" else None,
              "versions": {package: version(package) for package in ["mlx", "mlx-lm", "mlx-embeddings", "numpy"]},
              "kind": "synthetic forward/backward; same encoding chunks and RNG; no optimizer or retrieval evaluation"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
