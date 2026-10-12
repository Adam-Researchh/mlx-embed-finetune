"""Measure a synthetic contrastive forward/backward, including model memory.

Run uncached and cached modes in separate processes. This measures MLX tensor
allocations and one-pass latency, not retrieval quality or steady-state speed.
"""
import argparse
import json
import time
from importlib.metadata import version

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import train
from grad_cache import cached_value_and_grad
from objectives import embedding_objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/all-MiniLM-L6-v2-bf16")
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--cache-chunk-size", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.queries, args.sequence_length) < 1 or args.cache_chunk_size < 0:
        parser.error("Invalid batch, sequence length or cache chunk size")
    from pathlib import Path
    if Path(args.output).exists():
        parser.error("Choose a new output file")
    mx.random.seed(42)
    path = train.get_model_path(args.model)
    model = train.load_model(path, lazy=False)
    train.apply_lora_to_model(model, rank=8)
    train.freeze_base_and_enable_lora(model)
    # Autograd works in eval mode. Disable both backbone and LoRA dropout so
    # changing batch shapes does not change random masks between processes.
    model.eval()
    kwarg = train.detect_input_kwarg(model)
    b = args.queries
    # Fixed low-vocabulary token IDs work with the documented MiniLM recipe.
    rng = np.random.default_rng(42)
    q = {"input_ids": mx.array(rng.integers(100, 1000, (b, args.sequence_length))),
         "attention_mask": mx.ones((b, args.sequence_length), dtype=mx.int32)}
    c = {"input_ids": mx.array(rng.integers(100, 1000, (4 * b, args.sequence_length))),
         "attention_mask": mx.ones((4 * b, args.sequence_length), dtype=mx.int32)}
    positives = np.zeros((b, 4 * b), dtype=bool)
    scores, scored = np.zeros((b, 4 * b), np.float32), np.zeros((b, 4 * b), bool)
    for i in range(b):
        positives[i, [i, b + i]] = True
        scored[i, [i, b + i, 2 * b + i, 3 * b + i]] = True
        scores[i, [i, b + i, 2 * b + i, 3 * b + i]] = [.9, .8, .3, .1]
    positive_mask, valid = mx.array(positives), mx.ones((b, 4 * b), dtype=mx.bool_)
    teacher_scores, teacher_mask = mx.array(scores), mx.array(scored)

    def encode(m, tokens):
        return train.encode_texts(m, **tokens, input_kwarg=kwarg)

    def objective(qe, ce, start, stop):
        return embedding_objective(qe, ce, positive_mask[start:stop], valid[start:stop],
                                   teacher_scores=teacher_scores[start:stop],
                                   teacher_mask=teacher_mask[start:stop], distill_weight=.2)

    mx.eval(model.parameters(), q, c, positive_mask, valid, teacher_scores, teacher_mask)
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter()
    if args.cache_chunk_size:
        size = args.cache_chunk_size
        chunks = lambda tokens, n: [{key: value[i:i + size] for key, value in tokens.items()}
                                    for i in range(0, n, size)]
        loss, grads = cached_value_and_grad(model, encode, chunks(q, b), chunks(c, 4 * b), objective, 8)
    else:
        loss, grads = nn.value_and_grad(model, lambda m: objective(encode(m, q), encode(m, c), 0, b))(model)
    value, norm = train.validate_training_step(loss, grads)
    result = {"config": vars(args), "loss": value, "gradient_norm": norm,
              "peak_mlx_bytes": mx.get_peak_memory(), "seconds": time.perf_counter() - start,
              "mlx_version": version("mlx"), "dropout_enabled": False,
              "kind": "synthetic forward/backward; no optimizer update or quality benchmark"}
    with open(args.output, "x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
