"""Fixed-budget SciFact pilot; compare cache numerics and final retrieval.

This research harness reuses production batching, objectives, GradCache,
optimizer helpers and retrieval metrics. It never changes a serving model.
"""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten
import numpy as np

import evaluate
import train
from grad_cache import cached_value_and_grad, capture_rng, restore_rng
from objectives import embedding_objective
from retrieval import file_digest, known_positives, load_corpus, load_pairs, retrieval_metrics
from training_data import make_batch


PROTOCOL = {
    "model": "mlx-community/nomicai-modernbert-embed-base-4bit",
    "revision": "50ba8b0351a2a795299e15946e6c82e918bf38d5",
    "batch_size": 16, "cache_chunk_size": 4, "loss_chunk_size": 8,
    "max_length": 128, "epochs": 1, "lora_rank": 8, "lora_alpha": 16.,
    "lora_dropout": .1, "learning_rate": 2e-5, "weight_decay": .01,
    "temperature": .05, "query_prefix": "search_query: ", "doc_prefix": "search_document: ",
    "evaluation_batch_size": 32, "multi_positive": True,
}


def get_model(seed, float32=False):
    from huggingface_hub import snapshot_download
    mx.random.seed(seed)
    path = Path(snapshot_download(PROTOCOL["model"], revision=PROTOCOL["revision"], local_files_only=True,
                                  allow_patterns=["*.json", "*.safetensors", "*.py", "*.tiktoken", "*.txt", "*.model"]))
    model = train.load_model(path, lazy=False)
    tokenizer = train.load_tokenizer(path)
    if float32:
        model.set_dtype(mx.float32)
    train.apply_lora_to_model(model, rank=PROTOCOL["lora_rank"], alpha=PROTOCOL["lora_alpha"],
                              dropout=PROTOCOL["lora_dropout"])
    train.freeze_base_and_enable_lora(model)
    mx.eval(model.parameters())
    return model, tokenizer


def gradient_vector(grads):
    return np.concatenate([np.array(g).ravel().astype(np.float64) for _, g in tree_flatten(grads)])


def adapter_fingerprint(model):
    h = hashlib.sha256()
    for name, value in sorted(train.lora_weight_dict(model).items()):
        h.update(name.encode())
        h.update(np.array(value).tobytes())
    return h.hexdigest()


def gradient_check(model, reference, grads, loss, before, after):
    actual = gradient_vector(grads)
    restore_rng(before)
    try:
        target_loss, target_grads = nn.value_and_grad(model, reference)(model)
        mx.eval(target_loss, target_grads)
        expected = gradient_vector(target_grads)
        denom = np.linalg.norm(expected)
        record = {"relative_l2": float(np.linalg.norm(actual-expected) / max(denom, 1e-30)),
                  "reference_gradient_norm": float(denom),
                  "loss_absolute_error": abs(loss.item() - target_loss.item()),
                  "same_rng": capture_rng() == after}
        if not record["same_rng"]:
            raise RuntimeError("Gradient diagnostic changed expected RNG progression")
        return record
    finally:
        restore_rng(after)


def fit(data, out, arm, seed):
    rows = load_pairs(data / "train.jsonl")
    relevant = known_positives(rows)
    random.Random(seed).shuffle(rows)
    model, tokenizer = get_model(seed, float32=arm == "float32")
    model.train()
    kwarg = train.detect_input_kwarg(model)
    initial_hash = adapter_fingerprint(model)
    mx.save_safetensors(str(out / "initial_adapters.safetensors"), train.lora_weight_dict(model))
    steps = math.ceil(len(rows) / PROTOCOL["batch_size"])
    warmup = max(1, int(steps * .1))
    warm = opt.schedulers.linear_schedule(PROTOCOL["learning_rate"] / warmup,
                                         PROTOCOL["learning_rate"], max(1, warmup-1))
    cosine = opt.schedulers.cosine_decay(PROTOCOL["learning_rate"], max(1, steps-warmup))
    optimizer = opt.AdamW(learning_rate=opt.schedulers.join_schedules([warm, cosine], [warmup]),
                         weight_decay=PROTOCOL["weight_decay"])
    diagnostics_at = {1, math.ceil(steps / 2), steps}
    history = []
    training_started = time.perf_counter()
    with (out / "training.jsonl").open("x") as stream:
        for step, offset in enumerate(range(0, len(rows), PROTOCOL["batch_size"]), 1):
            mx.reset_peak_memory()
            started = time.perf_counter()
            batch = make_batch(rows[offset:offset+PROTOCOL["batch_size"]], relevant, multi_positive=True)
            def chunks(texts, prefix):
                texts = train.apply_prefix(texts, prefix)
                size = PROTOCOL["cache_chunk_size"]
                return [train.tokenize_batch(tokenizer, texts[i:i+size], PROTOCOL["max_length"])
                        for i in range(0, len(texts), size)]
            qc, cc = chunks(batch.queries, PROTOCOL["query_prefix"]), chunks(batch.candidates, PROTOCOL["doc_prefix"])
            positives, valid = mx.array(batch.positive_mask), mx.array(batch.valid_mask)
            def encode(m, tokens):
                return train.encode_texts(m, **tokens, input_kwarg=kwarg)
            def objective(q, c, start, stop):
                return embedding_objective(q, c, positives[start:stop], valid[start:stop], PROTOCOL["temperature"])
            def reference(m):
                q = mx.concatenate([encode(m, t) for t in qc])
                c = mx.concatenate([encode(m, t) for t in cc])
                return objective(q, c, 0, len(batch.queries))
            before = capture_rng()
            if arm == "reference":
                loss, grads = nn.value_and_grad(model, reference)(model)
            else:
                mode = "autodiff" if arm == "autodiff" else "plain"
                loss, grads = cached_value_and_grad(model, encode, qc, cc, objective,
                                                    PROTOCOL["loss_chunk_size"], mode)
            loss_value, grad_norm = train.validate_training_step(loss, grads, require_nonzero=True)
            elapsed = time.perf_counter() - started
            peak = mx.get_peak_memory()
            after = capture_rng()
            diagnostic = None
            if step in diagnostics_at:
                diagnostic = gradient_check(model, reference, grads, loss, before, after)
                gc.collect()
                mx.clear_cache()
            mx.reset_peak_memory()
            started = time.perf_counter()
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            elapsed += time.perf_counter() - started
            peak = max(peak, mx.get_peak_memory())
            record = {"step": step, "loss": loss_value, "gradient_norm": grad_norm,
                      "learning_rate": optimizer.learning_rate.item(), "training_seconds": elapsed,
                      "peak_mlx_bytes": peak, "rng_before": before, "rng_after": after,
                      "queries": len(batch.queries), "candidates": len(batch.candidates),
                      "gradient_diagnostic": diagnostic}
            history.append(record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            if step == 1 or step % 10 == 0 or step == steps:
                print(json.dumps({"arm": arm, "seed": seed, **record}), flush=True)
            if step in diagnostics_at:
                mx.save_safetensors(str(out / f"adapters-step-{step}.safetensors"), train.lora_weight_dict(model))
            del loss, grads
    mx.save_safetensors(str(out / "adapters.safetensors"), train.lora_weight_dict(model))
    return {"initial_adapter_sha256": initial_hash, "steps": steps,
            "order_sha256": hashlib.sha256(json.dumps([r["query_id"] for r in rows]).encode()).hexdigest(),
            "timed_training_seconds": sum(r["training_seconds"] for r in history),
            "training_wall_seconds_with_diagnostics": time.perf_counter() - training_started,
            "peak_mlx_bytes_excluding_diagnostics": max(r["peak_mlx_bytes"] for r in history),
            "final_loss": history[-1]["loss"],
            "diagnostics": [{"step": r["step"], **r["gradient_diagnostic"]} for r in history if r["gradient_diagnostic"]],
            "history": history}


def evaluate_final(data, out, seed, baseline=False):
    # Common serving precision for every arm, including float32 training.
    model, tokenizer = get_model(seed, float32=False)
    if not baseline:
        train.load_lora_checkpoint(model, out)
    model.eval()
    kwarg = train.detect_input_kwarg(model)
    corpus = load_corpus(data / "corpus.jsonl")
    index = {text: i for i, text in enumerate(corpus)}
    def encode(texts, prefix):
        return np.array(evaluate.encode_texts(model, tokenizer, evaluate.apply_prefix(texts, prefix),
                                             PROTOCOL["max_length"], PROTOCOL["evaluation_batch_size"],
                                             True, kwarg).astype(mx.float32))
    started = time.perf_counter()
    documents = encode(corpus, PROTOCOL["doc_prefix"])
    results = {}
    for split in ("validation", "test"):
        rows = load_pairs(data / f"{split}.jsonl")
        queries = encode([r["query"] for r in rows], PROTOCOL["query_prefix"])
        relevant = known_positives(rows)
        gold = [{index[t] for t in relevant[r["query"]]} for r in rows]
        result = retrieval_metrics(queries, documents, gold)
        result["query_ids"] = [r["query_id"] for r in rows]
        results[split] = result
        print(json.dumps({"split": split, **{k: v for k, v in result.items() if k not in ("per_query", "query_ids")}}), flush=True)
    results["seconds"] = time.perf_counter() - started
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", choices=["baseline", "reference", "plain", "autodiff", "float32"], required=True)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    data, out = Path(args.data_dir), Path(args.output_dir)
    if out.exists():
        parser.error("Choose a new output directory")
    manifest = json.loads((data / "manifest.json").read_text())
    for name, digest in manifest["file_sha256"].items():
        if file_digest(data / name) != digest:
            raise ValueError(f"Prepared data changed: {name}")
    out.mkdir(parents=True)
    source_hashes = {name: file_digest(name) for name in [
        "benchmarks/scifact_cache_ablation.py", "benchmarks/cache_ablation_data.py",
        "train.py", "grad_cache.py", "objectives.py", "training_data.py", "retrieval.py", "evaluate.py"]}
    report = {"protocol": PROTOCOL, "arm": args.arm, "seed": args.seed,
              "manifest_sha256": file_digest(data / "manifest.json"), "source_sha256": source_hashes,
              "versions": {name: version(name) for name in ["mlx", "mlx-lm", "mlx-embeddings", "numpy", "transformers"]}}
    (out / "run-config.json").write_text(json.dumps(report, indent=2) + "\n")
    report["training"] = None if args.arm == "baseline" else fit(data, out, args.arm, args.seed)
    gc.collect()
    mx.clear_cache()
    report["evaluation"] = evaluate_final(data, out, args.seed, baseline=args.arm == "baseline")
    for name, digest in manifest["file_sha256"].items():
        if file_digest(data / name) != digest:
            raise ValueError(f"Prepared data changed during experiment: {name}")
    with (out / "result.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
