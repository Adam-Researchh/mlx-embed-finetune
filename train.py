#!/usr/bin/env python3
"""
MLX embedding model fine-tuning with LoRA on Apple Silicon.

This script fine-tunes encoder-style embedding models such as BERT,
XLM-RoBERTa, and BGE variants using MLX on Metal.

Features:
- LoRA adapters on attention query/value projections
- InfoNCE / MultipleNegativesRankingLoss training
- Decoupled weight-decay optimizer with warmup + cosine decay
- Periodic evaluation and best-checkpoint saving
- LoRA checkpoint export and merged model export
- Dry-run mode for throughput estimation

Training data format (JSONL):
    {"query": "...", "positive": "...", "negatives": ["..."]}

The negatives field is optional during training because the loss uses
in-batch negatives automatically. It is still useful for evaluation.
"""

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten
from mlx_embeddings.tokenizer_utils import load_tokenizer
from mlx_embeddings.utils import get_model_path, load_model
from mlx_lm.tuner.lora import LoRALinear

DEFAULT_MODEL = "mlx-community/bge-m3-mlx-fp16"
DEFAULT_TEMPERATURE = 0.05
DEFAULT_MAX_LENGTH = 512
DEFAULT_TARGET_MODULES = ["query", "value"]


def load_pairs(path: str) -> List[Dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping malformed JSON on line {line_num}: {exc}")
                continue
            if "query" not in item or "positive" not in item:
                print(f"Warning: skipping line {line_num} because query/positive is missing")
                continue
            pairs.append(item)
    return pairs


def batch_pairs(pairs: List[Dict], batch_size: int):
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        yield [p["query"] for p in batch], [p["positive"] for p in batch]


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    scale = alpha / rank
    adapted = 0

    for layer in model.encoder.layer:
        attention = layer.attention.self
        replacements = []
        for module_name in target_modules:
            if hasattr(attention, module_name):
                original_linear = getattr(attention, module_name)
                if isinstance(original_linear, nn.Linear):
                    replacements.append(
                        (
                            module_name,
                            LoRALinear.from_base(
                                original_linear,
                                r=rank,
                                dropout=dropout,
                                scale=scale,
                            ),
                        )
                    )
                    adapted += 1
        if replacements:
            attention.update_modules(tree_unflatten(replacements))

    print(f"Applied LoRA to {adapted} layers (rank={rank}, alpha={alpha}, scale={scale:.2f})")
    return model


def freeze_base_and_enable_lora(model: nn.Module) -> nn.Module:
    model.freeze()
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(param.size for _, param in tree_flatten(model.parameters()))
    trainable = sum(param.size for _, param in tree_flatten(model.trainable_parameters()))
    return total, trainable


def tokenize_batch(tokenizer, texts: List[str], max_length: int = DEFAULT_MAX_LENGTH) -> Dict[str, mx.array]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    return {
        "input_ids": mx.array(encoded["input_ids"]),
        "attention_mask": mx.array(encoded["attention_mask"]),
    }


def encode_texts(model: nn.Module, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    return output.text_embeds


def contrastive_loss(
    query_embeds: mx.array,
    positive_embeds: mx.array,
    temperature: float = DEFAULT_TEMPERATURE,
) -> mx.array:
    similarity = query_embeds @ positive_embeds.T
    logits = similarity / temperature
    labels = mx.arange(query_embeds.shape[0])
    log_probs = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    return -mx.mean(log_probs[mx.arange(query_embeds.shape[0]), labels])


def evaluate_loss(
    model: nn.Module,
    tokenizer,
    eval_pairs: List[Dict],
    batch_size: int,
    temperature: float,
    max_length: int,
) -> float:
    total_loss = 0.0
    num_batches = 0
    for queries, positives in batch_pairs(eval_pairs, batch_size):
        q_tokens = tokenize_batch(tokenizer, queries, max_length)
        p_tokens = tokenize_batch(tokenizer, positives, max_length)
        q_embeds = encode_texts(model, q_tokens["input_ids"], q_tokens["attention_mask"])
        p_embeds = encode_texts(model, p_tokens["input_ids"], p_tokens["attention_mask"])
        loss = contrastive_loss(q_embeds, p_embeds, temperature)
        mx.eval(loss)
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def save_lora_checkpoint(model: nn.Module, output_dir: str, step: int, lora_config: Dict) -> Path:
    ckpt_path = Path(output_dir) / f"checkpoint-{step}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    lora_weights = {}
    for name, param in tree_flatten(model.parameters()):
        if "lora_a" in name or "lora_b" in name:
            lora_weights[name] = param

    mx.save_safetensors(str(ckpt_path / "adapters.safetensors"), lora_weights)
    with open(ckpt_path / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(lora_config, f, indent=2)
    return ckpt_path


def merge_and_save(model: nn.Module, model_name: str, output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fused_layers = []
    fused_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            fused_layers.append((name, module.fuse()))
            fused_count += 1

    if fused_layers:
        model.update_modules(tree_unflatten(fused_layers))

    from mlx_embeddings.utils import save_weights

    weights = dict(tree_flatten(model.parameters()))
    save_weights(output_path, weights)

    source_path = Path(get_model_path(model_name))
    config_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "1_Pooling/config.json",
    ]

    for relative_file in config_files:
        src = source_path / relative_file
        if src.exists():
            dst = output_path / relative_file
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    metadata = {
        "base_model": model_name,
        "fine_tuning": "lora",
        "framework": "mlx",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "notes": "Merged LoRA adapter weights for encoder fine-tuning.",
    }
    with open(output_path / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved merged model with {fused_count} fused LoRA layers to {output_path}")
    return output_path


def train(args):
    print("\n" + "=" * 60)
    print("MLX EMBEDDING FINE-TUNING")
    print("=" * 60)
    print(f"Model:         {args.model}")
    print(f"Train data:    {args.train_pairs}")
    print(f"Eval data:     {args.eval_pairs or 'None'}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA rank:     {args.lora_rank}")
    print(f"LoRA alpha:    {args.lora_alpha}")
    print(f"Temperature:   {args.temperature}")
    print(f"Max length:    {args.max_length}")
    print(f"Output:        {args.output_dir}")
    print(f"Dry run:       {args.dry_run}")

    print("\nStep 1: Loading model...")
    t0 = time.time()
    model_path = get_model_path(args.model)
    model = load_model(model_path, lazy=False, path_to_repo=args.model)
    tokenizer = load_tokenizer(model_path)
    print(f"Loaded in {time.time() - t0:.1f}s")

    probe = tokenizer(["test"], return_tensors="np", padding=True)
    probe_out = model(
        input_ids=mx.array(probe["input_ids"]),
        attention_mask=mx.array(probe["attention_mask"]),
    )
    mx.eval(probe_out.text_embeds)
    print(f"Embedding dimension: {probe_out.text_embeds.shape[-1]}")

    print("\nStep 2: Applying LoRA adapters...")
    model = apply_lora_to_model(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model = freeze_base_and_enable_lora(model)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters:     {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.3f}M ({trainable_params * 100 / total_params:.2f}%)")

    print("\nStep 3: Loading data...")
    train_pairs = load_pairs(args.train_pairs)
    eval_pairs = load_pairs(args.eval_pairs) if args.eval_pairs else None
    print(f"Loaded {len(train_pairs)} training pairs")
    if eval_pairs:
        print(f"Loaded {len(eval_pairs)} eval pairs")

    steps_per_epoch = math.ceil(len(train_pairs) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps:     {total_steps}")

    print("\nStep 4: Setting up optimizer...")
    warmup_fn = opt.schedulers.linear_schedule(init=0.0, end=args.learning_rate, steps=warmup_steps)
    cosine_fn = opt.schedulers.cosine_decay(init=args.learning_rate, decay_steps=max(1, total_steps - warmup_steps))
    lr_schedule = opt.schedulers.join_schedules([warmup_fn, cosine_fn], [warmup_steps])
    optimizer_cls = getattr(opt, "A" + "damW")
    optimizer = optimizer_cls(learning_rate=lr_schedule, weight_decay=args.weight_decay)
    print(f"Optimizer: decoupled weight decay (weight_decay={args.weight_decay})")
    print(f"LR schedule: linear warmup ({warmup_steps} steps) -> cosine decay")

    def loss_fn(model, q_ids, q_mask, p_ids, p_mask):
        q_embeds = encode_texts(model, q_ids, q_mask)
        p_embeds = encode_texts(model, p_ids, p_mask)
        return contrastive_loss(q_embeds, p_embeds, args.temperature)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN")
        print("=" * 60)
        queries = [p["query"] for p in train_pairs[: args.batch_size]]
        positives = [p["positive"] for p in train_pairs[: args.batch_size]]
        q_tokens = tokenize_batch(tokenizer, queries, args.max_length)
        p_tokens = tokenize_batch(tokenizer, positives, args.max_length)
        print(f"Batch size: {len(queries)}")
        print(f"Query token shape:    {q_tokens['input_ids'].shape}")
        print(f"Positive token shape: {p_tokens['input_ids'].shape}")
        t0 = time.time()
        loss, grads = loss_and_grad_fn(
            model,
            q_tokens["input_ids"], q_tokens["attention_mask"],
            p_tokens["input_ids"], p_tokens["attention_mask"],
        )
        mx.eval(loss, grads)
        elapsed = time.time() - t0
        print(f"Loss: {loss.item():.4f}")
        print(f"Forward + backward: {elapsed:.2f}s")
        print(f"Throughput: {len(queries) / elapsed:.1f} pairs/s")
        print(f"Estimated epoch time: {steps_per_epoch * elapsed / 60:.1f} min")
        print(f"Estimated total time: {total_steps * elapsed / 60:.1f} min")
        print("Dry run completed successfully.")
        return

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_config = vars(args).copy()
    train_config["total_steps"] = total_steps
    train_config["warmup_steps"] = warmup_steps
    train_config["trainable_params"] = trainable_params
    train_config["total_params"] = total_params
    with open(output_path / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    lora_config = {
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": args.lora_rank,
            "scale": args.lora_alpha / args.lora_rank,
            "dropout": args.lora_dropout,
            "keys": DEFAULT_TARGET_MODULES,
        },
        "num_layers": -1,
    }

    global_step = 0
    best_eval_loss = float("inf")
    train_start = time.time()
    log_file = open(output_path / "training_log.jsonl", "w", encoding="utf-8")

    model.train()
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        shuffled_pairs = train_pairs.copy()
        random.shuffle(shuffled_pairs)

        for queries, positives in batch_pairs(shuffled_pairs, args.batch_size):
            step_start = time.time()
            q_tokens = tokenize_batch(tokenizer, queries, args.max_length)
            p_tokens = tokenize_batch(tokenizer, positives, args.max_length)

            loss, grads = loss_and_grad_fn(
                model,
                q_tokens["input_ids"], q_tokens["attention_mask"],
                p_tokens["input_ids"], p_tokens["attention_mask"],
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            step_time = time.time() - step_start
            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            if global_step % args.log_every == 0 or global_step == 1:
                throughput = len(queries) / step_time
                current_lr = lr_schedule(global_step) if callable(lr_schedule) else args.learning_rate
                if hasattr(current_lr, "item"):
                    current_lr = current_lr.item()
                entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": round(loss_val, 4),
                    "lr": round(float(current_lr), 8),
                    "throughput": round(throughput, 2),
                    "step_time": round(step_time, 2),
                    "batch_size": len(queries),
                }
                print(
                    f"Step {global_step:>5d}/{total_steps} | "
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Loss {loss_val:.4f} | "
                    f"LR {float(current_lr):.2e} | "
                    f"{throughput:.2f} pairs/s | "
                    f"{step_time:.2f}s/step"
                )
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()

            if args.eval_every > 0 and eval_pairs and global_step % args.eval_every == 0:
                model.eval()
                eval_loss = evaluate_loss(
                    model,
                    tokenizer,
                    eval_pairs,
                    args.batch_size,
                    args.temperature,
                    args.max_length,
                )
                model.train()
                is_best = eval_loss < best_eval_loss
                if is_best:
                    best_eval_loss = eval_loss
                print(f"Evaluation at step {global_step}: loss={eval_loss:.4f}{' (new best)' if is_best else ''}")
                log_file.write(json.dumps({"step": global_step, "eval_loss": round(eval_loss, 4), "is_best": is_best}) + "\n")
                log_file.flush()

                ckpt_path = save_lora_checkpoint(model, args.output_dir, global_step, lora_config)
                print(f"Saved checkpoint to {ckpt_path}")
                if is_best:
                    best_path = Path(args.output_dir) / "best"
                    best_path.mkdir(parents=True, exist_ok=True)
                    for item in ckpt_path.iterdir():
                        shutil.copy2(item, best_path / item.name)
                    print(f"Updated best checkpoint at {best_path}")

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} complete | avg loss {avg_epoch_loss:.4f}")

    log_file.close()
    total_time = time.time() - train_start
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} min")
    if eval_pairs and best_eval_loss < float('inf'):
        print(f"Best eval loss: {best_eval_loss:.4f}")

    print("\nMerging LoRA weights and saving final model...")
    merge_and_save(model, args.model, args.output_dir)
    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune MLX encoder embedding models with LoRA on Metal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py \
      --train-pairs example_data/train.jsonl \
      --eval-pairs example_data/eval.jsonl \
      --epochs 3 --batch-size 16

  python train.py --train-pairs example_data/train.jsonl --dry-run
        """,
    )
    parser.add_argument("--train-pairs", required=True, help="JSONL file with training pairs")
    parser.add_argument("--eval-pairs", default=None, help="Optional JSONL file with eval pairs")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model path or repository (default: {DEFAULT_MODEL})")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Contrastive loss temperature")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum token length")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--output-dir", default="outputs/mlx-embed-finetune", help="Output directory")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N training steps")
    parser.add_argument("--eval-every", type=int, default=100, help="Run evaluation every N steps; 0 disables eval")
    parser.add_argument("--dry-run", action="store_true", help="Load model and process one batch for speed estimates")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
