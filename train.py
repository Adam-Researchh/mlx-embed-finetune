#!/usr/bin/env python3
"""
MLX embedding model fine-tuning with LoRA on Apple Silicon.

Fine-tunes encoder-style embedding models such as BERT, XLM-RoBERTa and BGE
variants using MLX on Metal.

Features:
- LoRA adapters on attention query/value projections (fp16/bf16 and quantized)
- InfoNCE / MultipleNegativesRankingLoss with optional explicit hard negatives
- False-negative masking for duplicate positives inside a batch
- Optional Matryoshka (MRL) multi-dimension loss
- Gradient accumulation for a larger effective batch (more in-batch negatives)
- Decoupled weight-decay optimizer with warmup + cosine decay
- Periodic evaluation, best-checkpoint tracking, and best-checkpoint export
- LoRA checkpoint export and merged model export
- Dry-run mode for throughput estimation

Training data format (JSONL):
    {"query": "...", "positive": "...", "negatives": ["...", "..."]}

`negatives` is optional. When present it is used as explicit hard negatives in
the loss, which is the strongest quality lever available here: with random
in-batch negatives alone a well-trained base model often sits near zero loss
and receives almost no gradient signal.
"""

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_embeddings.tokenizer_utils import load_tokenizer
from mlx_embeddings.utils import get_model_path, load_model
from mlx_lm.tuner.lora import LoRALinear

DEFAULT_MODEL = "mlx-community/bge-m3-mlx-fp16"
DEFAULT_TEMPERATURE = 0.05
DEFAULT_MAX_LENGTH = 512
DEFAULT_TARGET_MODULES = ["query", "value"]

# Layer types LoRA can wrap. QuantizedLinear is NOT a subclass of nn.Linear,
# so it must be named explicitly or every projection in a quantized model is
# silently skipped. mlx_lm's LoRALinear.from_base handles both.
ADAPTABLE_LINEAR_TYPES = (nn.Linear, nn.QuantizedLinear)


class NoAdaptedLayersError(RuntimeError):
    """Raised when LoRA matched nothing, instead of training zero parameters."""


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
    """Yield (queries, positives, hard_negatives) triples.

    hard_negatives is a flat list pooled across the batch; every row sees every
    other row's hard negatives as additional negatives, which is standard
    practice and costs nothing extra.
    """
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        queries = [p["query"] for p in batch]
        positives = [p["positive"] for p in batch]
        negatives: List[str] = []
        for p in batch:
            for neg in p.get("negatives") or []:
                if isinstance(neg, str) and neg:
                    negatives.append(neg)
        yield queries, positives, negatives


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
    seen_module_names = set()
    skipped_types = {}

    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layer"):
        raise NoAdaptedLayersError(
            "Model has no `.encoder.layer` stack. This pipeline targets encoder "
            "architectures (BERT / XLM-RoBERTa family). Top-level modules found: "
            f"{[name for name, _ in model.named_modules() if name and '.' not in name]}"
        )

    for layer in model.encoder.layer:
        attention = layer.attention.self
        replacements = []
        for module_name in target_modules:
            if not hasattr(attention, module_name):
                continue
            original_linear = getattr(attention, module_name)
            seen_module_names.add(module_name)
            if isinstance(original_linear, ADAPTABLE_LINEAR_TYPES):
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
            else:
                skipped_types[module_name] = type(original_linear).__name__
        if replacements:
            attention.update_modules(tree_unflatten(replacements))

    if adapted == 0:
        raise NoAdaptedLayersError(
            "LoRA adapted 0 layers, so there would be nothing to train.\n"
            f"  requested target modules: {target_modules}\n"
            f"  attention submodules found: {sorted(seen_module_names) or 'none'}\n"
            f"  found-but-unsupported types: {skipped_types or 'none'}\n"
            "Pass --target-modules with names that exist on this architecture."
        )

    print(f"Applied LoRA to {adapted} layers (rank={rank}, alpha={alpha}, scale={scale:.2f})")
    return model


def freeze_base_and_enable_lora(model: nn.Module) -> nn.Module:
    model.freeze()
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    trainable = sum(param.size for _, param in tree_flatten(model.trainable_parameters()))
    if trainable == 0:
        raise NoAdaptedLayersError(
            "No trainable parameters after freezing the base model. "
            "LoRA adapters were not registered correctly."
        )
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


def l2_normalize(x: mx.array, eps: float = 1e-12) -> mx.array:
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), eps)


def encode_texts(
    model: nn.Module,
    input_ids: mx.array,
    attention_mask: mx.array,
    normalize: bool = True,
) -> mx.array:
    """Encode a batch of texts.

    mlx-embeddings already returns L2-normalized `text_embeds` for the models
    tested here, so normalizing again is a no-op for them. It is kept explicit
    and on by default because the InfoNCE temperature (0.05) is only meaningful
    on unit vectors, and a converted encoder that does not normalize would
    otherwise train on saturated logits without any error being raised.
    """
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    embeds = output.text_embeds
    return l2_normalize(embeds) if normalize else embeds


def build_false_negative_mask(positives: Sequence[str], negatives: Sequence[str]) -> Optional[mx.array]:
    """Mask candidates that duplicate a row's own positive text.

    Without this, two rows sharing (or repeating) a positive teach the model
    that a correct match is wrong.
    """
    candidates = list(positives) + list(negatives)
    batch_size = len(positives)
    mask_rows = []
    needs_mask = False
    for i in range(batch_size):
        row = []
        for j, text in enumerate(candidates):
            duplicate = (j != i) and (text == positives[i])
            if duplicate:
                needs_mask = True
            row.append(-1e9 if duplicate else 0.0)
        mask_rows.append(row)
    if not needs_mask:
        return None
    return mx.array(mask_rows)


def info_nce_loss(
    query_embeds: mx.array,
    candidate_embeds: mx.array,
    temperature: float = DEFAULT_TEMPERATURE,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """InfoNCE / MultipleNegativesRankingLoss.

    candidate_embeds is [positives; pooled_hard_negatives]; the correct answer
    for row i is column i.
    """
    batch_size = query_embeds.shape[0]
    logits = (query_embeds @ candidate_embeds.T) / temperature
    if mask is not None:
        logits = logits + mask
    log_probs = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    targets = mx.arange(batch_size)
    return -mx.mean(log_probs[targets, targets])


def matryoshka_loss(
    query_embeds: mx.array,
    candidate_embeds: mx.array,
    dims: Sequence[int],
    temperature: float,
    mask: Optional[mx.array],
) -> mx.array:
    """Matryoshka Representation Learning: train nested prefixes jointly.

    Produces embeddings that stay useful when truncated, so downstream indexes
    can trade recall for memory without retraining.
    """
    full_dim = query_embeds.shape[-1]
    total = None
    used = 0
    for dim in dims:
        if dim > full_dim:
            continue
        q = l2_normalize(query_embeds[:, :dim])
        c = l2_normalize(candidate_embeds[:, :dim])
        term = info_nce_loss(q, c, temperature, mask)
        total = term if total is None else total + term
        used += 1
    if total is None:
        raise ValueError(f"No Matryoshka dims <= embedding dim {full_dim}: {list(dims)}")
    return total / used


def evaluate_loss(
    model: nn.Module,
    tokenizer,
    eval_pairs: List[Dict],
    batch_size: int,
    temperature: float,
    max_length: int,
    normalize: bool,
    use_hard_negatives: bool,
) -> float:
    total_loss = 0.0
    num_batches = 0
    for queries, positives, negatives in batch_pairs(eval_pairs, batch_size):
        if not use_hard_negatives:
            negatives = []
        q_tokens = tokenize_batch(tokenizer, queries, max_length)
        c_texts = positives + negatives
        c_tokens = tokenize_batch(tokenizer, c_texts, max_length)
        q_embeds = encode_texts(model, q_tokens["input_ids"], q_tokens["attention_mask"], normalize)
        c_embeds = encode_texts(model, c_tokens["input_ids"], c_tokens["attention_mask"], normalize)
        mask = build_false_negative_mask(positives, negatives)
        loss = info_nce_loss(q_embeds, c_embeds, temperature, mask)
        mx.eval(loss)
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def lora_weight_dict(model: nn.Module) -> Dict[str, mx.array]:
    return {
        name: param
        for name, param in tree_flatten(model.parameters())
        if "lora_a" in name or "lora_b" in name
    }


def save_lora_checkpoint(model: nn.Module, output_dir: str, step: int, lora_config: Dict) -> Path:
    ckpt_path = Path(output_dir) / f"checkpoint-{step}"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(ckpt_path / "adapters.safetensors"), lora_weight_dict(model))
    with open(ckpt_path / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(lora_config, f, indent=2)
    return ckpt_path


def prune_checkpoints(output_dir: str, keep: int) -> None:
    """Keep only the `keep` most recent numbered checkpoints ('best' is never pruned)."""
    if keep <= 0:
        return
    root = Path(output_dir)
    checkpoints = sorted(
        (p for p in root.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for stale in checkpoints[:-keep]:
        shutil.rmtree(stale, ignore_errors=True)


def load_lora_checkpoint(model: nn.Module, ckpt_dir: Path) -> nn.Module:
    """Restore adapter weights from a checkpoint directory into the live model."""
    weights_file = ckpt_dir / "adapters.safetensors"
    if not weights_file.exists():
        raise FileNotFoundError(f"No adapters.safetensors in {ckpt_dir}")
    weights = mx.load(str(weights_file))
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


def merge_and_save(model: nn.Module, model_name: str, output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fused_layers = []
    fused_count = 0
    dequantized = False
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            is_quantized = isinstance(module.linear, nn.QuantizedLinear)
            fused_layers.append((name, module.fuse(dequantize=is_quantized)))
            dequantized = dequantized or is_quantized
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
        "vocab.txt",
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
        "fused_layers": fused_count,
        "dequantized_on_merge": dequantized,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "notes": "Merged LoRA adapter weights for encoder fine-tuning.",
    }
    with open(output_path / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if dequantized:
        print("Note: base layers were quantized; merged weights are dequantized.")
    print(f"Saved merged model with {fused_count} fused LoRA layers to {output_path}")
    return output_path


def train(args):
    print("\n" + "=" * 60)
    print("MLX EMBEDDING FINE-TUNING")
    print("=" * 60)
    print(f"Model:            {args.model}")
    print(f"Train data:       {args.train_pairs}")
    print(f"Eval data:        {args.eval_pairs or 'None'}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size} x {args.grad_accum_steps} accum "
          f"= {args.batch_size * args.grad_accum_steps} effective")
    print(f"Learning rate:    {args.learning_rate}")
    print(f"LoRA rank/alpha:  {args.lora_rank} / {args.lora_alpha}")
    print(f"Temperature:      {args.temperature}")
    print(f"Max length:       {args.max_length}")
    print(f"Hard negatives:   {not args.no_hard_negatives}")
    print(f"Normalize embeds: {not args.no_normalize}")
    print(f"Matryoshka dims:  {args.matryoshka_dims or 'disabled'}")
    print(f"Output:           {args.output_dir}")
    print(f"Dry run:          {args.dry_run}")

    normalize = not args.no_normalize
    use_hard_negatives = not args.no_hard_negatives
    matryoshka_dims = (
        [int(d) for d in args.matryoshka_dims.split(",")] if args.matryoshka_dims else None
    )

    print("\nStep 1: Loading model...")
    t0 = time.time()
    model_path = get_model_path(args.model)
    model = load_model(model_path, lazy=False)
    tokenizer = load_tokenizer(model_path)
    print(f"Loaded in {time.time() - t0:.1f}s")

    probe = tokenizer(["test"], return_tensors="np", padding=True)
    probe_out = model(
        input_ids=mx.array(probe["input_ids"]),
        attention_mask=mx.array(probe["attention_mask"]),
    )
    mx.eval(probe_out.text_embeds)
    embed_dim = probe_out.text_embeds.shape[-1]
    probe_norm = float(mx.linalg.norm(probe_out.text_embeds[0]).item())
    print(f"Embedding dimension: {embed_dim}")
    print(f"Base embedding L2 norm: {probe_norm:.4f}"
          f"{' (already normalized)' if abs(probe_norm - 1.0) < 1e-3 else ''}")

    print("\nStep 2: Applying LoRA adapters...")
    model = apply_lora_to_model(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",")],
    )
    model = freeze_base_and_enable_lora(model)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters:     {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.3f}M ({trainable_params * 100 / total_params:.2f}%)")

    print("\nStep 3: Loading data...")
    train_pairs = load_pairs(args.train_pairs)
    eval_pairs = load_pairs(args.eval_pairs) if args.eval_pairs else None
    if not train_pairs:
        raise ValueError(f"No usable training pairs in {args.train_pairs}")
    with_negatives = sum(1 for p in train_pairs if p.get("negatives"))
    print(f"Loaded {len(train_pairs)} training pairs ({with_negatives} with hard negatives)")
    if use_hard_negatives and with_negatives == 0:
        print("Note: no hard negatives present; falling back to in-batch negatives only.")
    if eval_pairs:
        print(f"Loaded {len(eval_pairs)} eval pairs")

    micro_steps_per_epoch = math.ceil(len(train_pairs) / args.batch_size)
    steps_per_epoch = math.ceil(micro_steps_per_epoch / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    print(f"Optimizer steps per epoch: {steps_per_epoch}")
    print(f"Total optimizer steps:     {total_steps}")

    print("\nStep 4: Setting up optimizer...")
    warmup_fn = opt.schedulers.linear_schedule(init=0.0, end=args.learning_rate, steps=warmup_steps)
    cosine_fn = opt.schedulers.cosine_decay(init=args.learning_rate, decay_steps=max(1, total_steps - warmup_steps))
    lr_schedule = opt.schedulers.join_schedules([warmup_fn, cosine_fn], [warmup_steps])
    optimizer = opt.AdamW(learning_rate=lr_schedule, weight_decay=args.weight_decay)
    print(f"Optimizer: AdamW (decoupled weight decay={args.weight_decay})")
    print(f"LR schedule: linear warmup ({warmup_steps} steps) -> cosine decay")

    def loss_fn(model, q_ids, q_mask, c_ids, c_mask, fn_mask):
        q_embeds = encode_texts(model, q_ids, q_mask, normalize)
        c_embeds = encode_texts(model, c_ids, c_mask, normalize)
        if matryoshka_dims:
            return matryoshka_loss(q_embeds, c_embeds, matryoshka_dims, args.temperature, fn_mask)
        return info_nce_loss(q_embeds, c_embeds, args.temperature, fn_mask)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    def prepare_batch(queries, positives, negatives):
        if not use_hard_negatives:
            negatives = []
        q_tokens = tokenize_batch(tokenizer, queries, args.max_length)
        c_tokens = tokenize_batch(tokenizer, positives + negatives, args.max_length)
        fn_mask = build_false_negative_mask(positives, negatives)
        return q_tokens, c_tokens, fn_mask

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN")
        print("=" * 60)
        queries, positives, negatives = next(batch_pairs(train_pairs, args.batch_size))
        q_tokens, c_tokens, fn_mask = prepare_batch(queries, positives, negatives)
        print(f"Batch size: {len(queries)} queries vs {c_tokens['input_ids'].shape[0]} candidates "
              f"({len(positives)} positives + {c_tokens['input_ids'].shape[0] - len(positives)} hard negatives)")
        print(f"Query token shape:     {q_tokens['input_ids'].shape}")
        print(f"Candidate token shape: {c_tokens['input_ids'].shape}")
        print(f"False-negative mask:   {'applied' if fn_mask is not None else 'not needed'}")
        t0 = time.time()
        loss, grads = loss_and_grad_fn(
            model,
            q_tokens["input_ids"], q_tokens["attention_mask"],
            c_tokens["input_ids"], c_tokens["attention_mask"],
            fn_mask,
        )
        mx.eval(loss, grads)
        elapsed = time.time() - t0
        grad_norm = math.sqrt(sum(float(mx.sum(g * g).item()) for _, g in tree_flatten(grads)))
        print(f"Loss: {loss.item():.4f}")
        print(f"Gradient L2 norm: {grad_norm:.6f}")
        if grad_norm == 0.0:
            raise RuntimeError("Gradient norm is exactly zero — nothing would train.")
        print(f"Forward + backward: {elapsed:.2f}s")
        print(f"Throughput: {len(queries) / elapsed:.1f} pairs/s")
        print(f"Estimated epoch time: {micro_steps_per_epoch * elapsed / 60:.1f} min")
        print(f"Estimated total time: {micro_steps_per_epoch * args.epochs * elapsed / 60:.1f} min")
        print("Dry run completed successfully.")
        return

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_config = vars(args).copy()
    train_config.update(
        {
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "embedding_dim": embed_dim,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
        }
    )
    with open(output_path / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    lora_config = {
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": args.lora_rank,
            "scale": args.lora_alpha / args.lora_rank,
            "dropout": args.lora_dropout,
            "keys": [m.strip() for m in args.target_modules.split(",")],
        },
        "num_layers": -1,
    }

    global_step = 0
    best_eval_loss = float("inf")
    best_step = None
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

        accumulated = None
        accum_count = 0
        accum_loss = 0.0
        step_start = time.time()
        pairs_in_step = 0

        for queries, positives, negatives in batch_pairs(shuffled_pairs, args.batch_size):
            q_tokens, c_tokens, fn_mask = prepare_batch(queries, positives, negatives)
            loss, grads = loss_and_grad_fn(
                model,
                q_tokens["input_ids"], q_tokens["attention_mask"],
                c_tokens["input_ids"], c_tokens["attention_mask"],
                fn_mask,
            )
            accumulated = grads if accumulated is None else tree_map(mx.add, accumulated, grads)
            accum_count += 1
            accum_loss += loss.item()
            pairs_in_step += len(queries)

            if accum_count < args.grad_accum_steps:
                continue

            if args.grad_accum_steps > 1:
                accumulated = tree_map(lambda g: g / args.grad_accum_steps, accumulated)
            optimizer.update(model, accumulated)
            mx.eval(model.parameters(), optimizer.state)

            step_time = time.time() - step_start
            loss_val = accum_loss / accum_count
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1
            accumulated, accum_count, accum_loss = None, 0, 0.0

            if global_step % args.log_every == 0 or global_step == 1:
                current_lr = lr_schedule(global_step)
                if hasattr(current_lr, "item"):
                    current_lr = current_lr.item()
                entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": round(loss_val, 4),
                    "lr": round(float(current_lr), 8),
                    "throughput": round(pairs_in_step / step_time, 2),
                    "step_time": round(step_time, 2),
                    "effective_batch": pairs_in_step,
                }
                print(
                    f"Step {global_step:>5d}/{total_steps} | "
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Loss {loss_val:.4f} | "
                    f"LR {float(current_lr):.2e} | "
                    f"{pairs_in_step / step_time:.2f} pairs/s | "
                    f"{step_time:.2f}s/step"
                )
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()

            if args.eval_every > 0 and eval_pairs and global_step % args.eval_every == 0:
                model.eval()
                eval_loss = evaluate_loss(
                    model, tokenizer, eval_pairs, args.batch_size,
                    args.temperature, args.max_length, normalize, use_hard_negatives,
                )
                model.train()
                is_best = eval_loss < best_eval_loss
                if is_best:
                    best_eval_loss = eval_loss
                    best_step = global_step
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
                prune_checkpoints(args.output_dir, args.keep_checkpoints)

            step_start = time.time()
            pairs_in_step = 0

        # Flush a partial accumulation window at the end of the epoch.
        if accumulated is not None and accum_count > 0:
            accumulated = tree_map(lambda g: g / accum_count, accumulated)
            optimizer.update(model, accumulated)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += accum_loss / accum_count
            epoch_steps += 1
            global_step += 1

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} complete | avg loss {avg_epoch_loss:.4f}")

    log_file.close()
    total_time = time.time() - train_start
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} min")

    best_path = Path(args.output_dir) / "best"
    if best_step is not None:
        print(f"Best eval loss: {best_eval_loss:.4f} at step {best_step}")
    if args.merge == "best" and best_path.exists():
        print(f"Restoring best checkpoint (step {best_step}) before merge...")
        load_lora_checkpoint(model, best_path)
    elif args.merge == "best":
        print("No best checkpoint recorded (no evaluation ran); merging final weights.")
    else:
        print("Merging final-step weights (--merge final).")

    print("\nMerging LoRA weights and saving final model...")
    merge_and_save(model, args.model, args.output_dir)
    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune MLX encoder embedding models with LoRA on Metal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py \\
      --train-pairs example_data/train.jsonl \\
      --eval-pairs example_data/eval.jsonl \\
      --epochs 3 --batch-size 16 --grad-accum-steps 4

  python train.py --train-pairs example_data/train.jsonl --dry-run
        """,
    )
    parser.add_argument("--train-pairs", required=True, help="JSONL file with training pairs")
    parser.add_argument("--eval-pairs", default=None, help="Optional JSONL file with eval pairs")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model path or repository (default: {DEFAULT_MODEL})")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Micro-batch size (pairs per forward pass)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Accumulate gradients over N micro-batches before an optimizer step")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Contrastive loss temperature")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum token length")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--target-modules", default=",".join(DEFAULT_TARGET_MODULES),
                        help="Comma-separated attention submodules to adapt (default: query,value)")
    parser.add_argument("--no-hard-negatives", action="store_true",
                        help="Ignore the `negatives` field and use in-batch negatives only")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip explicit L2 normalization of embeddings (not recommended)")
    parser.add_argument("--matryoshka-dims", default=None,
                        help="Comma-separated nested dims for Matryoshka loss, e.g. 768,512,256,128,64")
    parser.add_argument("--merge", choices=["best", "final"], default="best",
                        help="Which adapter weights to fuse into the exported model (default: best)")
    parser.add_argument("--keep-checkpoints", type=int, default=3,
                        help="Number of numbered checkpoints to retain (0 = keep all)")
    parser.add_argument("--output-dir", default="outputs/mlx-embed-finetune", help="Output directory")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N optimizer steps")
    parser.add_argument("--eval-every", type=int, default=100, help="Evaluate every N optimizer steps; 0 disables eval")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling and MLX")
    parser.add_argument("--dry-run", action="store_true", help="Load model and process one batch for speed estimates")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps must be >= 1")
    if args.seed is not None:
        random.seed(args.seed)
        mx.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
