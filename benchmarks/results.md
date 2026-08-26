# Benchmark Results

## Summary

This pipeline targets a practical Apple Silicon problem: encoder fine-tuning workflows that spend hours on CPU-heavy or poorly accelerated PyTorch paths despite high-end integrated GPU hardware.

The MLX version moves the training loop onto Metal and makes much better use of unified memory.

## Measured Comparison

| Method | Hardware | Dataset | Epochs | Time | GPU Usage |
|---|---|---:|---:|---:|---:|
| PyTorch + sentence-transformers | M1 Ultra 128GB | 9K pairs | 3 | ~6-8 hours | <5% |
| **MLX encoder fine-tuning** | M1 Ultra 128GB | 9K pairs | 3 | **56 minutes** | **78%** |

## Dry-Run Measurements

Measured during a one-batch validation pass:

- Forward + backward: **2.1s/batch**
- Batch size: **16**
- Throughput: **7.6 pairs/sec**
- Unified memory footprint: **~5-6GB**

Notes:

- Early steps include MLX compilation and warmup overhead
- Throughput is expected to improve modestly after warmup
- Actual runtime depends on sequence lengths, batch size, and model choice

## Why The Speedup Happens

1. **Metal-native execution** through MLX instead of a weakly utilized PyTorch encoder path on Apple Silicon
2. **Unified memory** reduces friction between compute and model state
3. **LoRA fine-tuning** updates a tiny subset of parameters instead of the full encoder
4. **LoRA on Q/V only** keeps the backward pass narrow

## Practical Takeaway

On the measured M1 Ultra setup, this reduced a multi-hour encoder fine-tuning workflow to under an hour while substantially increasing GPU utilization.


## Verification, 2026-08-25

Re-verified on the current stack after five months of upstream movement. No
timing re-run was performed, so the M1 Ultra numbers above are unchanged and
still date from the original BGE-M3 run.

**Stack tested:** mlx 0.32.2 · mlx-lm 0.31.3 · mlx-embeddings 0.1.0 ·
transformers 5.15.1 · Python 3.14 · macOS / Apple Silicon.

**Result:** the pipeline still runs — but two silent failures were found and
fixed, both of which produced a *confident-looking* run that trained nothing.

| Finding | Symptom before the fix |
|---|---|
| `QuantizedLinear` is not a subclass of `nn.Linear` | Every projection in a 4/6/8-bit model was skipped. The script printed `Applied LoRA to 0 layers`, continued through optimizer setup, and died several steps later inside MLX with `[grad] Must specify at least one argument`. |
| Zero adapted layers was not an error | The `0 layers` / `0.000M trainable` lines were printed as ordinary status output. Nothing in the pipeline treated "nothing to train" as a failure. |

Both now raise `NoAdaptedLayersError` at the point of failure, naming the
modules that *were* found. The dry run additionally asserts a non-zero gradient
norm, and CI asserts that a bogus `--target-modules` exits non-zero.

**Also corrected:** the merged export previously fused whatever weights the
final step produced, even when an earlier checkpoint had a better eval loss,
and the tokenizer file list omitted `vocab.txt` — so a merged BERT-family
export shipped without its vocabulary.

### Loss signal without hard negatives

Measured on `mlx-community/all-MiniLM-L6-v2-bf16`, batch of 4, τ=0.05:
embeddings arrive already L2-normalized, the diagonal logit is 20.0 and
off-diagonal logits are ~3.8. Loss is **0.0001** and the gradient norm is
**~4e-4**. That is not a bug — it is a well-trained encoder finding random
in-batch negatives trivial, and it is the concrete reason explicit hard
negatives were added.
