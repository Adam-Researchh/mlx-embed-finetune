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
4. **In-batch negatives** give a strong contrastive signal without heavyweight mining pipelines

## Practical Takeaway

On the measured M1 Ultra setup, this reduced a multi-hour encoder fine-tuning workflow to under an hour while substantially increasing GPU utilization.
