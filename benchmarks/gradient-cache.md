# Gradient-cache validation

The cache is tested against an ordinary backward pass over the same logical
candidate pool. The test matrix covers single/multiple positives, Matryoshka
loss, hardness weighting, teacher ranking KL, and dropout in all 32 combinations.
Tests compare every parameter gradient and the loss, not only gradient norms.
Real bf16 and 4-bit MiniLM LoRA models also undergo parity checks with dropout
and all objectives enabled. RNG state after replay matches one encoding pass.

With dropout, comparisons use identical encoding chunks and RNG draws. A full
unsplit batch can draw different masks. A separate deterministic test compares
cached gradients directly with one unsplit forward/backward pass.

## Memory measurement

Measured on October 11, 2026 on an Apple M1 Ultra with 128 GiB unified memory,
with MLX 0.32.2 using a cached MiniLM bf16 model,
LoRA rank 8, 32 queries, 128 candidate documents, and 128 tokens per input.
The synthetic objective includes two positives per query and teacher ranking
KL. Both runs disable all dropout to make their losses directly comparable.

| Mode | Peak MLX allocation | Forward/backward time |
|---|---:|---:|
| Uncached | 5.060 GiB | 0.350 s |
| Cached, encoding chunks of 4 | 0.400 GiB | 0.933 s |

This single workload used **92.1% less peak MLX tensor memory**, with extra
compute from replay. Loss differed by 4.77e-7 and gradient norm by 6.71e-8.
Separate gradient-parity tests verify parameter-wise agreement. These numbers
are not a general speed promise or a retrieval-quality improvement. They omit
Python memory and optimizer updates/state, include model allocations, and are
single forward/backward timings rather than steady-state throughput averages.

Reproduce in separate processes on Apple Silicon after installing requirements:

```bash
python -m benchmarks.cache_memory \
  --queries 32 --sequence-length 128 --output outputs/memory-uncached.json
python -m benchmarks.cache_memory \
  --queries 32 --sequence-length 128 --cache-chunk-size 4 \
  --output outputs/memory-cached.json
```

Use new output paths. Raw settings and measurements are in
[gradient-cache-2026-10-11.json](gradient-cache-2026-10-11.json).

## Teacher and full-loop validation

The end-to-end smoke recipe scores two-positive training rows with a
768-dimensional ModernBERT teacher, then trains a 384-dimensional 4-bit MiniLM
student. It combines multi-positive supervision, cache chunk size 3, loss chunk
size 3, gradient accumulation, Matryoshka dimensions 384/128/64, hardness
weighting, dropout, periodic/final validation, best-checkpoint restoration,
export and exported-model evaluation. Small irregular sizes exercise partial
chunks and final optimizer windows.

This is a wiring/correctness test on synthetic example data. Adoption as a
quality default still requires a held-out multi-seed ablation on meaningful
retrieval data. The existing SciFact recipe can evaluate such an experiment,
with a separate validation split used for checkpoint selection.
