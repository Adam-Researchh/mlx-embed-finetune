# Benchmark results

The recorded measurements answer different questions: memory use during a
synthetic gradient computation, retrieval quality of an untuned baseline, and
elapsed time in an older training workflow. Use the settings and scope attached
to each measurement when comparing a new run.

| Measurement | Result | Evidence |
|---|---|---|
| Gradient cache, October 11, 2026 | Peak MLX allocation 5.060 → 0.400 GiB; single forward/backward time 0.350 → 0.933 s | [Recipe](gradient-cache.md), [raw results](gradient-cache-2026-10-11.json) |
| Untuned MiniLM on SciFact, September 24, 2026 | nDCG@10 0.6455; MRR@10 0.6049; Recall@10 0.7850 | [Recipe](scifact.md), [raw results](scifact-baseline-2026-09-24.json) |
| Original BGE-M3 training workflow | 56 minutes with MLX versus approximately 6–8 hours for the recorded PyTorch workflow | [Historical record below](#historical-bge-m3-workflow) |

The cache measurement demonstrates a memory/compute tradeoff on its stated
workload. The SciFact result establishes a baseline for the evaluator. Neither
measurement establishes a retrieval-quality gain from fine-tuning. Correctness
and model coverage are documented separately in [validation](../docs/validation.md).

## Gradient cache measurement

The synthetic workload uses 32 queries, 128 candidate documents, 128 tokens per
input, LoRA rank 8, multiple positives and ranking KL. Both runs disable dropout
for a direct comparison. On an M1 Ultra with 128 GiB unified memory and MLX
0.32.2, encoding chunks of four reduced peak MLX tensor allocation by **92.1%**.
Loss differed by 4.77e-7; separate tests verify every-parameter gradient agreement.

The recorded times are single forward/backward passes in separate processes.
They include model allocations but exclude Python memory and optimizer state/
updates. See [the complete measurement](gradient-cache.md) before extrapolating
to another sequence length, candidate pool, model or training loop.

## SciFact retrieval baseline

The full corpus contains 5,183 documents and the test split contains 300 queries.
The recorded model is `mlx-community/all-MiniLM-L6-v2-bf16`, using empty prefixes
and a 256-token maximum length. Titles are prepended to abstracts; relevance is
binary and exact duplicate document text is collapsed.

| Dimensions | nDCG@10 | MRR@10 | Recall@10 |
|---|---:|---:|---:|
| 384 | 0.6455 | 0.6049 | 0.7850 |
| 128 | 0.5991 | 0.5544 | 0.7583 |
| 64 | 0.4839 | 0.4450 | 0.6145 |

Smaller vectors are truncations of the stock model with re-normalization. The
model was not newly trained with Matryoshka loss for this measurement. Dataset
and model hashes, versions and commands are in [the SciFact recipe](scifact.md).

## Historical BGE-M3 workflow

These observations were recorded before the August 25, 2026 verification pass.
That pass did not repeat the timing comparison. The figures describe the
original workflow and do not establish a general speed ratio against current,
optimized PyTorch/MPS implementations.

| Method | Hardware | Dataset | Epochs | Time | GPU usage |
|---|---|---:|---:|---:|---:|
| PyTorch + sentence-transformers | M1 Ultra 128GB | 9K pairs | 3 | approximately 6–8 hours | below 5% |
| MLX embedding fine-tuning | M1 Ultra 128GB | 9K pairs | 3 | 56 minutes | approximately 78% |

The original one-batch validation notes recorded 2.1 s forward/backward at batch
size 16, 7.6 pairs/s, and approximately 5–6 GB unified memory. The original notes
are not a version-pinned, command-complete speed benchmark; use these figures as
historical context rather than a current framework ranking.

## Historical verification on August 25

Recorded stack: MLX 0.32.2, mlx-lm 0.31.3, mlx-embeddings 0.1.0, transformers
5.15.1, Python 3.14, macOS/Apple Silicon.

That correctness pass fixed quantized layers being skipped during adapter
selection, made zero adapted/trainable parameters an error, restored the best
checkpoint before merge, and included missing BERT vocabulary files in exports.
See [the changelog](../CHANGELOG.md) for subsequent correctness work.

The historical MiniLM batch-of-four diagnostic at temperature 0.05 recorded
loss near 0.0001 and gradient norm near 4e-4 on easy in-batch contrasts. Easy
negatives can provide little training signal; the observation does not imply
that every dataset requires the same negative-mining recipe.

### Historical architecture checks

The earlier fixed block walk did not cover ModernBERT's fused projections or
decoder-style naming. Path-based presets enabled the following checks:

| Model | Preset | Adapted modules | Trainable parameters | Gradient norm |
|---|---|---:|---:|---:|
| MiniLM bf16 | `bert/xlm-roberta` | 12 | 0.074M | 4.3e-4 |
| MiniLM bf16, all-linear | `all-linear` | 36 | 0.332M | 7.4e-4 |
| Nomic ModernBERT bf16 | `modernbert` | 44 | 0.811M | 8.6e-2 |
| Nomic ModernBERT 4-bit | `modernbert` | 44 | 0.811M | 2.3e-1 |
| EmbeddingGemma 300M 4-bit | `decoder-style` | 48 | 0.492M | 2.6e-1 |

These are execution diagnostics. A larger gradient norm or a larger adapted
parameter count does not by itself show improved retrieval quality.

## Add a comparable result

Record the command, model revision, dataset hash/split, hardware, package
versions, precision, token lengths, query/candidate counts, warmup and repeat
protocol. Distinguish tensor allocation from total process memory, individual
passes from steady-state throughput, and base-model scores from tuned-model
improvements. Follow [the evaluation guide](../docs/evaluation.md) for ablations.
