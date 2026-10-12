# Autodiff cache forwards: a research candidate

**Status: experimental implementation with numerical evidence; research
priority is unverified.** This work does not establish a new contrastive loss,
a new gradient-caching algorithm, a world first, or better retrieval quality.

The subsequent [SciFact training pilot](scifact-cache-ablation.md) tested twelve
matched training runs on this Mac. Numerical fidelity improved, but mean test
nDCG@10 was 0.6587 for autodiff versus 0.6606 for plain caching, with an
exploratory paired interval including zero. Plain remains the default.

## The question

Can a gradient cache match an ordinary differentiated encoder's gradients in
low precision without promoting the entire model to float32 or retaining the
whole batch's activation graph?

Our [adversarial audit](adversarial-audit-2026-10-12.md) found a relevant failure
of a practical assumption. On the tested MLX stack, an encoder called normally
can return slightly different embeddings from the same encoder inside
`nn.value_and_grad`, even with identical input chunks, weights and RNG state.
Those differences can propagate into the shared embedding-loss gradients.
Using float32 removed most of the discrepancy, but changed the precision and
resource profile of the computation.

Ordinary GradCache first caches embeddings without retaining the encoder's
backward graph, differentiates the shared loss, then replays chunks with the
cached embedding gradients. If the cached forward gives `z_plain` and the
differentiated replay gives `z_AD`, the resulting computation uses an embedding
loss gradient evaluated at `z_plain` with the differentiated encoder Jacobian.
Matching the RNG does not by itself make those two forward values identical.

## The implementation

The opt-in `--cache-forward-mode autodiff` changes the initial cache pass:

1. For one encoding chunk, trace a scalar sum of its embeddings with
   `nn.value_and_grad`, returning the embeddings as an auxiliary output.
2. Discard the unused parameter-gradient graph before evaluating it.
3. Materialize only the differentiated forward embeddings and stop their
   gradient connection before retaining them in the cache.
4. Compute the ordinary shared embedding objective and its representation
   gradients, using the existing query-row loss chunks.
5. Replay encoder chunks with the recorded RNG states and cached gradients.

MLX's lazy evaluation makes step 3 possible. This uses a documented framework
capability; it does not invent lazy autodiff or partial graph evaluation.
The scalar sum is a tracing device, not an additional training objective.
No dummy parameter gradients are added to the optimizer update. Model dtypes
and quantized weights remain unchanged.

The reference for agreement is an ordinary backward pass over the **same
encoding chunks and random draws**. An unsplit batch can select other kernels
or dropout masks. Floating-point reduction order can still differ, so we test
numerical tolerances rather than promise bitwise equality for every model.

Implementation: [grad_cache.py](../grad_cache.py). Integration and regression
checks: [training](../train.py), [objective combinations](../tests/test_cached_training.py),
and [real models](../tests/test_export.py).

## Reproduction

Run from the repository root with cached weights and project dependencies:

```bash
mkdir -p outputs/replay-research
HF_HUB_OFFLINE=1 python -m benchmarks.replay_consistency \
  --model mlx-community/nomicai-modernbert-embed-base-4bit \
  --seeds 41 42 43 --queries 12 --sequence-length 64 \
  --chunk-size 3 --loss-chunk-size 5 --repeats 3 \
  --output outputs/replay-research/modernbert.json
```

Repeat for `mlx-community/embeddinggemma-300m-4bit` and
`mlx-community/all-MiniLM-L6-v2-4bit`, choosing distinct output files. Omit
`HF_HUB_OFFLINE=1` if the weights have not been downloaded.

To reproduce the variable-length text control, add `--fixture ragged`,
`--seeds 431 432 433 --chunk-size 2 --loss-chunk-size 1`, and a fresh output
path. This fixture has three query rows and six deduplicated candidates;
`--queries` is unused and `--sequence-length` is its tokenizer length limit.

The [benchmark source](../benchmarks/replay_consistency.py) compares three modes:
an ordinary same-chunk backward reference, the existing plain cache pass, and
the autodiff cache pass. It activates both LoRA matrices, enables dropout, and
combines multi-positive supervision, Matryoshka dimensions, hardness and
teacher-ranking distillation. All modes share weights, tokens, labels, chunk
boundaries and initial RNG. Each mode is warmed once and measured three times;
the measurement order reverses on even seeds. Gradients copied to host memory
are compared outside the timed step.

The default workload uses synthetic token IDs; the ragged control uses the
small text fixture embedded in the script. Neither is a quality benchmark.
Peak allocation is MLX tensor memory,
including the resident model; it excludes Python/NumPy memory and is not
process RSS. Timings cover forward/backward plus Python tracing, without an
optimizer update. Results therefore establish neither retrieval quality nor
whole-training throughput.

## Recorded results

Measured on an Apple M1 Ultra with 128 GiB unified memory, MLX 0.32.2,
mlx-lm 0.31.3, and mlx-embeddings 0.1.0. The
[raw results](../benchmarks/replay-consistency-2026-10-12.json) record every
seed, repeat, model snapshot and resolved library version. Three seeds were
run sequentially per configuration, with three measured repeats after warmup.
All modes preserved the reference RNG progression.

Gradient error below is `||g - g_reference||_2 / ||g_reference||_2` over all
trainable parameters. Ranges span seeds; they are ratios, not percentages.

| Model, all 4-bit | Fixture | Plain gradient error | Autodiff gradient error |
|---|---|---:|---:|
| ModernBERT | 12 queries / 48 candidates, 64 tokens | 0.0622–0.1986 | 1.16e-7–1.38e-7 |
| Gemma | 12 queries / 48 candidates, 64 tokens | 1.18e-7–1.23e-7 | 1.18e-7–1.23e-7 |
| MiniLM | 12 queries / 48 candidates, 64 tokens | 9.71e-6–2.40e-5 | 5.56e-6–1.36e-5 |
| ModernBERT | Three variable-length query rows | 0.00479–0.00820 | 6.70e-8–0.000898 |
| Gemma | Three variable-length query rows | 0.0157–0.0217 | 5.98e-8–6.56e-8 |

The fixed-length Gemma case is a negative control: the plain cache already
agrees and autodiff gives no numerical benefit. The ragged ModernBERT case
retains a residual discrepancy on two seeds. These observations rule out
claiming universal exactness or universal benefit. Different sequence shapes,
kernels and reduction orders matter.

For the 12-query synthetic workload, peaks are the maximum over the nine
measurements per mode; time is the median of those nine measurements:

| Model | Ordinary backward peak | Plain / autodiff cache peak | Ordinary / plain / autodiff seconds |
|---|---:|---:|---:|
| ModernBERT | 2.939 GiB | 0.330 / 0.330 GiB | 0.675 / 0.943 / 1.060 |
| Gemma | 4.252 GiB | 0.455 / 0.455 GiB | 0.790 / 1.073 / 1.249 |
| MiniLM | 1.060 GiB | 0.142 / 0.142 GiB | 0.231 / 0.383 / 0.412 |

The autodiff cache retained **86.6–89.3% lower peak MLX allocation** than the
ordinary same-chunk backward reference on this workload, with **7.6–16.4%**
extra median time relative to plain caching. On the much smaller ragged
fixture, memory reductions were only **16.9–18.6%** and time overhead was
**22.5–27.1%**. The overhead matters even when the numerical benefit is small.
These are one-machine measurements, not cross-hardware performance guarantees.

## Prior work and the novelty boundary

The search was performed on October 12, 2026 UTC over papers, public source,
framework documentation and GitHub issue history. It cannot establish the
absence of unpublished or differently described work.

| Prior work | What it already establishes | Difference investigated here |
|---|---|---|
| [GradCache, Gao et al. (2021)](https://arxiv.org/abs/2101.06983) and its [implementation](https://github.com/luyug/GradCache/blob/main/src/grad_cache/grad_cache.py) | Decouple representation gradients from chunked encoder replay to support large contrastive batches; no-grad first pass and RNG replay | Select the differentiated forward numerics during the initial cache pass |
| [GradCache issue 16 (2022)](https://github.com/luyug/GradCache/issues/16) | Numerical differences between cached and ordinary training were already reported | Hold chunk shapes/RNG fixed, isolate an AD-path difference, and test a targeted correction |
| [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html) | Forward outputs from a gradient computation can be evaluated without executing the backward graph | Apply that existing capability inside GradCache for forward/replay numerical agreement |
| [Sentence Transformers cached losses](https://sbert.net/docs/package_reference/sentence_transformer/losses.html) | Maintained cached contrastive training, loss chunking, and related features | Evaluate this MLX-specific cache-forward choice; no claim that general caching or chunking is new |
| [RWKV-Metal MLX GradCache](https://github.com/RafaelUI/rwkv-metal/blob/main/rwkv_metal/embedding/gradcache.py) and [Sentence Transformers' shared engine](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/base/losses/gradcache.py) | Existing MLX/PyTorch cache engines with an ordinary/no-grad first pass and differentiated replay | Their inspected first passes do not use the trace-and-discard cache-forward choice; MLX GradCache itself is already implemented elsewhere |
| [Transformer Engine precision notes](https://docs.nvidia.com/deeplearning/transformer-engine/examples/advanced_optimizations.html) | Caching and low-precision computations can affect numerical outputs | A different cache and framework; included to reject a broad claim that cache-related numerical drift is newly discovered |

The candidate contribution is **a memory-bounded use of MLX autodiff forward
evaluation to reduce the cached-versus-reference gradient gap, with unchanged
model precision and matched RNG**. The sources checked did not establish an
earlier identical application, but that is not evidence sufficient for a
world-first claim. It is an implementation technique built on established
methods, with a narrowly testable benefit.

Two broader ideas were rejected as novelty targets during this review:
time-aware memory retrieval has close work such as
[MemStrata](https://arxiv.org/abs/2606.26511), and upgrading an embedding model
while retaining its old index has close work such as
[Drift-Adapter](https://arxiv.org/abs/2509.23471). Those may still be useful
features; they do not justify claiming to invent those capabilities.

## Adoption and remaining experiments

Enable the experiment on an otherwise ordinary cached run:

```bash
python train.py --train-pairs example_data/train.jsonl \
  --model mlx-community/all-MiniLM-L6-v2-4bit \
  --multi-positive --batch-size 8 --cache-chunk-size 3 \
  --cache-forward-mode autodiff --dry-run
```

The default remains `plain`. Export metadata records the selected mode.
Local validation passed **168 tests** with real-model tests enabled. An
independent reviewer also checked zero-sum embeddings, frozen parameters,
unchanged dtypes and retained-output memory growth, and found no new defect
within that scope.
The encoder must still be stateless apart from RNG, have trainable parameters,
and return one embedding per input. This is not a solution for BatchNorm,
incorrect masks, pooling defects or arbitrary custom derivative behavior.
In particular, it does not repair the known quantized Gemma padding defect.

Before promoting this technique beyond experimental status:

- Reproduce on other Apple GPU generations and MLX versions, with pinned model
  snapshots and the same precision, sequence lengths and dropout conditions.
- Measure longer training trajectories and held-out retrieval against plain
  caching and a float32 control beyond the completed 41-update SciFact pilot.
  Closer gradients did not establish a retrieval gain in that first comparison.
- Extend prior-work review and independent replication before using any
  priority claim. If an earlier equivalent implementation is found, credit it
  and retain this as a reproduction or engineering adaptation.
