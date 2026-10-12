# SciFact cache ablation on one Mac

**Result: improved numerical fidelity, without a demonstrated retrieval-quality
gain. Keep plain caching as the default.** All twelve training runs completed
on this Mac. Autodiff caching had a lower test nDCG@10 point estimate than plain
caching in each seed; the exploratory paired interval includes zero. This
does not establish superiority, inferiority or equivalence.

The [summary and diagnostics](../benchmarks/scifact-cache-ablation-2026-10-12.json)
and [all per-query test measurements](../benchmarks/scifact-cache-ablation-2026-10-12-per-query.jsonl)
are public. An independent reviewer verified pairing, recomputed the metrics
from per-query results, and reproduced the primary bootstrap interval with a
separate implementation. The final local regression suite passed **180 tests**.

## Retrieval results

The untuned pinned model scored **0.629264 nDCG@10**, **0.597652 MRR@10** and
**0.756056 Recall@10** with the same 128-token limit and serving settings.

| Training arm | Seed 41 nDCG@10 | Seed 42 | Seed 43 | Mean |
|---|---:|---:|---:|---:|
| Ordinary same-chunk backward | 0.660793 | 0.661237 | 0.654731 | 0.658920 |
| Plain cache | 0.660419 | 0.662233 | 0.659026 | 0.660559 |
| Autodiff cache | 0.660227 | 0.661568 | 0.654206 | 0.658667 |
| Float32 training, common low-precision serving | 0.656874 | 0.661145 | 0.649394 | 0.655804 |

| Autodiff minus plain | Mean difference | Exploratory paired 95% interval |
|---|---:|---:|
| nDCG@10, primary | -0.001893 | [-0.006112, +0.000350] |
| MRR@10 | -0.002509 | [-0.007968, +0.000310] |
| Recall@10 | 0.000000 | [0.000000, 0.000000] |

Recall@10 agrees for every paired seed/query observation in this experiment.
The nDCG differences by seed are -0.000192, -0.000665 and -0.004820.
Fine-tuning improved the test baseline in every arm/seed, but that improvement
cannot be attributed to the new cache mode. Mean validation nDCG@10 was
0.625345 / 0.625745 / 0.624422 / 0.623813 for reference / plain / autodiff /
float32, against the untuned 0.621189; it was not used for model selection.

## Multi-step fidelity and resource cost

Across the nine sampled training states per arm, plain cache gradients had
relative L2 error **0.01194–0.12548** against the ordinary backward at the
same current weights. Eight autodiff checks were between **6.39e-8 and 1.66e-7**;
one reached **0.001019** at seed 41, update 21. This supports improved local
gradient fidelity, with a residual discrepancy rather than universal exactness.

Final adapter distance from the reference trajectory, normalized by the
reference adapter update norm from initialization, was:

| Seed | Plain cache | Autodiff cache |
|---|---:|---:|
| 41 | 0.19250 | 0.15514 |
| 42 | 0.19189 | 0.12466 |
| 43 | 0.37512 | 0.10763 |

Autodiff stays closer in this LoRA parameterization but does not reproduce the
reference trajectory. Parameter distance is not prediction distance or a
retrieval metric, and the reference itself did not outperform plain caching
on mean test nDCG in this pilot.

| Arm | Median timed training for 41 updates | Maximum peak MLX allocation |
|---|---:|---:|
| Reference | 30.84 s | 6.087 GiB |
| Plain cache | 44.58 s | 0.772 GiB |
| Autodiff cache | 48.19 s | 0.771 GiB |
| Float32 cache training | 48.06 s | 1.337 GiB |

The autodiff arm used **87.3% less peak MLX allocation** than the ordinary
reference and **8.1% more median training time** than plain caching. The memory
saving is the benefit of caching; this experiment's new cache pass largely
preserved it. No retrieval evaluation, diagnostics or model loading is included
in those timed training totals. See the protocol below for measurement limits.

This result supports retaining an opt-in numerical-fidelity mode, not promoting
it as a retrieval-quality upgrade. A longer run, other tasks, precision settings
or hardware could behave differently; none was selected after inspecting these
test outcomes.

## Protocol fixed before retrieval outcomes

This is an exploratory, single-machine pilot of `--cache-forward-mode autodiff`.
It tests whether the numerical improvement translates into retrieval quality
after training, and whether gradient agreement persists across optimizer steps.
It does not certify a world first or change the default training mode.

Primary comparison: **autodiff cache minus plain cache, test binary nDCG@10**.
Secondary comparisons: MRR@10, Recall@10, autodiff versus the same-chunk ordinary
backward reference, and an alternative float32 training control. Every arm,
seed and result is reported. No hyperparameter or checkpoint is selected using
the test set. The final checkpoint after one fixed epoch is evaluated.

### Data

- Official BEIR SciFact archive SHA-256:
  `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.
- 5,183 title-plus-abstract documents and 300 official test queries.
- Of 809 official training queries, exclude IDs `871` and `1291`: their exact
  query texts occur in the test split under different IDs.
- Split the remaining 807 rows by connected components of exact query text
  and positive-document text, using seed `20261012` and a 20% validation target.
  This yields **644 training / 163 validation** rows.
- Mine two fixed BM25 negatives per training row from the full corpus, excluding
  all known positives for that query. Unjudged documents are assumed negative;
  they may contain false negatives. Mining is identical for every arm.
- This is a shared-corpus retrieval setting: 181 positive document IDs overlap
  the official train/test splits. The internal split separates positive groups,
  but negative candidates can cross that boundary. Near-duplicate claims can
  remain. The checkpoint may also have seen SciFact during original pretraining.
  Holdout claims apply only to the fine-tuning experiment described here.

### Model and fixed budget

| Setting | Value |
|---|---|
| Model | `mlx-community/nomicai-modernbert-embed-base-4bit` |
| Snapshot | `50ba8b0351a2a795299e15946e6c82e918bf38d5` |
| Hardware | Apple M1 Ultra, 128 GiB unified memory |
| Training seeds | 41, 42, 43 |
| Budget | One epoch, 644 rows, 41 optimizer updates per arm/seed |
| Logical batch / encoding chunk / loss chunk | 16 / 4 / 8 |
| Token limit | 128 for both training and evaluation |
| LoRA | Rank 8, alpha 16, dropout 0.1; ordinary zero-B initialization |
| Optimizer | AdamW, peak learning rate 2e-5, weight decay 0.01 |
| Schedule | 10% warmup with positive first step, then cosine decay |
| Objective | Normalized full-dimensional multi-positive contrastive loss, temperature 0.05 |
| Other objectives | No teacher distillation, MRL or hardness weighting |
| Query / document prefixes | `search_query: ` / `search_document: ` |
| Evaluation | Full corpus, fixed encoding batch 32, no reranker |

Four training arms use the same initial adapters, row order, data, masks and
encoder chunks for each seed:

1. **reference**: ordinary backward through the concatenated chunk encodings.
2. **plain**: the existing plain cache pass.
3. **autodiff**: the experimental differentiated cache pass.
4. **float32**: plain caching after promoting floating model tensors to float32;
   quantized packed weights remain quantized. This is an alternative training
   precision, not the ground truth for the low-precision reference.

All final adapters are reloaded onto the **same original low-precision base**
for evaluation, so float32 training is not confounded with float32 inference.
An untuned baseline uses identical serving precision, prefixes, token limit
and corpus. Validation is reported descriptively, without selecting a winner.

### Diagnostics and uncertainty

At updates 1, 21 and 41, each arm's gradient is compared with an ordinary
same-chunk backward at that arm's current weights and RNG. Diagnostics do not
update weights or the optimizer; RNG is restored after checking. These local
gradient checks are distinct from the final differences between independently
evolving training trajectories.

Training timing/peak MLX allocation exclude diagnostics, model loading and
retrieval evaluation. Peak allocation includes the resident model and optimizer,
but is not process RSS. Runs execute sequentially; timing is descriptive on
this Mac, not a cross-hardware speed claim.

Report every per-seed metric and paired difference. Exploratory 95% bootstrap
intervals resample paired seeds and connected positive-document/query groups
among test queries, preserving query weights and using the same sampled groups
across sampled seeds. Use 10,000 draws with seed `20261012`. Three seeds and one
small task are insufficient for a general superiority or equivalence claim.

## Reproduce

Download and verify the archive using the [SciFact recipe](../benchmarks/scifact.md).
Prepare the fixed split and lexical negatives:

```bash
python -m benchmarks.cache_ablation_data \
  --data-dir outputs/beir/scifact --output-dir outputs/cache-ablation-data
```

The preparation manifest records exact input/output hashes, excluded IDs and
query groups. The pinned model snapshot must already be cached. Run the
untuned baseline, then each arm with seeds 41, 42 and 43, choosing new output
directories:

```bash
HF_HUB_OFFLINE=1 python -m benchmarks.scifact_cache_ablation \
  --data-dir outputs/cache-ablation-data --arm baseline --seed 41 \
  --output-dir outputs/cache-ablation-runs/baseline-41
HF_HUB_OFFLINE=1 python -m benchmarks.scifact_cache_ablation \
  --data-dir outputs/cache-ablation-data --arm autodiff --seed 41 \
  --output-dir outputs/cache-ablation-runs/autodiff-41
```

Repeat the second command for `reference`, `plain`, `autodiff` and `float32`
with all three seeds. This experiment used order `reference, plain, autodiff,
float32` for seed 41; `plain, autodiff, float32, reference` for seed 42; and
`autodiff, float32, reference, plain` for seed 43.

After all twelve training runs are complete:

```bash
python -m benchmarks.summarize_cache_ablation \
  --runs-dir outputs/cache-ablation-runs \
  --manifest outputs/cache-ablation-data/manifest.json \
  --output outputs/cache-ablation-summary.json \
  --per-query-output outputs/cache-ablation-per-query.jsonl
```

Analysis rejects mismatched run identities, source/protocol/data hashes,
library versions, query order, paired initial adapters, training order, step
counts or RNG histories. It uses paired, crossed seed-and-document-group
bootstrap resampling, not 900 independent seed/query observations.

The protocol was published before outcomes in
[commit 276ae81](https://github.com/Adam-Researchh/mlx-embed-finetune/commit/276ae8157d95579d504ef05c6771dbe6296c4826).
One startup attempt failed before inference because the offline cache lacked
non-model repository files. All reported runs use the corrected required-file
resolver in [af4c8d9](https://github.com/Adam-Researchh/mlx-embed-finetune/commit/af4c8d9).
The production trainer and Adam Embed v3 service are not changed by this pilot.
