# SciFact cache ablation on one Mac

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
query groups. Training/evaluation commands and results accompany the completed
pilot. The production trainer and Adam Embed v3 service are not changed by it.
