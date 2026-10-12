# Changelog

Entries use GitHub merge dates in UTC. These are repository changes, not tagged
package releases. See [validation](docs/validation.md) for evidence attached to
the latest training-feature release.

## 2026-10-12

[PR 3](https://github.com/Adam-Researchh/mlx-embed-finetune/pull/3) added:

- Opt-in true multi-positive training with uniform positive targets and
  deduplicated candidate documents.
- Gradient-cached contrastive batches with exact MLX RNG replay, chunked
  embedding-loss gradients and bounded validation encoding.
- Offline teacher score generation and optional ranking KL distillation with
  dataset-hash and candidate-identity validation.
- Export provenance for feature settings and teacher data, real-model parity
  tests, a combined-feature CI workflow and a reproducible memory benchmark.

Existing commands retain default single-positive behavior. The new features
are opt-in. Teacher and student dimensions may differ; the student still serves
one vector per text. The release was validated locally on October 11 and merged
on October 12 UTC.

## 2026-09-24

[PR 2](https://github.com/Adam-Researchh/mlx-embed-finetune/pull/2) added corpus
evaluation, multiple relevance judgments, binary nDCG@10, dimension sweeps,
negative mining with optional teacher filtering, hardness weighting and
retrieval-based checkpoint selection.

Correctness fixes covered stale best checkpoints, partial accumulation windows,
final evaluation, finite-value checks, one-step warmup updates, numeric edge
cases, prompt metadata, report-file protection and mixed quantized exports.

Compatibility changes: pair parsing became strict; training requires a fresh
output directory and rejects exact train/validation query overlap; report paths
must be new files. The published SciFact measurement is an untuned baseline.

## 2026-08-26

[PR 1](https://github.com/Adam-Researchh/mlx-embed-finetune/pull/1) introduced the
earlier correctness and architecture pass: path-based adapter targeting,
quantized-layer handling, hard negatives, duplicate-positive protection,
Matryoshka training, prefix metadata, best-checkpoint export and CI smoke checks.
The associated local verification was recorded on August 25.

The original BGE-M3 timings remain available as
[historical measurements](benchmarks/results.md#historical-bge-m3-workflow).
