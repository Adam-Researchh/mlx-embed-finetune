# Validation record

The October 2026 feature release validates execution and numerical correctness
across training, caching, distillation and export. Its evidence consists of
unit/reference tests, real-model checks, full workflows and separate benchmarks.

## Published release evidence

- [PR 3](https://github.com/Adam-Researchh/mlx-embed-finetune/pull/3) added
  multi-positive training, gradient caching and ranking distillation.
- Tested source commit: `3d31ab34d48d02559cd5487a42b8e04d6ed0927c`.
- Published main commit: `07f80f4d21bbe9f3b03c25953077da4e5d9c9a5e`.
- [PR CI](https://github.com/Adam-Researchh/mlx-embed-finetune/actions/runs/38187881985)
  and [post-merge CI](https://github.com/Adam-Researchh/mlx-embed-finetune/actions/runs/38188131856)
  passed on Linux and Apple Silicon.
- Local validation on October 11, 2026 passed **127 tests** with real-model
  checks enabled. The post-merge CI ran on October 12 UTC: 122 tests passed with
  five opt-in model tests skipped, then all five model tests passed separately.

Test counts describe that release. The commands below run the current suite.

## Independent adversarial follow-up

The [October 12 audit](adversarial-audit-2026-10-12.md) used a fresh reviewer,
independent numerical references and additional real-model probes. It found
and fixed a ranking-distillation numerical bug; all **131 local tests** pass
with real-model checks enabled. Four new regression cases fail before the fix.
The audit also isolated an existing upstream quantized EmbeddingGemma padding
defect and records that model's current limitation.

## What the tests establish

The subsequent [cache-forward experiment](replay-consistency.md) extends the
objective/dropout matrix to both plain and autodiff cache passes, adds tight
same-chunk ModernBERT/Gemma gradient comparisons without precision promotion,
and records reproducible synthetic memory/timing measurements. These checks
do not promote Gemma's separate padding behavior to verified correctness.

The [SciFact cache ablation](scifact-cache-ablation.md) adds twelve fixed-budget
training runs, three paired seeds, a common low-precision serving base, and
independently checked retrieval statistics. It found no demonstrated retrieval
gain for the autodiff mode. The current local suite passed **180 tests**;
the added checks cover data leakage, transitive positive groups, paired
bootstrap weighting, invalid pairings, optimizer updates and diagnostic RNG.

| Area | Checks |
|---|---|
| Retrieval | Chunked rankings and scalar metrics match independent references across 150 randomized cases; ties, cutoffs and multiple positives are covered |
| Multi-positive supervision | Candidate deduplication, global relevance unions, hard-positive gradients and compatibility with the legacy single-positive objective |
| Gradient caching | Loss and every-parameter gradients match references across 32 combinations of positive mode, MRL, hardness, distillation and dropout |
| Randomness and memory | Replay preserves the caller's RNG stream; partial chunks and a failing loss are covered; validation uses bounded encoding chunks; BatchNorm is rejected |
| Teacher scores | Scores retain text identity through shuffling; stale hashes, unknown/duplicate/missing entries, nonfinite values and insufficient groups are rejected |
| Real models | bf16 and 4-bit MiniLM LoRA gradient parity with both adapter matrices active; export/reload parity and per-layer quantization configuration |
| Training lifecycle | One-step updates, partial accumulation, final evaluation, independent saves, retention, best-checkpoint restore and finite-value guards |
| Full workflow | A 768-dimensional ModernBERT teacher scores a 384-dimensional QLoRA student workflow through training, validation, export, reload and dimension evaluation |

ModernBERT and EmbeddingGemma also passed local combined-feature forward/backward
checks. The [model table](models.md) distinguishes these from full export-parity
coverage. The bundled synthetic examples are useful for wiring tests; their
saturated scores cannot measure retrieval-quality improvements.

## Tested dependency configurations

| Component | Local feature validation, October 11 | Post-merge CI, October 12 UTC |
|---|---|---|
| Python | 3.14.6 | 3.12.10 arm64 |
| MLX | 0.32.2 | 0.32.3 |
| mlx-lm | 0.31.3 | 0.32.0 |
| mlx-embeddings | 0.1.0 | 0.1.0 |
| transformers | 5.17.0 | 5.19.0 |
| NumPy | 2.5.3 | 2.5.4 |
| sentencepiece | 0.2.2 | 0.2.2 |

`requirements.txt` specifies dependency floors, not a lockfile or a test matrix
for every combination. CI resolves current compatible releases and records
them in its logs. Save `python -m pip freeze` with a real training run when exact
environment reproduction matters. `LoRALinear` is imported from an internal
mlx-lm path; the smoke workflow checks that integration regularly.

## Run the checks

On Apple Silicon with the project dependencies installed:

```bash
python -m pip install pytest
python -m pytest -q
RUN_MODEL_TESTS=1 python -m pytest -q
```

The second command with `RUN_MODEL_TESTS=1` may download MiniLM weights. Set
`HF_HUB_OFFLINE=1` only after the required model snapshots have been cached.

The pure NumPy suite also runs on Linux without importing MLX:

```bash
python -m pip install numpy pytest
python -m pytest -q tests/test_retrieval.py tests/test_mining.py tests/test_benchmark.py tests/test_training_data.py
```

See [the workflow](../.github/workflows/smoke.yml) for the maintained architecture
and command-line smoke checks. Follow [evaluation guidance](evaluation.md) for
quality experiments and [benchmark results](../benchmarks/results.md) for the
separate measurements already recorded.
