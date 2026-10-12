# Benchmarks

Start with [results and scope](results.md), then use the corresponding recipe.
Run commands from the repository root and choose fresh output paths.

| Recipe or utility | Purpose |
|---|---|
| [SciFact](scifact.md) | Reproduce a public corpus retrieval baseline with pinned model/data identities |
| [Gradient cache](gradient-cache.md) | Compare cached and uncached forward/backward memory on the same synthetic workload |
| [BEIR preparation](prepare_beir.py) | Convert local BEIR files into pair/corpus JSONL with binary relevance |
| [Cache measurement utility](cache_memory.py) | Record loss, gradient norm, peak MLX allocation and one-pass time |
| [Cache replay consistency](../docs/replay-consistency.md) | Compare plain and autodiff cache forwards against matched gradients, with raw multi-seed measurements |

Raw JSON files are dated measurements, not promises for other configurations.
[Validation coverage](../docs/validation.md) records correctness tests separately.
[Design rationale](../docs/design-rationale.md) explains the choices behind the
training features, and [evaluation guidance](../docs/evaluation.md) describes
how to measure a quality improvement.
