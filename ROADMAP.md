# Roadmap

The implemented foundation includes LoRA/QLoRA, multiple positives, gradient
caching, offline ranking distillation, negative mining, retrieval evaluation
and merged MLX export. The next decisions should be driven by measured retrieval
quality and deployability. This is a prioritized direction, not a release schedule.

## Validate quality improvements

Run controlled ablations on held-out domain data and public retrieval tasks.
Separate the effect of mining, positive labels, batch size, hardness, Matryoshka
dimensions and teacher supervision. Hold preprocessing and training budgets
fixed, repeat seeds, and include out-of-domain checks. Promote experimental
defaults only after a reproducible quality or resource benefit.

## Extend verified model coverage

Evaluate Qwen3-Embedding and compact Harrier/Granite embedding checkpoints as
candidate additions. They are candidates, not newly verified support entries.
Each needs reference embedding agreement, correct prompts/pooling, intended
adapter targeting, finite gradients, cache parity, export/reload tests and a
retrieval benchmark on the intended hardware and precision.

## Improve long training runs

Exact resume would need optimizer state, model/adapter state, step and epoch,
sampler position, Python/MLX RNG state and dataset/configuration compatibility
checks. Current adapter checkpoints are insufficient. Streaming data and
teacher-score storage are also candidates when dataset size becomes limiting.

## Keep larger retrieval changes separate

Multi-vector retrieval requires token-level representations, MaxSim-style
scoring, new loss/masking rules and indexing/storage evaluation. Multimodal
models require image/audio/video preprocessing, data formats and benchmarks.
Hybrid search and reranker serving are downstream retrieval-system integrations.
These should be separate changes with explicit input/output contracts.

The rationale for the current boundary is in
[design rationale](docs/design-rationale.md). The current implementation and
compatibility notes are in the [changelog](CHANGELOG.md).
