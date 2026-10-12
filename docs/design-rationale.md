# Design rationale

The project aims to make embedding fine-tuning inspectable and practical on
Apple Silicon while retaining one vector per text at serving time. Its main
training choices address three constraints: limited activation memory, queries
with several correct answers, and transferring relevance judgments from a
stronger teacher into a smaller student.

## Keep the serving interface simple

LoRA, multi-positive training, gradient caching and ranking distillation change
how weights are learned. The exported student still emits a single dense
embedding. Existing cosine/dot-product retrieval infrastructure can consume it
when dimensions, prompts and normalization are configured correctly.

The embedding space itself changes during fine-tuning. Existing corpus vectors
generally need to be recomputed. This project does not promise interchangeable
vector spaces across base, tuned, teacher or unrelated models.

Multi-vector methods such as ColBERT require token-level outputs, different
scoring, additional masking and different index/storage assumptions. They are
a separate extension in the [roadmap](../ROADMAP.md), not a hidden option in
the current trainer. Reranker serving and multimodal input pipelines are also
outside the implemented training/export contract.

## Use MLX and adapt the loaded model

MLX provides native Apple Silicon array operations and automatic differentiation.
The project reuses model implementations from `mlx-embeddings` and LoRA layers
from `mlx-lm`, with adapter targets selected by module path. This covers the
tested BERT, XLM-RoBERTa, ModernBERT and decoder-style naming conventions without
hard-coding one transformer block layout into the trainer.

LoRA limits trainable parameters; QLoRA also keeps supported base layers
quantized. Both still need activation memory. That is why adapter efficiency
and gradient caching are complementary. Neither model-loading success nor a
nonzero gradient proves correct pooling or export: [model coverage](models.md)
records the stronger checks where they exist.

The export is native MLX. Fusing a quantized adapter dequantizes that adapted
layer and removes its stale per-layer quantization setting. Other layers may
remain quantized. Cross-framework conversions need their own round-trip tests.

## Preserve default single-positive behavior

Without `--multi-positive`, each row has one designated target. Other known
positives for that query are masked from the negative pool. This protects
alternate correct answers without changing the historical objective by default.

With `--multi-positive`, all known positives for each batch query are included
and candidate text is deduplicated. Relevance is the union across rows with the
same exact query text. The objective is uniform cross entropy over positives:

`L_query = -(1 / number_of_positives) * sum(log P(positive | query))`

The batch loss averages these query losses. Every positive contributes, so an
easy positive cannot completely hide an ignored harder positive as it can in
an objective that only maximizes total probability assigned to the positive
set. The design assumes all provided positives deserve equal supervision;
graded relevance and per-positive weights are not implemented. Repeated query
rows retain their sampling weight.

Known positives never receive a hardness penalty. Optional hardness weighting
adds `strength * stop_gradient(cosine_similarity)` to eligible negative logits.
Detaching that value preserves the intended weighting interpretation. A zero
strength retains the standard objective. See the
[Sentence Transformers loss reference](https://sbert.net/docs/package_reference/sentence_transformer/losses.html).

Implementation: [training_data.py](../training_data.py) builds labels and
candidate pools; [objectives.py](../objectives.py) defines the shared losses.

## Separate contrastive batches from optimizer batches

`--batch-size` controls how many queries share a contrastive candidate pool.
`--grad-accum-steps` combines gradients from independent pools before one
optimizer update. Accumulation therefore increases the optimizer batch but
does not add cross-pool negatives to any query's loss.

GradCache lets a larger logical contrastive batch fit by separating the encoder
backward pass from the embedding loss. The implementation follows the gradient
caching idea from [Gao et al.](https://arxiv.org/abs/2101.06983):

1. Encode small chunks and materialize embeddings without retaining an encoder
   backward graph.
2. Compute the shared loss and its gradients with respect to embeddings. Query
   row chunks bound each loss-gradient score matrix.
3. Replay each encoding chunk and propagate its cached embedding gradient
   through the model. Accumulate parameter gradients before the optimizer step.

Replay must use the same stochastic realization as the first encoding pass.
The cache captures MLX PRNG keys, verifies their restoration through the public
key/seed APIs, and restores the caller's random stream after replay. Dropout
parity is tested against an uncached backward pass using the same encoding
chunks and masks. A separate deterministic test compares with an unsplit batch.
BatchNorm models are rejected because their forward state can change on replay.
This is mathematical gradient equivalence, with finite-precision tolerances
in practice. Quantized kernels can produce different rounding under autodiff;
the [audit](adversarial-audit-2026-10-12.md) records additional model-specific
measurements and a separate upstream padding limitation.

Stored embeddings, token inputs, masks, teacher-score matrices and embedding
gradients still consume memory. The cache trades extra encoding computation
for lower activation memory. The [measured workload](../benchmarks/gradient-cache.md)
reduced peak MLX allocation by 92.1% while increasing forward/backward time.
It does not establish a universal memory ratio or speedup.

Implementation: [grad_cache.py](../grad_cache.py). Tests include irregular chunks,
RNG restoration after a failing loss, and all 32 objective/dropout combinations.

## Distill rankings rather than vector coordinates

A teacher and student may have different embedding dimensions and internal
architectures. Ranking distillation compares their score distributions over
the same query-document group, so their vector coordinates need not match.

`teacher.py` computes cosine scores offline. Student training loads the saved
numbers, not the teacher model. The total objective is:

`L = L_contrastive + distill_weight * KL(P_teacher || P_student)`

Both score distributions use `distill_temperature`; no temperature-squared
multiplier is applied. The KL term uses full-dimensional student scores.
Matryoshka prefixes retain their supervised contrastive objectives, keeping
their dimensionality tradeoff separate from the teacher's full ranking signal.

Each query has a fixed scored group derived from its training positives and
explicit negatives. Scored group documents are included in the batch. Unscored
cross-row candidates participate in contrastive learning but receive no invented
teacher labels. Each valid document receives teacher probability once, even
when the legacy candidate list contains duplicates. Single-positive mode still
masks alternate positives, and at least two valid scored candidates must remain.

The file format can accept calibrated scores produced by an external reranker.
This repository does not execute or calibrate that reranker. Temperature and
distillation weight are experimental controls, not universal quality defaults.
Teacher quality and label consistency must be evaluated on held-out retrieval.
For numerical stability, each scored group is centered before temperature
scaling and zero teacher-probability terms contribute zero to KL. A common
score offset must not change the teacher distribution; extreme finite scores
are covered by regression tests.

Implementation: [teacher.py](../teacher.py), [cache validation](../training_data.py)
and [ranking KL](../objectives.py). Research on soft-label contrastive distillation
includes [KaLM-Embedding-V2](https://arxiv.org/abs/2506.20923); this implementation
does not reproduce that paper's complete training recipe.

## Make validation and provenance part of training

Training loss diagnoses optimization. Corpus retrieval measures whether useful
answers rank well. Candidate pools and objective terms change across recipes,
so raw training losses are not a fair cross-recipe quality comparison.

Validation supports multiple positives, binary nDCG@10, MRR@10 and Recall@k.
Checkpoint selection can use a retrieval metric. Partial accumulation windows
take the same evaluation/save path, and enabled evaluation covers the final
state. The selected checkpoint is restored before export.

New runs require empty output directories, and reports use new files. Dataset
and teacher hashes are captured at load time. Teacher caches must match the
exact training file and contain complete, finite, unambiguous scores. These
checks prevent silent reuse of stale state; they do not replace good labeling
or train/validation/test separation.

The [validation record](validation.md) distinguishes correctness checks,
synthetic memory measurements, public retrieval baselines and future quality
ablations. Experimental training features remain opt-in until task-specific
evidence supports making them defaults.
