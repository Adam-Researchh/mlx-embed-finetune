# Training guide

Prepare labeled pairs, optionally mine negatives and score a teacher, then train
a student with the required prompt recipe. Run all commands from the repository
root. Commands using `data/` paths assume you have supplied those datasets.
For a complete example using bundled data, see [the combined smoke workflow](#combined-smoke-workflow).

## Training data format

```json
{"query": "search query", "positive": "relevant document", "negatives": ["a hard negative", "another"]}
```

`negatives` is optional. Explicit hard negatives can increase training signal
when random in-batch candidates are easy. Candidate documents are shared across
the batch, with known positives protected according to the selected positive
mode. Choose a mining policy using your data and held-out retrieval results.

Additional relevant answers can be listed in `positives`:

```json
{"query":"reset my password","positive":"Open account settings to reset it.","positives":["Use the password recovery page."],"negatives":["Change your display name."]}
```

By default, `positive` is the training target and other known positives are
masked out of the negative pool. With `--multi-positive`, all known positives
become supervised targets and candidates are deduplicated. Evaluation counts
all known positives, merging relevance for identical query text across rows. Each input row contributes to
the metric average. JSONL parsing is strict: malformed rows, empty strings, and
invalid negative lists raise with a file and line number.

## Training

```bash
python train.py \
  --train-pairs data/train.jsonl \
  --eval-pairs data/eval.jsonl \
  --model mlx-community/bge-m3-mlx-fp16 \
  --epochs 3 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --learning-rate 2e-5 \
  --lora-rank 8 \
  --output-dir outputs/run1
```

Use a **new, empty output directory** for every run. Existing runs are never
silently overwritten or used as an implicit resume. Train/eval queries must not
overlap. Split by source document or topic before mining where possible; the
exact-query overlap check is only a basic leakage guard.

See the [complete command reference](cli.md#training) for all flags and defaults.

## Larger contrastive batches and multiple positives

```bash
python train.py \
  --train-pairs data/train.jsonl --eval-pairs data/validation.jsonl \
  --model mlx-community/all-MiniLM-L6-v2-bf16 \
  --multi-positive --batch-size 64 --cache-chunk-size 8 --loss-chunk-size 16 \
  --best-metric ndcg_at_10 --seed 42 --output-dir outputs/cached-run
```

Here, all **64 queries share one candidate pool**, including their positives
and explicit negatives. Only eight query/document texts are encoded at a time.
The model first computes embeddings without retaining encoder activations,
then differentiates the shared embedding loss, then replays each encoding
chunk with its cached embedding gradient. `--grad-accum-steps 2` would accumulate
two independent 64-query losses, not create a 128-query candidate pool.

Multi-positive training averages the negative log probability of **each**
positive for each query, then averages across queries. It therefore rewards a
harder relevant answer even if an easier answer already ranks well. Relevance
is merged across identical query text throughout the training file; all known
answers for a batch query are included. Repeated query rows retain their data
weight. Hardness weighting applies only to negatives, never known positives.

GradCache restores the exact RNG key for each replay, including dropout.
With dropout active, parity is against an ordinary backward pass using the
same encoding chunks and random masks; changing batch shapes can change masks.
The caller's random stream advances once through encoding, not twice. Current
encoder forwards must be stateless apart from RNG; BatchNorm models are rejected.

`--cache-forward-mode autodiff` enables an experimental cache pass that follows
MLX's differentiated encoder forward path. It can reduce the numerical gap
between the initial cached embeddings and their later replay in low precision.
It requires a positive `--cache-chunk-size`, preserves model dtypes, and does
not change the loss. The default `plain` keeps the existing cache behavior.
This mode has extra tracing/forward cost and is supported by a
[numerical benchmark and prior-work review](replay-consistency.md), not a claim
of improved retrieval quality. It does not fix model padding or pooling bugs.

Memory still grows with the stored embeddings, token inputs, masks, teacher
scores, and embedding gradients. `--loss-chunk-size` bounds the score matrix
used by each cached loss-gradient computation. Validation also respects the
encoding chunk limit. Caching adds a forward replay; it is a memory feature,
not a promise of faster updates. See [the reproducible measurement](../benchmarks/gradient-cache.md).

## Distill teacher rankings offline

Score the training groups once with a teacher, then train the student:

```bash
python teacher.py \
  --train-pairs data/train.jsonl \
  --model mlx-community/nomicai-modernbert-embed-base-bf16 \
  --query-prefix "search_query: " --doc-prefix "search_document: " \
  --output outputs/teacher-scores.jsonl

python train.py \
  --train-pairs data/train.jsonl --eval-pairs data/validation.jsonl \
  --model mlx-community/all-MiniLM-L6-v2-4bit \
  --multi-positive --batch-size 64 --cache-chunk-size 8 \
  --teacher-scores outputs/teacher-scores.jsonl --distill-weight 0.2 \
  --distill-temperature 0.1 --best-metric ndcg_at_10 \
  --output-dir outputs/distilled-run
```

The teacher runs in evaluation mode with its own prompt recipe. It scores each
query against the union of that query's positives and explicit negatives. Mine
negatives first if a query has fewer than two candidates. The teacher and
student can have different embedding dimensions: we match **rank distributions**,
not embedding coordinates. No teacher model is loaded during student training.

The objective is `contrastive_loss + distill_weight * KL(teacher || student)`.
Both ranking distributions use `distill_temperature`; no temperature-squared
factor is applied. The KL term uses full-dimensional student cosine scores;
optional Matryoshka dimensions retain their supervised contrastive losses.
Only scored candidates participate in KL. Other in-batch documents still
participate in the contrastive loss without receiving invented teacher labels.
Scored groups are included in the batch pool, and each valid document receives
teacher probability once. With single-positive mode, alternate known positives
remain masked; at least two valid scored candidates must remain for each query.

`teacher.py` writes cosine scores. An external reranker can write the same JSONL
schema with calibrated scores (larger means more relevant); this project does
not run that reranker or automatically calibrate its raw logits. The first line
is a manifest, followed by one record per unique query:

```json
{"type":"teacher_manifest","schema_version":1,"train_sha256":"SHA256_OF_THE_EXACT_TRAIN_FILE","teacher":"model identifier"}
{"query":"search query","scores":[{"document":"relevant document","score":0.9},{"document":"hard negative","score":0.3}]}
```

Every training query and every document in its fixed group must be scored
exactly once. Unknown, duplicate, missing or nonfinite scores are rejected.
The training-file hash must match; regenerate scores after editing or mining
the dataset. Scores stay attached to text identities during shuffling and
candidate deduplication. Validation uses supervised full-dimension contrastive
loss and/or retrieval metrics, without training teacher labels. Compare recipes
on held-out retrieval, since mixed training losses are not directly comparable.

All three features are independently selectable and work together with LoRA,
QLoRA, accumulation, Matryoshka loss, hardness weighting and checkpoint export.
Exports still produce one vector per text and need no teacher at inference.
Deploying changed embedding weights generally requires re-embedding the corpus.

## Mine hard negatives

```bash
python mine.py \
  --train-pairs data/train.jsonl --corpus data/corpus.jsonl \
  --model mlx-community/all-MiniLM-L6-v2-bf16 \
  --num-negatives 3 --range-max 100 --relative-margin 0.05 \
  --output outputs/mined-train.jsonl

python train.py \
  --train-pairs outputs/mined-train.jsonl --eval-pairs data/validation.jsonl \
  --model mlx-community/all-MiniLM-L6-v2-bf16 \
  --best-metric ndcg_at_10 --hardness-strength 2 --output-dir outputs/run2
```

The miner retrieves candidates, excludes every known positive for that query,
and filters candidates scoring too close to a positive. `--range-min` and
`--range-max` are zero-based bounds in the original retrieval ranking, before
filtering. It replaces existing `negatives` and records scores, ranks and stable
text hashes. Output metadata records input hashes and shortfalls. It never
relaxes filters to fill the requested count.

An optional `--guide-model MODEL` uses a separate MLX embedding teacher for
filtering. Set its `--guide-query-prefix` / `--guide-doc-prefix` to that model's
recipe, independently of the miner's `--query-prefix` / `--doc-prefix`. Models
are encoded sequentially. This is **offline embedding-based filtering**, not a
cross-encoder reranker, online GIST loss, or distillation.

The threshold is `positive_score - abs(positive_score) * relative_margin -
absolute_margin`, using the least-similar known positive in the guide space.
Without a guide, the miner supplies those scores. Inspect shortfalls before
increasing the candidate range or changing margins. Use only training queries
and permitted training documents for mining.

Hardness weighting adds `strength * stop_gradient(cosine_similarity)` to
negative logits before softmax. Known positive targets receive no hardness
penalty, and positive masks still apply. Zero strength disables the penalty;
the selected single/multi-positive objective remains in effect.
This follows the all-negative variant described in the
[Sentence Transformers loss documentation](https://sbert.net/docs/package_reference/sentence_transformer/losses.html).
It has no inference overhead, but quality gains need a held-out ablation.

## Combined smoke workflow

This exercises every new training feature using synthetic examples. The added
positive is a near-duplicate for wiring validation; it is not a quality benchmark.
Choose a new `outputs/docs-demo` directory when repeating the example.

```bash
mkdir -p outputs/docs-demo
python - <<'PYDOC'
import json
from pathlib import Path
rows = [json.loads(line) for line in Path("example_data/train.jsonl").read_text().splitlines()]
for row in rows:
    row["positives"] = [row["positive"] + " This passage explains the same topic."]
with Path("outputs/docs-demo/train.jsonl").open("x") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
PYDOC
python teacher.py \
  --train-pairs outputs/docs-demo/train.jsonl \
  --model mlx-community/nomicai-modernbert-embed-base-bf16 \
  --query-prefix "search_query: " --doc-prefix "search_document: " \
  --batch-size 8 --output outputs/docs-demo/teacher.jsonl
python train.py \
  --train-pairs outputs/docs-demo/train.jsonl --eval-pairs example_data/eval.jsonl \
  --model mlx-community/all-MiniLM-L6-v2-4bit \
  --multi-positive --batch-size 8 --cache-chunk-size 3 --loss-chunk-size 3 \
  --grad-accum-steps 2 --matryoshka-dims 384,128,64 --hardness-strength 2 \
  --teacher-scores outputs/docs-demo/teacher.jsonl --distill-weight 0.2 \
  --lora-dropout 0.1 --epochs 1 --eval-every 3 --best-metric ndcg_at_10 \
  --seed 42 --output-dir outputs/docs-demo/model
python evaluate.py \
  --base-model mlx-community/all-MiniLM-L6-v2-4bit \
  --tuned-model outputs/docs-demo/model --eval-pairs example_data/eval.jsonl \
  --dims 128,64 --json-out outputs/docs-demo/evaluation.json
```

The teacher has 768 embedding dimensions and the student has 384. Only scores
cross that boundary. Both the model directory and report paths must be new.

## Checkpoints and exports

`--save-every` saves adapters independently of evaluation. The final adapters
are always saved. `--keep-checkpoints` retains the newest numbered checkpoints;
the separate `best/` directory is preserved. These are adapter checkpoints,
not a full optimizer/RNG resume facility.

When evaluation is enabled, every completed optimizer window, including a
partial final window, uses the same evaluation and save hooks. The final state
is evaluated if the interval did not already cover it. `--eval-every 0` disables
both periodic and final evaluations. `--merge best` uses the best checkpoint
recorded during this run; if none exists, it exports the final state.

Exports contain merged MLX weights, available tokenizer/configuration files,
`training_config.json`, `training_metadata.json`, and the training log. Metadata
records the chosen step, feature settings, prefixes, data hashes and optional
teacher provenance. It does not copy the teacher model into the student.

QLoRA fusion dequantizes adapted layers; untouched layers can remain quantized.
The export records remaining quantized linear modules and removes stale
per-layer quantization overrides for fused layers. Validate any Transformers,
Sentence Transformers or GGUF conversion separately. Re-embed your corpus when
deploying changed weights, even if the output dimension is unchanged.

For the reasons behind these choices, see [design rationale](design-rationale.md).
For known failure messages, see [troubleshooting](troubleshooting.md).
