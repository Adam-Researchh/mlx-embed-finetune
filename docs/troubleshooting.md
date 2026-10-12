# Troubleshooting

Use the smallest dry run that exercises your model and selected features before
starting a long training run. Keep the failing command, traceback, dependency
versions and relevant configuration when reporting an issue.

## Installation and model loading

| Symptom | Action |
|---|---|
| No Metal device or unsupported binary | Use a native arm64 Python environment on an Apple Silicon Mac with GPU access. Check whether the process is running in a restricted, virtualized or headless environment. |
| Missing `mlx_lm.tuner.lora` import | Compare your environment with the [tested configurations](validation.md). Preserve the resolved versions when reporting upstream API drift. |
| Model type unsupported | Confirm the checkpoint can load through the installed `mlx-embeddings`; a model hosted on Hugging Face alone does not establish MLX compatibility. |
| LoRA adapted zero layers | Read the available module paths printed by the error. Choose a supported preset or an explicit suffix/regex; do not bypass the zero-parameter guard. |
| Token-level output rejected | The trainer requires one vector per input. Multi-vector and multimodal paths need separate support; see [model coverage](models.md). |

## Training and memory

`--batch-size` is the logical contrastive batch. `--cache-chunk-size` is the
number of texts encoded or replayed at once. If activation memory is too high,
enable caching and reduce that chunk size. Reduce `--loss-chunk-size` if the
embedding loss uses too much memory. If memory is still excessive, reduce the
logical batch, candidate count, sequence length or model size: embeddings,
labels, teacher-score matrices and gradients remain resident.

Gradient accumulation changes the optimizer batch but retains separate
contrastive pools. It cannot substitute for a larger shared negative pool.
Caching adds computation, so a slower update can still be the intended memory
tradeoff. [The benchmark](../benchmarks/gradient-cache.md) records both.

A zero-gradient dry run can mean the batch has no useful contrastive signal,
for example one query with only its positive or a pool containing only duplicate
answers. Inspect labels and use relevant hard negatives. Very small loss on easy
examples is not evidence of a useful domain adaptation.

Nonfinite loss or gradients stop training before an optimizer update. Check
input text, temperature, learning rate and model numerics. Retrieval evaluation
also rejects zero/nonfinite vectors and overflowing similarities. Do not treat
an arbitrary ranking of degenerate vectors as a successful benchmark.

An unsupported PRNG-key error prevents unverifiable replay. Use a tested MLX
configuration and report the version. Do not assign a replacement object to
`mx.random.state`; current MLX releases can expose an immutable state sentinel.
BatchNorm models are rejected because replay can update forward state twice.

## Data and teacher caches

Pair JSONL requires nonempty `query` and `positive` strings. Optional `positives`
and `negatives` must be lists of nonempty strings. Parsing fails with a file and
line number for malformed rows. Exact train/evaluation query overlap is rejected;
other forms of leakage still require careful splitting.

If mining reports a shortfall, inspect the candidate range and score margins.
The miner does not relax its filters automatically. A larger candidate range
can help, but do not label alternate correct answers as negatives to fill a quota.

Teacher distillation requires a matching dataset hash and complete score groups.
Regenerate scores after editing or mining the dataset; manually replacing the
hash defeats the consistency check. Unknown, duplicate, missing and nonfinite
scores are errors. Scores must be calibrated so larger means more relevant.

Use `--teacher-scores` together with a positive `--distill-weight`, or omit both.
Distillation requires normalized embeddings and the explicit candidate groups;
it cannot be combined with `--no-normalize` or `--no-hard-negatives`. At least two
valid scored candidates must remain per query. In single-positive mode,
alternate positives are still masked; enable multi-positive training or provide
real negatives when those masks leave only one candidate.

## Checkpoints and output files

Training output directories must be empty. Mining, teacher scoring and evaluation
reports require fresh output paths. Choose a new run name instead of deleting a
previous result automatically. See the [training guide](training.md#checkpoints-and-exports)
for checkpoint retention and best/final export behavior.

`--eval-every 0` disables all evaluation. With evaluation enabled, the final
state is evaluated even if the run is shorter than the interval. A run without
a recorded best checkpoint exports final weights. Adapter checkpoints do not
contain enough state for exact optimizer/RNG resume.

If export/reload fails, record whether the base is quantized and preserve
`config.json` plus the export metadata. Adapted QLoRA layers become dense while
other layers can remain quantized. The supported export format is MLX; another
framework or converter needs separate verification.

## Unexpected retrieval scores

Match the checkpoint's prompt recipe, model revision, sequence limit,
normalization and output dimensions. For local tuned exports, evaluation can
read saved prefixes; malformed metadata raises. Other settings, including
maximum length, must be supplied consistently.

The small example set is already solved by the base model. Use the
[SciFact recipe](../benchmarks/scifact.md) or a meaningful held-out domain set.
Report corpus size and all known positives, and distinguish pairwise diagnostics
from corpus-wide retrieval. Apply the same serving preprocessing and re-embed
the corpus after changing the weights.
