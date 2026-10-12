# Evaluation guide

Use validation data to select checkpoints and a separate test set to compare
finished models. Loss and retrieval quality answer different questions; use
corpus retrieval metrics to decide whether a training recipe is useful.

## Compare models

```bash
python evaluate.py \
  --base-model mlx-community/bge-m3-mlx-fp16 \
  --tuned-model outputs/run1 \
  --eval-pairs data/eval.jsonl \
  --show-rows --json-out results.json
```

Reports two views, base vs tuned:

1. **Retrieval** — every positive and negative in the file is pooled into one
   corpus and each query is ranked against all of it: **Recall@1/3/5/10 and
   MRR@10 and binary nDCG@10**. Recall counts retrieved relevant documents
   divided by all known relevant documents for each row.
2. **Pairwise** — accuracy and mean cosine margin against each row's *hardest*
   negative.

Add `--corpus data/corpus.jsonl` to search a larger corpus (`{"id":"doc-1",
"text":"document content"}` per line). Pair documents are included automatically;
identical text is deduplicated. `--query-chunk-size` and `--corpus-chunk-size`
bound score-matrix memory. Ties use corpus order deterministically. Embeddings
are still held in memory; this is exact retrieval, not an ANN index.

Use `--dims 512,256,128` for a Matryoshka evaluation sweep (dimensions must fit
the model). Truncated embeddings are re-normalized. Full dimension is always
reported. Omit `--tuned-model` for a baseline-only run. JSON results include
per-query metrics, file hashes, runtime versions, and the evaluation settings.
Report paths must be new files: `--json-out` never overwrites existing inputs or
reports. Invalid prompt metadata, zero/nonfinite embeddings, and overflowing
similarities raise instead of silently producing misleading metrics.

For a reproducible public task, see [the SciFact recipe](../benchmarks/scifact.md).
Use validation data for `--best-metric ndcg_at_10`; keep the final test set out
of checkpoint selection.


## Baselines and fair comparisons

Omit `--tuned-model` for a baseline-only evaluation. The same text corpus and
prefixes are used for both models in a base/tuned comparison. If the tuned
model is a local directory, its recorded prefixes are the defaults; explicit
CLI prefixes override them. Other settings such as `--max-length` must be
chosen explicitly to match the intended serving setup.

Do not use the shared-prefix comparison for unrelated models that require
different input recipes. Run separate baseline-only evaluations with each
model's correct prefixes, holding the corpus, relevance judgments and relevant
length limits fixed. Model capabilities and input instructions are documented
in [model coverage](models.md).

Each JSONL row contributes equally to reported retrieval means. Repeated query
text shares the union of known positive documents but still contributes once
per row. Deduplicate query rows before evaluation if your benchmark expects one
measurement per unique query. Exact duplicate document text is collapsed.

Recall@k is the fraction of all known relevant documents retrieved within k.
MRR@10 uses the rank of the first relevant document and is zero when no relevant
document appears in the first ten results. Binary nDCG@10 accounts for the
positions of all relevant documents in the top ten; relevance grades are not
represented in the pair format.

Pairwise accuracy compares the designated positive with its hardest listed
negative. Known positives are excluded from that negative list. Rows with no
remaining negatives are omitted from the pairwise view. This diagnostic does
not replace corpus-wide retrieval evaluation.

## Validate a training change

1. Split queries and source documents/topics before mining. Keep final test
   judgments out of teacher scoring and checkpoint selection.
2. Fix the model revision, preprocessing, sequence limits, dataset hashes,
   training budget and evaluation corpus.
3. Compare the base model with a standard training run, then change one feature
   at a time: mining, multi-positive loss, cache batch size, hardness or distillation.
4. Repeat with multiple seeds and report paired uncertainty where practical.
   Include out-of-domain retrieval to detect regressions.
5. Measure training memory/time and serving latency separately from retrieval
   quality. Compare the dimensions and quantization you intend to deploy.

A larger contrastive pool changes the task's loss denominator. Multi-positive
cross entropy has a different target distribution, and distillation adds another
term. Raw training-loss values therefore cannot rank these recipes fairly.
Validation loss uses the selected single/multi-positive labels at full dimension,
without training teacher scores or hardness penalties. It remains dependent on
the validation candidate pool. A fixed retrieval benchmark is the stronger
cross-recipe comparison.

See [benchmark results](../benchmarks/results.md) for the current measurements,
[validation coverage](validation.md) for correctness checks, and
[the command reference](cli.md#evaluation) for flags.
