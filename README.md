# mlx-embed-finetune

Fine-tune embedding models with LoRA on Apple Silicon using MLX.

`mlx-lm` is excellent for decoder-style *language* models. This project targets
**embedding** models for retrieval and semantic search — BERT, XLM-RoBERTa,
ModernBERT, and decoder-style embedders like EmbeddingGemma — in a single
readable training script you can fork rather than a framework you adopt.

## Prior art — read this before choosing a tool

This is not the first or the most capable MLX embedding-training project, and
an earlier version of this README wrongly implied it was. If you want a
batteries-included option, start with one of these:

| Project | Since | What it gives you |
|---|---|---|
| [`jina-ai/mlx-retrieval`](https://github.com/jina-ai/mlx-retrieval) | Aug 2025 | Embedding **and** reranker training, hard-negative mining, gradient accumulation, MLX Data streaming, MTEB evaluation, W&B logging |
| [`Goekdeniz-Guelmez/mlx-embeddings-lora`](https://github.com/Goekdeniz-Guelmez/mlx-embeddings-lora) | Nov 2025 | Installable CLI: LoRA / DoRA / full / QLoRA, five loss types including GISTEmbed, gradient checkpointing, multiple optimizers |
| [`Blaizzy/mlx-embeddings`](https://github.com/Blaizzy/mlx-embeddings) | Jul 2024 | The inference and model-loading layer this project is built on |

**What this repo is for:** readable training, evaluation, and negative-mining
scripts with shared retrieval utilities and a merged MLX export. Fork it and
change the loss. The focus is small, inspectable workflows on Apple Silicon.

## Quick start

```bash
pip install -r requirements.txt

# Verify your stack works before spending an hour on a real run:
python train.py \
  --train-pairs example_data/train.jsonl \
  --model mlx-community/all-MiniLM-L6-v2-bf16 \
  --batch-size 4 --dry-run
```

The dry run loads the model, applies LoRA, runs one forward+backward pass, and
prints the gradient norm and a projected epoch time. It **fails loudly** if
LoRA matched zero layers, the gradient is exactly zero, or loss/gradients are
nonfinite. Training also checks for nonfinite values before every update.

## Features

- **Architecture-agnostic LoRA targeting.** Adapters are placed by matching
  module *paths*, not by walking one hardcoded block structure, so BERT
  (`attention.self.query`), ModernBERT (`attn.Wqkv` — a fused QKV projection)
  and decoder-style embedders (`self_attn.q_proj`) all work from one script.
  `--target-modules auto` detects the family; `all-linear` adapts every linear
  in a block; explicit names and regexes both still work.
- **Instruction prefixes** (`--query-prefix` / `--doc-prefix`) for models
  pretrained with asymmetric query/document prompts, recorded in the export so
  inference can match training
- Works on **fp16/bf16 and quantized (QLoRA)** base models
- **InfoNCE / MultipleNegativesRankingLoss** with **explicit hard negatives**
  pooled across the batch, not just in-batch negatives
- **False-negative masking** for all known positives of a query, including
  additional relevant documents and positives recorded on other rows
- **Negative mining** with optional embedding-teacher margin filtering
- Optional **hardness-weighted InfoNCE**, controlled by `--hardness-strength`
- **Matryoshka (MRL)** multi-dimension loss, so embeddings stay useful when
  truncated
- **Gradient accumulation** for a larger optimizer batch; the contrastive
  negative pool remains local to each micro-batch
- Explicit L2 normalization, so the InfoNCE temperature means what it says
- AdamW with linear warmup → cosine decay
- Periodic and final evaluation, loss or retrieval-based checkpoint selection,
  independent adapter saving, and checkpoint retention
- **The exported model is the best checkpoint by default**, not whatever the
  last step happened to produce (`--merge final` to opt out)
- Merged MLX export including tokenizer files and run provenance

## Training data format

```json
{"query": "search query", "positive": "relevant document", "negatives": ["a hard negative", "another"]}
```

`negatives` is optional but **strongly recommended**. Every negative in a batch
becomes a negative for every query in that batch, so a handful per row goes a
long way. Without them you are relying on random in-batch negatives, which a
competent base model already separates easily — you will see near-zero loss and
almost no gradient signal.

Additional relevant answers can be listed in `positives`:

```json
{"query":"reset my password","positive":"Open account settings to reset it.","positives":["Use the password recovery page."],"negatives":["Change your display name."]}
```

The `positive` field remains the training target; other known positives are
masked out of the negative pool. Evaluation counts all known positives, merging
relevance for identical query text across rows. Each input row contributes to
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

### Arguments worth knowing

| Flag | Default | Notes |
|---|---|---|
| `--batch-size` | 16 | Micro-batch: queries per forward pass |
| `--grad-accum-steps` | 1 | Effective batch = batch-size × this |
| `--target-modules` | `auto` | `auto`, `all-linear`, names (`query,value`), or regexes |
| `--query-prefix` / `--doc-prefix` | empty | Instruction prefixes |
| `--no-hard-negatives` | off | Ignore `negatives`, in-batch only |
| `--matryoshka-dims` | off | e.g. `1024,512,256,128,64` |
| `--merge` | `best` | `best` or `final` adapter weights for export |
| `--keep-checkpoints` | 3 | `0` keeps everything |
| `--best-metric` | `loss` | `loss`, `ndcg_at_10`, `mrr_at_10`, or `recall_at_10` |
| `--eval-corpus` | none | Additional validation documents, JSONL with `id` / `text` |
| `--eval-every` | 100 | Periodic evaluation plus final evaluation; `0` disables both |
| `--save-every` | 100 | Adapter saves independent of evaluation; final adapters always saved |
| `--hardness-strength` | 0 | Detached cosine penalty on all unmasked negatives; try `2` as an experiment |
| `--temperature` | 0.05 | InfoNCE temperature (assumes unit vectors) |
| `--seed` | none | Set for reproducible shuffling |

## Evaluation

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

For a reproducible public task, see [the SciFact recipe](benchmarks/scifact.md).
Use validation data for `--best-metric ndcg_at_10`; keep the final test set out
of checkpoint selection.

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
negative logits before softmax. The designated positive is unchanged and
known-positive masks still apply. Zero strength gives standard InfoNCE.
This follows the all-negative variant described in the
[Sentence Transformers loss documentation](https://sbert.net/docs/package_reference/sentence_transformer/losses.html).
It has no inference overhead, but quality gains need a held-out ablation.

## Supported models

Anything `mlx-embeddings` can load, in fp16/bf16 or quantized. Verified end to
end:

| Model | Family | Auto-detected as | Modules adapted |
|---|---|---|---|
| `mlx-community/all-MiniLM-L6-v2-bf16` / `-4bit` | BERT | `bert/xlm-roberta` | 12 |
| `mlx-community/nomicai-modernbert-embed-base-bf16` / `-4bit` | ModernBERT | `modernbert` | 44 |
| `mlx-community/embeddinggemma-300m-4bit` | Gemma 3 | `decoder-style` | 48 |
| `mlx-community/bge-m3-mlx-fp16` | XLM-RoBERTa | `bert/xlm-roberta` | 48 |

Two things vary across these families and are handled automatically: the block
structure (matched by path) and the forward signature — encoder models take
`input_ids`, decoder-style embedders take `inputs`, and the model is inspected
once at load to work out which.

### Instruction prefixes matter

Use the exact prompt recipe for your selected checkpoint. Many nomic, E5,
Qwen3-Embedding and EmbeddingGemma models expect asymmetric prompts; model
family alone is not enough to choose them. For example:

```bash
python train.py ... \
  --model mlx-community/nomicai-modernbert-embed-base-bf16 \
  --query-prefix "search_query: " \
  --doc-prefix "search_document: "
```

The prefixes are written into `training_metadata.json`, and `evaluate.py` reads
them back automatically so evaluation matches training.

## Benchmarks

See [`benchmarks/results.md`](benchmarks/results.md). Short version: on an
M1 Ultra, a 9K-pair 3-epoch BGE-M3 run took **56 minutes at ~78% GPU** on MLX
versus **6–8 hours at under 5% GPU** for the equivalent PyTorch +
sentence-transformers workflow on the same machine.

## Example data

`example_data/` is a **wiring test, not a benchmark**. It is 50 synthetic
training rows and 10 eval rows across generic topics, and a stock MiniLM
already scores Recall@1 = 1.0 on it. It exists to prove the pipeline runs, not
to demonstrate that fine-tuning helped.

## Caveats

- If targeting matches nothing, it raises immediately and prints the adaptable
  module paths it *did* find, rather than training zero parameters.
- The three auto-detect presets cover the families listed above. Anything else
  needs an explicit `--target-modules` regex — which is a one-flag change, not
  a code change.
- `LoRALinear` is imported from `mlx_lm.tuner.lora`, an internal path in
  mlx-lm. Verified on mlx-lm 0.29.1 and 0.31.3; the CI smoke test is there to
  catch the day it moves.
- Merging into a quantized base **dequantizes adapted layers**; other layers
  may remain quantized. Metadata records both facts. The export is in MLX
  format; conversion to Transformers or GGUF needs separate validation.
- Saved checkpoints contain adapters and adapter configuration, not optimizer
  or RNG state. Exact training resume is not implemented.
- Gradient accumulation averages several micro-batch gradients. It does not
  give you the larger *in-batch negative pool* that a genuinely larger batch
  would — for that you need GradCache-style caching, which this does not
  implement.

## Tests

```bash
pip install pytest
python -m pytest -q
# Also check real bf16 and QLoRA export/reload parity (downloads MiniLM):
RUN_MODEL_TESTS=1 python -m pytest -q
```

CI runs pure data/mining/metric tests on Linux and the full regression suite,
architecture smoke tests, mining CLI, and export parity checks on Apple Silicon.
GradCache and teacher-score distillation remain follow-up work; this release
establishes the evaluation and correctness foundation for those experiments.

## License

Apache 2.0. See [`LICENSE`](LICENSE).

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) by Prince Canuma
- [Sentence-Transformers](https://www.sbert.net/) for the loss vocabulary this borrows
- Matryoshka Representation Learning ([Kusupati et al., 2022](https://arxiv.org/abs/2205.13147))
