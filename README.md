# mlx-embed-finetune

Fine-tune **encoder** embedding models with LoRA on Apple Silicon using MLX.

`mlx-lm` is excellent for decoder-style language models. This project targets
**encoders** — BERT, XLM-RoBERTa, BGE — for retrieval and semantic search, in a
single readable training script you can fork rather than a framework you adopt.

## Prior art — read this before choosing a tool

This is not the first or the most capable MLX embedding-training project, and
an earlier version of this README wrongly implied it was. If you want a
batteries-included option, start with one of these:

| Project | Since | What it gives you |
|---|---|---|
| [`jina-ai/mlx-retrieval`](https://github.com/jina-ai/mlx-retrieval) | Aug 2025 | Embedding **and** reranker training, hard-negative mining, gradient accumulation, MLX Data streaming, MTEB evaluation, W&B logging |
| [`Goekdeniz-Guelmez/mlx-embeddings-lora`](https://github.com/Goekdeniz-Guelmez/mlx-embeddings-lora) | Nov 2025 | Installable CLI: LoRA / DoRA / full / QLoRA, five loss types including GISTEmbed, gradient checkpointing, multiple optimizers |
| [`Blaizzy/mlx-embeddings`](https://github.com/Blaizzy/mlx-embeddings) | Jul 2024 | The inference and model-loading layer this project is built on |

**What this repo is for:** two files, no framework, ~1,100 lines total, focused
on the encoder + LoRA + contrastive case with a merged Hugging Face export at
the end (which is what you need for a GGUF conversion). Fork it and change the
loss. If you want features instead of legibility, use one of the above.

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
LoRA matched zero layers or the gradient is exactly zero — the two ways this
kind of pipeline silently trains nothing.

## Features

- LoRA adapters on attention Q/V projections, on **fp16/bf16 and quantized
  (QLoRA)** base models
- **InfoNCE / MultipleNegativesRankingLoss** with **explicit hard negatives**
  pooled across the batch, not just in-batch negatives
- **False-negative masking** — candidates that duplicate a row's own positive
  are masked out instead of being taught as wrong answers
- **Matryoshka (MRL)** multi-dimension loss, so embeddings stay useful when
  truncated
- **Gradient accumulation** for a larger effective batch (more negatives per
  query is the main quality lever in contrastive training)
- Explicit L2 normalization, so the InfoNCE temperature means what it says
- AdamW with linear warmup → cosine decay
- Periodic evaluation, best-checkpoint tracking, checkpoint retention limit
- **The exported model is the best checkpoint by default**, not whatever the
  last step happened to produce (`--merge final` to opt out)
- Merged Hugging Face export including the tokenizer files, ready for GGUF
  conversion

## Training data format

```json
{"query": "search query", "positive": "relevant document", "negatives": ["a hard negative", "another"]}
```

`negatives` is optional but **strongly recommended**. Every negative in a batch
becomes a negative for every query in that batch, so a handful per row goes a
long way. Without them you are relying on random in-batch negatives, which a
competent base model already separates easily — you will see near-zero loss and
almost no gradient signal.

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

### Arguments worth knowing

| Flag | Default | Notes |
|---|---|---|
| `--batch-size` | 16 | Micro-batch: queries per forward pass |
| `--grad-accum-steps` | 1 | Effective batch = batch-size × this |
| `--target-modules` | `query,value` | Attention submodules to adapt |
| `--no-hard-negatives` | off | Ignore `negatives`, in-batch only |
| `--matryoshka-dims` | off | e.g. `1024,512,256,128,64` |
| `--merge` | `best` | `best` or `final` adapter weights for export |
| `--keep-checkpoints` | 3 | `0` keeps everything |
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
   MRR@10**. This is the number that matters.
2. **Pairwise** — accuracy and mean cosine margin against each row's *hardest*
   negative.

For a real benchmark, point this at your own retrieval set, or use
[MTEB](https://github.com/embeddings-benchmark/mteb) on the exported model.

## Supported models

- `BAAI/bge-m3` via `mlx-community/bge-m3-mlx-fp16` (the model this was built
  for) and its 4/6/8-bit variants
- Any MLX-converted BERT / XLM-RoBERTa encoder from `mlx-community` whose
  attention submodules are named `query` / `value` — otherwise pass
  `--target-modules`

Smoke-tested on `mlx-community/all-MiniLM-L6-v2-bf16` and
`mlx-community/all-MiniLM-L6-v2-4bit`.

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

- Targets encoder architectures with an `encoder.layer[*].attention.self`
  stack. Anything else raises immediately with the module names it did find.
- `LoRALinear` is imported from `mlx_lm.tuner.lora`, an internal path in
  mlx-lm. Verified on mlx-lm 0.29.1 and 0.31.3; the CI smoke test is there to
  catch the day it moves.
- Merging into a quantized base **dequantizes** the fused layers; the export is
  full-precision and `training_metadata.json` records that it happened.
- Gradient accumulation averages several micro-batch gradients. It does not
  give you the larger *in-batch negative pool* that a genuinely larger batch
  would — for that you need GradCache-style caching, which this does not
  implement.

## License

Apache 2.0. See [`LICENSE`](LICENSE).

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) by Prince Canuma
- [Sentence-Transformers](https://www.sbert.net/) for the loss vocabulary this borrows
- Matryoshka Representation Learning ([Kusupati et al., 2022](https://arxiv.org/abs/2205.13147))
