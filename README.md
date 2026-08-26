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
| `--target-modules` | `auto` | `auto`, `all-linear`, names (`query,value`), or regexes |
| `--query-prefix` / `--doc-prefix` | empty | Instruction prefixes |
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

nomic-embed, E5, BGE, Qwen3-Embedding and EmbeddingGemma are all pretrained
with asymmetric prompts. Fine-tuning without them trains the model off its own
distribution:

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
