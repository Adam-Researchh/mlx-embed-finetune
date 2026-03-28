# mlx-embed-finetune

Fine-tune embedding and encoder models with LoRA on Apple Silicon using MLX.

This repository packages what appears to be the first open-source MLX LoRA fine-tuning pipeline focused on **embedding models**, not LLMs. `mlx-lm` is excellent for decoder-style language models; this project targets **encoders** such as **BERT**, **XLM-RoBERTa**, and **BGE-M3**.

## What This Is

An MLX-native training pipeline for contrastive embedding fine-tuning on Metal.

It is designed for retrieval and semantic search use cases where you want to improve an encoder on domain-specific query/document pairs without leaving the Apple Silicon stack.

## Why It Matters

- PyTorch on Apple Silicon often under-utilizes the GPU for encoder fine-tuning
- This runs entirely on **Metal via MLX**
- Unified memory means no CPU↔GPU transfer overhead during the core training path
- In practice, this produced a major speedup over a standard sentence-transformers workflow

## Benchmarks

Real training numbers from the original pipeline:

| Method | Hardware | Time (9K pairs, 3 epochs) | GPU Usage |
|---|---|---:|---:|
| PyTorch + sentence-transformers | M1 Ultra 128GB | ~6-8 hours | <5% GPU |
| **MLX (this repo)** | M1 Ultra 128GB | **56 minutes** | **78% GPU** |

Dry-run stats:

- Forward + backward: **2.1s/batch** (`batch_size=16`)
- Throughput: **7.6 pairs/sec**
- Memory: **~5-6GB unified** for model + optimizer + gradients
- Steady-state throughput improves after JIT warmup

More detail: see [`benchmarks/results.md`](benchmarks/results.md).

## Quick Start

```bash
pip install mlx mlx-lm mlx-embeddings transformers sentencepiece
python train.py --train-pairs example_data/train.jsonl --eval-pairs example_data/eval.jsonl --epochs 3 --dry-run
```

## Supported Models

- `BAAI/bge-m3` (tested, verified via MLX-converted weights)
- Any MLX-converted BERT / XLM-RoBERTa family encoder from `mlx-community` on Hugging Face

Default model:

- `mlx-community/bge-m3-mlx-fp16`

## Training Data Format

Each line should be JSON:

```json
{"query": "search query", "positive": "relevant document", "negatives": ["irrelevant document"]}
```

Notes:

- `query` and `positive` are required for training
- `negatives` are optional during training because the loss uses in-batch negatives automatically
- `negatives` are helpful for simple evaluation and sanity checks

## Features

- LoRA adapters on attention **Q/V projections**
- Configurable LoRA rank, alpha, and dropout
- **MultipleNegativesRankingLoss / InfoNCE** contrastive training
- Decoupled weight-decay optimizer with linear warmup and cosine decay
- Periodic evaluation with best-model checkpointing
- LoRA adapter checkpoint export
- LoRA merge into full model weights
- HF-format export compatible with downstream conversion workflows including GGUF conversion
- Dry-run mode for speed estimation
- Runs **100% on Metal GPU** through MLX

## Repository Layout

```text
mlx-embed-finetune/
├── README.md
├── LICENSE
├── train.py
├── evaluate.py
├── example_data/
│   ├── train.jsonl
│   └── eval.jsonl
└── benchmarks/
    └── results.md
```

## Training

Example full run:

```bash
python train.py \
  --train-pairs example_data/train.jsonl \
  --eval-pairs example_data/eval.jsonl \
  --model mlx-community/bge-m3-mlx-fp16 \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --output-dir outputs/mlx-embed-finetune
```

### Important arguments

- `--train-pairs`: JSONL training set
- `--eval-pairs`: optional JSONL eval set
- `--model`: MLX-converted encoder model
- `--batch-size`: number of query/positive pairs per step
- `--lora-rank`: LoRA rank
- `--lora-alpha`: LoRA scaling
- `--temperature`: InfoNCE temperature
- `--dry-run`: load one batch and report timing without full training

## Evaluation

`evaluate.py` compares a base model and a fine-tuned model on the same query/relevant/irrelevant test set.

Example:

```bash
python evaluate.py \
  --base-model mlx-community/bge-m3-mlx-fp16 \
  --tuned-model outputs/mlx-embed-finetune \
  --eval-pairs example_data/eval.jsonl
```

It reports:

- average positive-vs-negative similarity gap
- accuracy on the discrimination test
- per-query scores for the base and fine-tuned models

## How It Works

1. Load an MLX-converted encoder model
2. Insert LoRA adapters into attention query/value projections
3. Freeze the base model and train only LoRA weights
4. Use in-batch negatives with contrastive loss
5. Periodically evaluate and save adapter checkpoints
6. Merge LoRA weights back into the base model for export

## Example Data

The `example_data/` directory contains fully synthetic examples across:

- technology documentation
- science articles
- history facts
- cooking recipes
- movie reviews

These files are intentionally generic and safe for public release.

## Caveats

- This pipeline currently targets encoder architectures with attention modules named like `query` and `value`
- You should validate the layer mapping for any new architecture before large training runs
- The included evaluation script is deliberately simple; for production benchmarking you may want MTEB-style tasks or your own retrieval dataset

## License

Apache 2.0. See [`LICENSE`](LICENSE).

## Acknowledgments

Built on top of:

- [MLX](https://github.com/ml-explore/mlx)
- [mlx-lm](https://github.com/ml-explore/mlx-lm)
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)
- encoder models such as BGE-M3 and other BERT/XLM-RoBERTa variants
