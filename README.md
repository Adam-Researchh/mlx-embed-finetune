# mlx-embed-finetune

Fine-tune single-vector embedding models on Apple Silicon with MLX. Train LoRA
or QLoRA adapters, evaluate retrieval quality, and export merged MLX weights for
local semantic search.

The training pipeline supports **true multi-positive supervision**, **GradCache**
for larger contrastive batches, and **offline teacher-ranking distillation**.
These features are opt-in and preserve a single-vector serving interface.

## Documentation

| Read this | To understand |
|---|---|
| [Training guide](docs/training.md) | Data format, mining, caching, teacher scores, checkpoints and exports |
| [Evaluation guide](docs/evaluation.md) | Retrieval metrics, dimension sweeps and fair comparisons |
| [Command reference](docs/cli.md) | Every command, flag and default |
| [Design rationale](docs/design-rationale.md) | Why the objectives, caching and serving choices work this way |
| [Model coverage](docs/models.md) | Tested models, prompt requirements and support boundaries |
| [Validation](docs/validation.md) | Test coverage, tested dependency stacks and release evidence |
| [Troubleshooting](docs/troubleshooting.md) | Input, memory, gradient, cache and export failures |
| [Benchmark results](benchmarks/results.md) | Measured memory, retrieval baselines and historical timings |
| [Roadmap](ROADMAP.md) | Remaining work and the evidence required to adopt it |
| [Changelog](CHANGELOG.md) | Published changes and compatibility notes |
| [Contributing](CONTRIBUTING.md) | Development setup and verification expectations |

## Quick start

Use a native Apple Silicon Python environment. The recorded configurations use
Python 3.12 and 3.14; see [validation](docs/validation.md) for exact versions.

```bash
git clone https://github.com/Adam-Researchh/mlx-embed-finetune.git
cd mlx-embed-finetune
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Check model loading, adapter targeting, loss and gradients:
python train.py \
  --train-pairs example_data/train.jsonl \
  --model mlx-community/all-MiniLM-L6-v2-bf16 \
  --batch-size 4 --dry-run

# Complete a small training and export run:
python train.py \
  --train-pairs example_data/train.jsonl --eval-pairs example_data/eval.jsonl \
  --model mlx-community/all-MiniLM-L6-v2-bf16 \
  --epochs 1 --batch-size 4 --best-metric ndcg_at_10 \
  --seed 42 --output-dir outputs/quickstart

python evaluate.py \
  --base-model mlx-community/all-MiniLM-L6-v2-bf16 \
  --tuned-model outputs/quickstart --eval-pairs example_data/eval.jsonl \
  --dims 128,64 --json-out outputs/quickstart-evaluation.json
```

Choose fresh output paths when repeating a run. Model weights are downloaded
on first use. The [bundled examples](example_data/README.md) validate execution;
their easy retrieval scores do not demonstrate a fine-tuning improvement.

## Training capabilities

| Capability | Behavior |
|---|---|
| LoRA and QLoRA | Match attention projections by module path across the tested encoder and decoder-style families |
| Hard negatives | Pool explicit negatives across the contrastive batch while protecting known positives |
| Multi-positive loss | `--multi-positive` supervises all known relevant answers and deduplicates candidates |
| Gradient caching | `--cache-chunk-size` bounds encoding/replay chunks while `--batch-size` sets the shared contrastive pool |
| Ranking distillation | `teacher.py` saves teacher scores; `--teacher-scores` and `--distill-weight` train a student without loading the teacher |
| Matryoshka loss | Train selected embedding prefixes; evaluate the resulting dimension/quality tradeoff |
| Hardness weighting | Optionally increase the contribution of difficult negatives with detached similarities |
| Checkpoint selection | Choose by validation loss, nDCG@10, MRR@10 or Recall@10; restore the chosen checkpoint before export |
| Corpus evaluation | Chunked exact scoring with multiple relevant documents, binary nDCG@10, Recall@k and MRR@10 |

For 64 queries sharing one candidate pool with encoding chunks of eight:

```bash
python train.py \
  --train-pairs data/train.jsonl --eval-pairs data/validation.jsonl \
  --model mlx-community/all-MiniLM-L6-v2-bf16 \
  --multi-positive --batch-size 64 --cache-chunk-size 8 \
  --best-metric ndcg_at_10 --output-dir outputs/cached-run
```

Supply your own `data/` files for this command. The
[combined smoke workflow](docs/training.md#combined-smoke-workflow) is a fully
runnable example of all three new training features using the bundled data.

## Evidence and tradeoffs

- **Memory:** one deterministic MiniLM forward/backward workload on an M1 Ultra
  used **5.060 GiB uncached and 0.400 GiB cached**, a 92.1% reduction in peak MLX
  tensor allocation. Replay increased that single-pass time from 0.350 s to
  0.933 s. [Settings and scope](benchmarks/gradient-cache.md).
- **Retrieval baseline:** untuned MiniLM scored **nDCG@10 0.6455** on 300 SciFact
  test queries against 5,183 documents. This is a reproducible evaluation
  baseline. [Dataset, model revision and recipe](benchmarks/scifact.md).
- **Correctness:** the October 2026 feature release passed **127 local tests**,
  including all 32 objective/dropout combinations, real bf16/4-bit checks, and
  a complete teacher-to-student training/export workflow. Linux and Apple
  Silicon CI also passed. [Validation evidence](docs/validation.md).

Retrieval-quality improvements from a particular recipe require a held-out
ablation. The older BGE-M3 timing comparison is retained as a
[historical measurement](benchmarks/results.md#historical-bge-m3-workflow).

## Scope and deployment

The implemented path expects one embedding vector per text input. The
[coverage table](docs/models.md) distinguishes full training/export tests from
forward/backward checks. Loader compatibility alone does not establish correct
training, pooling or export behavior.

Merged exports use MLX's weight format. QLoRA fusion dequantizes adapted layers;
other layers can remain quantized. Transformers and GGUF conversions require
separate verification. Saved adapter checkpoints do not implement exact
optimizer/RNG resume. GradCache needs stateless encoder forwards apart from
MLX randomness and rejects BatchNorm models.

Match the checkpoint's query/document prompts and normalization during serving.
Deploying changed embedding weights generally requires re-embedding the corpus,
even when the output dimension and serving API stay the same.

## Related projects and references

This project builds on established MLX and retrieval work:

- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) provide the array, model and adapter primitives.
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) supplies the model-loading and inference layer.
- [jina-ai/mlx-retrieval](https://github.com/jina-ai/mlx-retrieval) and [mlx-embeddings-lora](https://github.com/Goekdeniz-Guelmez/mlx-embeddings-lora) are related MLX training projects.
- [Sentence Transformers](https://sbert.net/) documents retrieval losses and training practices.
- [GradCache](https://arxiv.org/abs/2101.06983) and [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) motivate the corresponding training options.

The rationale and implementation links are collected in
[design rationale](docs/design-rationale.md). Contributions are welcome through
[the documented development workflow](CONTRIBUTING.md).

## License

Apache 2.0 for this repository's code. See [LICENSE](LICENSE). Base models and
datasets retain their own licenses.
