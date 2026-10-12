# Model coverage

The trainer expects a model that returns one embedding vector per input text.
It detects token-input argument names and LoRA projection paths, but model
support also depends on tokenizer behavior, attention masks, pooling,
normalization and export/reload behavior.

## Recorded coverage

| Model | Family and auto preset | Adapted modules | Evidence |
|---|---|---:|---|
| `mlx-community/all-MiniLM-L6-v2-bf16` | BERT, `bert/xlm-roberta` | 12 | Training/export/reload, parameter-gradient parity, and real LoRA checks with all objectives and dropout |
| `mlx-community/all-MiniLM-L6-v2-4bit` | BERT, `bert/xlm-roberta` | 12 | QLoRA training/export/reload, per-layer quantization configuration, cache parity and the full combined-feature workflow |
| `mlx-community/nomicai-modernbert-embed-base-bf16` | ModernBERT, `modernbert` | 44 | CI forward/backward checks and offline teacher scoring |
| `mlx-community/nomicai-modernbert-embed-base-4bit` | ModernBERT, `modernbert` | 44 | Local combined-feature forward/backward check with cache, multi-positive loss, distillation, MRL, hardness and dropout |
| `mlx-community/embeddinggemma-300m-4bit` | Gemma 3, `decoder-style` | 48 | Execution checks pass, but an upstream padding defect is confirmed; see the limitation below |
| `mlx-community/bge-m3-mlx-fp16` | XLM-RoBERTa, `bert/xlm-roberta` | 48 | Original training workflow and historical measurements; not the current full export-parity CI fixture |

Coverage is recorded for the October 2026 release and earlier historical
measurements; see [validation](validation.md) and [benchmark results](../benchmarks/results.md).
A forward/backward check proves execution with a finite nonzero gradient. The
full export-parity checks are currently concentrated on MiniLM. This table does
not certify every quantization, long-context configuration or upstream revision.

## Known quantized EmbeddingGemma limitation

With `mlx-embeddings==0.1.0` and the audited MLX stack, the quantized Gemma
implementation casts its padding mask to the packed embedding weight's
`uint32` dtype. This removes the mask. The unchanged base model can therefore
produce substantially different vectors for the same text when the other texts
in its padded batch change. Gradient caching's smaller encoding chunks can
expose the same defect, but it also occurs without training or caching.

Treat this checkpoint as experimental on that stack. Use the verified MiniLM
path for training and retrieval until upstream padding behavior is corrected
and retested. A finite-gradient CI smoke test does not establish masking
correctness. The [audit record](adversarial-audit-2026-10-12.md) contains the
reproduction and distinguishes this issue from the fixed distillation bug.

## Adapter targeting

The automatic presets match attention query/value projections for BERT/XLM-R
and decoder-style models. ModernBERT uses fused `attn.Wqkv` and `attn.Wo`
projections. `--target-modules all-linear` selects linear layers inside the
recognized transformer block paths. Explicit suffix names and regular
expressions are also supported.

An explicit target specification is comma-separated, so avoid commas inside an
individual regular expression. Matching zero adaptable layers is an error that
prints examples of available paths. Quantized linear layers are handled
explicitly. Gradient norms across different target sets are diagnostics, not
evidence that one target set improves retrieval more than another.

## Match the checkpoint input recipe

Read the selected checkpoint's model card before choosing query/document
prefixes, pooling assumptions and sequence lengths. Families can contain
checkpoints with different recipes. For the Nomic ModernBERT example:

```bash
python train.py \
  --train-pairs example_data/train.jsonl \
  --model mlx-community/nomicai-modernbert-embed-base-bf16 \
  --query-prefix "search_query: " --doc-prefix "search_document: " \
  --batch-size 4 --dry-run
```

Training records its prefixes. Evaluation reads them from a local tuned model's
metadata unless overridden. Teacher scoring has its own prefix arguments;
those need not equal the student's. Sequence limits must fit the model and the
available memory. A model's advertised context window is not a tested training
batch size for this project.

## Adding another model

Model-loading support upstream is a starting point. Before adding a model to
the verified table, check its output rank, preprocessing, pooling and attention
behavior against the reference implementation; confirm intended LoRA targets;
test finite gradients, cached/uncached parity and dropout; and verify exported
embeddings after reload. Include the intended dimensions and quantization.

The current loss and evaluator accept one vector per text. Token-level
multi-vector output and image/audio/video input paths require additional
implementation. The trainer rejects token-level outputs rather than silently
pooling them using an arbitrary rule. Candidate additions are tracked in the
[roadmap](../ROADMAP.md).
