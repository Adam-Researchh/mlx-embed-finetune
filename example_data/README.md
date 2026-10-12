# Example data

`train.jsonl` contains 50 synthetic training rows and `eval.jsonl` contains ten
synthetic evaluation rows across generic topics. They exercise loading,
tokenization, adapter training and retrieval reporting.

These examples are intentionally small and easy. The base MiniLM model already
achieves Recall@1 = 1.0 on the evaluation file. Use them for execution checks,
not to establish improved retrieval or compare training recipes.

Each row contains a query, designated positive and explicit negatives. The
[training guide](../docs/training.md#training-data-format) documents optional
additional positives and the
[combined smoke workflow](../docs/training.md#combined-smoke-workflow) shows how
CI constructs a two-positive fixture without altering these original files.

For meaningful evaluation, use a held-out domain dataset or the
[SciFact recipe](../benchmarks/scifact.md). Keep validation and final test data
separate from training, mining and teacher scoring.
