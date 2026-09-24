# Reproducible corpus retrieval evaluation

The example JSONL files test wiring. SciFact provides a harder public check:
300 test queries against 5,183 scientific abstracts, including queries with
multiple relevant documents. This recipe uses the dataset distributed by
[BEIR](https://github.com/beir-cellar/beir), with titles prepended to abstracts
and binary relevance (`score > 0`). It does not require installing BEIR,
PyTorch, or MTEB.

Download the official archive and verify it against the recorded SHA-256:

```bash
mkdir -p outputs/beir
curl -fL https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip \
  -o outputs/beir/scifact.zip
python - <<'PY'
import hashlib
from pathlib import Path
p = Path("outputs/beir/scifact.zip")
expected = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, "Dataset archive changed"
PY
python -m zipfile -e outputs/beir/scifact.zip outputs/beir
python -m benchmarks.prepare_beir \
  --data-dir outputs/beir/scifact --split test --output-dir outputs/scifact-test
```

The converter records hashes of the source corpus, queries and qrels. It writes
one pair row per query with all positive judgments, plus a separate full corpus.
Outputs must be new directories. Graded qrels are collapsed to binary relevance;
do not compare this metric to graded nDCG on other datasets without accounting
for that difference. Documents are identified by exact text in the pair format.

To reproduce the recorded baseline, first resolve the exact model revision:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "mlx-community/all-MiniLM-L6-v2-bf16",
    revision="b6691709eacd8f0afcc3faace288cf50e611f3aa",
    local_dir="outputs/scifact-minilm",
)
PY
python evaluate.py \
  --base-model outputs/scifact-minilm \
  --eval-pairs outputs/scifact-test/pairs.jsonl \
  --corpus outputs/scifact-test/corpus.jsonl \
  --max-length 256 --batch-size 32 --dims 128,64 \
  --query-chunk-size 64 --corpus-chunk-size 4096 \
  --json-out outputs/scifact-baseline.json
```

Measured September 24, 2026, using the untuned MiniLM model and empty prefixes:

| Dimensions | nDCG@10 | MRR@10 | Recall@10 |
|---|---:|---:|---:|
| 384 | 0.6455 | 0.6049 | 0.7850 |
| 128 | 0.5991 | 0.5544 | 0.7583 |
| 64 | 0.4839 | 0.4450 | 0.6145 |

The smaller vectors are truncations of the stock model, **not newly trained
Matryoshka models**. These results validate a useful evaluation path; they do
not demonstrate that hardness weighting or mining improves retrieval. Exact
settings, versions and hashes are in
[the recorded baseline](scifact-baseline-2026-09-24.json). Runtime and speed
comparisons are deliberately omitted because other GPU validation was running.

For a fine-tuning experiment, prepare the official `train` split separately,
split it into training and validation queries before mining, and keep this test
split untouched until the final comparison. Prefer grouping related queries by
source document/topic for internal splits. Train's exact-query overlap check
does not detect every kind of leakage. Use the validation split for checkpoint
selection; never pass the final test split as `--eval-pairs` to `train.py`.

For an ablation, fix the base revision, splits, sequence limits and training
budget; compare standard InfoNCE with mined/filtered negatives and then hardness
weighting over multiple seeds. Report held-out retrieval metrics and uncertainty
alongside memory and training cost before changing defaults.
