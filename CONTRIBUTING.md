# Contributing

Keep changes inspectable, preserve documented defaults, and verify the behavior
that makes a retrieval experiment trustworthy. Start with [design rationale](docs/design-rationale.md)
and [model coverage](docs/models.md) before changing training or model support.

## Development setup

Use native Apple Silicon Python for MLX training checks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest
python -m pytest -q
RUN_MODEL_TESTS=1 python -m pytest -q
```

The opt-in tests download MiniLM weights when needed. Pure data, mining, teacher
cache and metric tests can run on Linux with NumPy and pytest; see
[validation commands](docs/validation.md#run-the-checks).

## Code map

| File | Responsibility |
|---|---|
| `train.py` | Model loading, adapter targeting, optimization, validation, checkpointing and export |
| `training_data.py` | Candidate pools, positive labels and teacher-cache validation |
| `objectives.py` | Active single/multi-positive, Matryoshka and ranking-distillation objectives |
| `grad_cache.py` | Embedding-gradient caching, encoder replay and PRNG restoration |
| `teacher.py` | Offline teacher score generation |
| `mine.py` | Candidate mining and optional embedding-teacher filtering |
| `evaluate.py` | Model encoding, dimension sweeps and result reports |
| `retrieval.py` | Shared JSONL validation, identities, chunked search and metrics |
| `benchmarks/` | Reproduction utilities and measured results |

## Verification expectations

Loss changes need numerical/reference tests that verify intended positive,
negative and teacher behavior. Cache changes need loss and parameter-gradient
agreement, dropout replay and irregular-chunk checks. New model support needs
reference preprocessing/pooling and export/reload agreement at the intended
precision. Data or metric changes need cases with duplicates, multiple positives
and malformed inputs, plus an independent ranking/metric reference where useful.

Run checks appropriate to the change. Documentation-only changes should verify
links, CLI flags/defaults and executable examples; they do not need new model
training tests just to restate existing behavior. The repository CI will still
run its maintained workflows on pull requests.

Use new output paths under the ignored `outputs/` directory for experiments.
Do not commit model weights, private datasets, teacher scores or credentials.
When publishing a measurement, record hardware, versions, model/dataset
revisions or hashes, parameters, workload and what the metric includes.

## Pull requests

Explain the concrete problem and resulting behavior, relevant validation and
material limits. Update the applicable guide, command reference, model table
and benchmark notes when behavior or claims change. Identify changes to defaults
or data formats clearly. A lower training loss or larger gradient is not enough
to claim better retrieval; use a held-out comparison.
