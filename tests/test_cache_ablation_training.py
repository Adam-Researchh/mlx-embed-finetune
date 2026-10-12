import json

import mlx.core as mx
import pytest

from benchmarks import scifact_cache_ablation as pilot
from test_training import TinyEncoder, TinyTokenizer
import train


@pytest.mark.parametrize("arm", ["reference", "plain", "autodiff", "float32"])
def test_pilot_updates_and_preserves_rng_during_diagnostics(tmp_path, monkeypatch, arm):
    def model_fixture(seed, float32=False):
        mx.random.seed(seed)
        model = TinyEncoder()
        train.apply_lora_to_model(model, rank=2, alpha=4, dropout=.1, target_spec="query")
        train.freeze_base_and_enable_lora(model)
        mx.eval(model.parameters())
        return model, TinyTokenizer()
    monkeypatch.setattr(pilot, "get_model", model_fixture)
    rows = [{"query_id": str(i), "query": f"question {i}", "positive": f"answer {i}",
             "negatives": [f"wrong {i}"]} for i in range(17)]
    (tmp_path / "train.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    out = tmp_path / "out"
    out.mkdir()
    result = pilot.fit(tmp_path, out, arm, 41)
    assert result["steps"] == 2
    assert all(d["same_rng"] for d in result["diagnostics"])
    assert max(d["relative_l2"] for d in result["diagnostics"]) < 1e-5
    before = mx.load(str(out / "initial_adapters.safetensors"))
    after = mx.load(str(out / "adapters.safetensors"))
    assert any(not mx.array_equal(before[k], after[k]).item() for k in before)
    assert result["timed_training_seconds"] > 0
