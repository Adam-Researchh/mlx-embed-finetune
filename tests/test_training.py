import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import train


def test_hardness_zero_matches_reference_and_mask_blocks_gradient():
    q = mx.array([[1., 0.], [0., 1.]])
    c = mx.array([[1., 0.], [0., 1.], [.8, .6]])
    logits = np.array(q) @ np.array(c).T / .5
    reference = np.mean(np.log(np.exp(logits).sum(axis=1)) - logits[np.arange(2), np.arange(2)])
    assert train.info_nce_loss(q, c, .5).item() == pytest.approx(reference)
    assert train.info_nce_loss(q, c, .5, hardness_strength=3).item() > reference
    mask = mx.array([[0., 0., -1e9], [0., 0., -1e9]])
    grad = mx.grad(lambda candidates: train.info_nce_loss(q, candidates, .5, mask, 3))(c)
    np.testing.assert_array_equal(np.array(grad)[2], [0., 0.])


def test_hardness_similarity_is_detached():
    q = mx.array([[1., 0.]])
    c = mx.array([[.8, .6], [.6, .8]])
    strength, temperature = 2., .5
    actual = mx.grad(lambda x: train.info_nce_loss(x, c, temperature, hardness_strength=strength))(q)
    logits = np.array(q) @ np.array(c).T / temperature
    logits[0, 1] += strength * float((q @ c.T)[0, 1].item())
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[0, 0] -= 1
    expected = probs @ np.array(c) / temperature
    np.testing.assert_allclose(np.array(actual), expected, atol=1e-6)


def test_mask_includes_other_known_positives():
    mask = train.build_false_negative_mask(["a", "b"], ["c", "a"], [{"a", "b"}, {"b"}])
    np.testing.assert_array_equal(np.array(mask), [[0, -1e9, 0, -1e9], [0, 0, 0, 0]])


def test_mask_cannot_be_overridden_by_hardness():
    q = mx.array([[1., 0.]])
    c = mx.array([[1., 0.], [1., 0.]])
    mask = train.build_false_negative_mask(["same"], ["same"])
    assert train.info_nce_loss(q, c, .05, mask, hardness_strength=1e10).item() == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_or_gradient_rejected(bad):
    with pytest.raises(FloatingPointError):
        train.validate_training_step(mx.array(bad), {"x": mx.array([1.])})
    with pytest.raises(FloatingPointError):
        train.validate_training_step(mx.array(1.), {"x": mx.array([bad])})


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(4, 4, bias=False)

    def __call__(self, x):
        return self.query(x)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.layers = [Block()]

    def __call__(self, input_ids, attention_mask):
        hidden = self.layers[0](self.embedding(input_ids))
        pooled = (hidden * attention_mask[..., None]).sum(axis=1) / attention_mask.sum(axis=1, keepdims=True)
        return SimpleNamespace(text_embeds=pooled)


class TinyTokenizer:
    def __call__(self, texts, max_length=32, **kwargs):
        tokens = [[ord(c) % 31 + 1 for c in text[:max_length]] for text in texts]
        length = max(map(len, tokens))
        return {"input_ids": np.array([t + [0] * (length - len(t)) for t in tokens]),
                "attention_mask": np.array([[1] * len(t) + [0] * (length - len(t)) for t in tokens])}


def setup_run(tmp_path, monkeypatch, rows=9, extra=()):
    train_path, eval_path = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    train_path.write_text("".join(json.dumps({"query": f"q{i}", "positive": f"answer{i}",
                                            "negatives": ["distractor"]}) + "\n" for i in range(rows)))
    eval_path.write_text('{"query":"held out","positive":"validation","negatives":["other"]}\n')
    args = train.build_parser().parse_args([
        "--train-pairs", str(train_path), "--eval-pairs", str(eval_path),
        "--output-dir", str(tmp_path / "out"), "--target-modules", "query", "--lora-rank", "2",
        "--epochs", "1", "--batch-size", "2", "--grad-accum-steps", "2",
        "--eval-every", "3", "--save-every", "0", "--log-every", "1", *extra,
    ])
    monkeypatch.setattr(train, "get_model_path", lambda name: tmp_path)
    monkeypatch.setattr(train, "load_model", lambda *a, **kw: TinyEncoder())
    monkeypatch.setattr(train, "load_tokenizer", lambda *a: TinyTokenizer())
    eval_calls, exported = [], []
    monkeypatch.setattr(train, "evaluate_loss", lambda *a: eval_calls.append(True) or .5)
    monkeypatch.setattr(train, "merge_and_save", lambda *a, **kw: exported.append(kw["run_metadata"]))
    return args, eval_calls, exported


@pytest.mark.parametrize("rows,expected_steps", [(9, 3), (5, 2)])
def test_partial_windows_and_short_runs_evaluate_final_once(tmp_path, monkeypatch, rows, expected_steps):
    args, calls, exported = setup_run(tmp_path, monkeypatch, rows)
    train.train(args)
    assert len(calls) == 1
    assert exported[0]["best_step"] == exported[0]["selected_step"] == expected_steps
    log = [json.loads(line) for line in (tmp_path / "out/training_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log if "eval_metrics" in r] == [expected_steps]
    assert (tmp_path / f"out/checkpoint-{expected_steps}/adapters.safetensors").exists()


def test_no_evaluation_still_saves_final_adapter(tmp_path, monkeypatch):
    args, calls, exported = setup_run(tmp_path, monkeypatch, extra=["--eval-every", "0"])
    train.train(args)
    assert not calls and exported[0]["best_step"] is None
    assert exported[0]["selected_step"] == 3
    assert (tmp_path / "out/checkpoint-3/adapters.safetensors").exists()


def test_periodic_saving_and_retention_without_evaluation(tmp_path, monkeypatch):
    args, calls, _ = setup_run(tmp_path, monkeypatch, extra=["--eval-every", "0",
                                                           "--save-every", "1", "--keep-checkpoints", "2"])
    train.train(args)
    assert not calls
    assert sorted(p.name for p in (tmp_path / "out").glob("checkpoint-*")) == ["checkpoint-2", "checkpoint-3"]


def test_old_best_directory_rejected_before_model_load(tmp_path, monkeypatch):
    args, calls, exported = setup_run(tmp_path, monkeypatch)
    (tmp_path / "out/best").mkdir(parents=True)
    monkeypatch.setattr(train, "load_model", lambda *a, **kw: pytest.fail("should not load a model"))
    with pytest.raises(ValueError, match="not empty"):
        train.train(args)
    assert not calls and not exported


def test_retrieval_metric_maximizes_and_restores_best(tmp_path, monkeypatch):
    import evaluate
    args, calls, exported = setup_run(tmp_path, monkeypatch,
                                    extra=["--eval-every", "1", "--best-metric", "ndcg_at_10",
                                           "--keep-checkpoints", "1"])
    scores = iter([.9, .8, .7])
    monkeypatch.setattr(evaluate, "evaluate_loaded_model", lambda *a, **kw: {
        "retrieval": {"ndcg_at_10": next(scores), "mrr_at_10": .5, "recall_at": {10: .7}}})
    train.train(args)
    assert exported[0]["selected_step"] == 1
    assert exported[0]["best_value"] == .9
    assert not (tmp_path / "out/checkpoint-1").exists()
    assert (tmp_path / "out/best/adapters.safetensors").exists()


@pytest.mark.parametrize("flags", [["--temperature", "nan"], ["--batch-size", "0"],
                                   ["--hardness-strength", "-1"], ["--lora-rank", "0"],
                                   ["--matryoshka-dims", "0,4"], ["--log-every", "0"]])
def test_invalid_settings_rejected(flags):
    args = train.build_parser().parse_args(["--train-pairs", "unused", *flags])
    with pytest.raises(ValueError):
        train.validate_args(args)


def test_single_optimizer_step_actually_changes_adapters(tmp_path, monkeypatch):
    args, _, _ = setup_run(tmp_path, monkeypatch, rows=1,
                           extra=["--eval-every", "0", "--weight-decay", "0"])
    mx.random.seed(11)
    snapshots = {}
    freeze = train.freeze_base_and_enable_lora

    def capture_before(model):
        freeze(model)
        snapshots["before"] = {name: np.array(value) for name, value in train.lora_weight_dict(model).items()}
        return model

    def capture_after(model, *args, **kwargs):
        snapshots["after"] = {name: np.array(value) for name, value in train.lora_weight_dict(model).items()}

    monkeypatch.setattr(train, "freeze_base_and_enable_lora", capture_before)
    monkeypatch.setattr(train, "merge_and_save", capture_after)
    train.train(args)
    assert any(not np.array_equal(before, snapshots["after"][name])
               for name, before in snapshots["before"].items())


def test_half_precision_normalization_handles_tiny_and_zero_vectors():
    x = mx.array([[1e-5, 0.], [0., 0.]], dtype=mx.float16)
    normalized = np.array(train.l2_normalize(x))
    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(normalized[0], [1., 0.], atol=1e-5)


def test_logged_learning_rate_is_the_one_actually_applied(tmp_path, monkeypatch):
    args, _, _ = setup_run(tmp_path, monkeypatch, extra=["--eval-every", "0"])
    applied = []
    original = train.opt.AdamW.update

    def record_update(optimizer, *args, **kwargs):
        original(optimizer, *args, **kwargs)
        applied.append(float(optimizer.learning_rate.item()))

    monkeypatch.setattr(train.opt.AdamW, "update", record_update)
    train.train(args)
    entries = [json.loads(line) for line in (tmp_path / "out/training_log.jsonl").read_text().splitlines()]
    assert [row["lr"] for row in entries if "lr" in row] == applied
