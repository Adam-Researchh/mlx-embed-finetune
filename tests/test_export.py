"""Opt-in integration checks; downloads small MiniLM weights if not cached."""
import os
import json
import shutil

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

import train
from evaluate import load_encoder


@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 for model downloads")
@pytest.mark.parametrize("suffix", ["bf16", "4bit"])
def test_export_reload_preserves_trained_embeddings(tmp_path, suffix):
    model_name = f"mlx-community/all-MiniLM-L6-v2-{suffix}"
    model, tokenizer = load_encoder(model_name)
    train.apply_lora_to_model(model, rank=2)
    train.freeze_base_and_enable_lora(model)
    tokens = train.tokenize_batch(tokenizer, ["semantic search", "vector retrieval", "cooking pasta"])
    model.train()

    def loss_fn(m):
        embeddings = train.encode_texts(m, **tokens)
        return train.info_nce_loss(embeddings[:1], embeddings[1:], temperature=.5)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    train.validate_training_step(loss, grads, require_nonzero=True)
    optimizer = optim.Adam(learning_rate=.001)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    model.eval()
    before = np.array(train.encode_texts(model, **tokens).astype(mx.float32))
    train.merge_and_save(model, model_name, str(tmp_path))
    reloaded, _ = load_encoder(str(tmp_path))
    after = np.array(train.encode_texts(reloaded, **tokens).astype(mx.float32))
    # Fusing bf16/fp16 base layers rounds weights; compare both direction and
    # components with a tolerance appropriate to the source precision.
    cosine = np.sum(before * after, axis=1) / (
        np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1))
    assert np.min(cosine) > .999
    np.testing.assert_allclose(after, before, atol=.005, rtol=.03)


@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 for model downloads")
def test_export_removes_stale_per_layer_quantization_overrides(tmp_path, monkeypatch):
    model_name = "mlx-community/all-MiniLM-L6-v2-4bit"
    source = train.get_model_path(model_name)
    model, tokenizer = load_encoder(model_name)
    train.apply_lora_to_model(model, rank=2)
    adapted_names = [name for name, module in model.named_modules() if isinstance(module, train.LoRALinear)]
    custom_source = tmp_path / "source"
    custom_source.mkdir()
    # A valid source configuration can explicitly mark each adapted path as
    # quantized, which overrides the loader's usual saved-scales detection.
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".json", ".txt", ".model"):
            shutil.copy2(path, custom_source / path.name)
    config = json.loads((custom_source / "config.json").read_text())
    for name in adapted_names:
        config["quantization"][name] = True
    (custom_source / "config.json").write_text(json.dumps(config))
    tokens = train.tokenize_batch(tokenizer, ["semantic search", "unrelated document"])
    model.eval()
    before = np.array(train.encode_texts(model, **tokens).astype(mx.float32))
    monkeypatch.setattr(train, "get_model_path", lambda _: custom_source)
    target = tmp_path / "export"
    train.merge_and_save(model, model_name, str(target))
    loaded, _ = load_encoder(str(target))
    after = np.array(train.encode_texts(loaded, **tokens).astype(mx.float32))
    np.testing.assert_allclose(after, before, atol=.005, rtol=.03)


@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 for model downloads")
@pytest.mark.parametrize("suffix", ["bf16", "4bit"])
def test_real_lora_cached_gradients_match_with_all_objectives(suffix):
    from grad_cache import cached_value_and_grad, capture_rng, restore_rng
    from training_data import make_batch, teacher_groups
    from test_cached_training import ROWS, objective_for
    from mlx.utils import tree_flatten

    mx.random.seed(31)
    model, tokenizer = load_encoder(f"mlx-community/all-MiniLM-L6-v2-{suffix}")
    train.apply_lora_to_model(model, rank=2, dropout=.1)
    train.freeze_base_and_enable_lora(model)
    # Exercise gradients through both adapter matrices, not only the nonzero
    # lora_b gradient at the usual all-zero-B initialization.
    for _, module in model.named_modules():
        if isinstance(module, train.LoRALinear):
            module.lora_b = mx.random.normal(module.lora_b.shape) * .001
    model.train()
    mx.eval(model.parameters())
    teacher = {q: {doc: i / 10 for i, doc in enumerate(docs)} for q, docs in teacher_groups(ROWS).items()}
    batch = make_batch(ROWS, multi_positive=True, teacher=teacher)
    chunks = lambda texts: [train.tokenize_batch(tokenizer, texts[i:i + 2]) for i in range(0, len(texts), 2)]
    query_chunks, candidate_chunks = chunks(batch.queries), chunks(batch.candidates)
    objective = objective_for(batch, hardness=2, dims=[384, 128], distill=.2)
    encode = lambda m, tokens: train.encode_texts(m, **tokens)
    initial_rng = capture_rng()

    def reference(m):
        q = mx.concatenate([encode(m, tokens) for tokens in query_chunks])
        c = mx.concatenate([encode(m, tokens) for tokens in candidate_chunks])
        return objective(q, c, 0, len(ROWS))

    expected, expected_grads = nn.value_and_grad(model, reference)(model)
    mx.eval(expected, expected_grads)
    expected_rng = capture_rng()
    restore_rng(initial_rng)
    actual, grads = cached_value_and_grad(model, encode, query_chunks, candidate_chunks, objective, 2)
    train.validate_training_step(actual, grads, require_nonzero=True)
    assert actual.item() == pytest.approx(expected.item(), abs=2e-4)
    assert capture_rng() == expected_rng
    for (name, g), (ename, eg) in zip(tree_flatten(grads), tree_flatten(expected_grads)):
        assert name == ename
        actual_array, expected_array = np.array(g), np.array(eg)
        np.testing.assert_allclose(actual_array, expected_array, atol=5e-4, rtol=.03, err_msg=name)
        expected_norm = np.linalg.norm(expected_array)
        if expected_norm > 1e-8:
            assert np.linalg.norm(actual_array - expected_array) / expected_norm < .02, name
