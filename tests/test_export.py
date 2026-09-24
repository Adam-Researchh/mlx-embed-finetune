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
