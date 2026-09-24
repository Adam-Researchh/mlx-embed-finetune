"""Opt-in integration checks; downloads small MiniLM weights if not cached."""
import os

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
