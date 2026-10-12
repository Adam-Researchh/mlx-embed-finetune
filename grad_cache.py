"""Exact gradient caching for stateless embedding forwards on MLX.

Encode without a backward graph, differentiate the shared embedding loss, then
replay encoder chunks with cached embedding gradients and identical randomness.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map


def capture_rng():
    """Represent MLX's two-word key as a seed using its public key/seed APIs.

    New MLX releases expose an immutable thread-local random-state sentinel;
    assigning to mx.random.state silently disconnects it from the real PRNG.
    Check the key round-trip rather than assuming mutable global state.
    """
    key = mx.array(mx.random.state[0])
    if key.shape != (2,) or key.dtype != mx.uint32:
        raise RuntimeError("Unsupported MLX PRNG key format for gradient cache replay")
    words = key.tolist()
    seed = (int(words[0]) << 32) | int(words[1])
    if not bool(mx.array_equal(mx.random.key(seed), key).item()):
        raise RuntimeError("MLX PRNG key cannot be restored exactly")
    return seed


def restore_rng(seed):
    mx.random.seed(seed)


def cached_value_and_grad(model, encode, query_chunks, candidate_chunks, loss_for_rows,
                          loss_chunk_size=64):
    """loss_for_rows(q, c, start, stop) returns the mean loss for those query rows.

    Encoder activation memory is bounded by an encoding chunk. Embeddings,
    token inputs, labels and embedding gradients remain resident. Each loss
    gradient graph holds at most loss_chunk_size x candidate_count logits.
    """
    if not query_chunks or not candidate_chunks or loss_chunk_size < 1:
        raise ValueError("Nonempty encoding chunks and a positive loss chunk size are required")
    if any(isinstance(module, nn.BatchNorm) for _, module in model.named_modules()):
        raise ValueError("Gradient caching does not support stateful BatchNorm forwards")
    initial_rng = capture_rng()
    after_cache_rng = initial_rng
    states, embeddings = [], []
    chunks = [*query_chunks, *candidate_chunks]
    try:
        for tokens in chunks:
            states.append(capture_rng())
            out = mx.stop_gradient(encode(model, tokens))
            mx.eval(out)
            if out.ndim != 2:
                raise ValueError("Gradient caching requires one embedding vector per input")
            embeddings.append(out)
        after_cache_rng = capture_rng()
        q = mx.concatenate(embeddings[:len(query_chunks)])
        c = mx.concatenate(embeddings[len(query_chunks):])
        mx.eval(q, c)
        del embeddings

        query_grads, candidate_grad = [], None
        loss = mx.array(0.0)
        for start in range(0, q.shape[0], loss_chunk_size):
            stop = min(start + loss_chunk_size, q.shape[0])
            value_grad = mx.value_and_grad(
                lambda queries, docs: loss_for_rows(queries, docs, start, stop), argnums=(0, 1))
            value, (gq, gc) = value_grad(q[start:stop], c)
            weight = (stop - start) / q.shape[0]
            loss = loss + value * weight
            query_grads.append(gq * weight)
            candidate_grad = gc * weight if candidate_grad is None else candidate_grad + gc * weight
            mx.eval(loss, query_grads[-1], candidate_grad)
        query_grad = mx.concatenate(query_grads)
        mx.eval(query_grad)

        def surrogate(m, tokens, cached_grad):
            return mx.sum(encode(m, tokens) * mx.stop_gradient(cached_grad))

        replay = nn.value_and_grad(model, surrogate)
        grads = None
        state_index = 0
        for token_chunks, embedding_grad in ((query_chunks, query_grad), (candidate_chunks, candidate_grad)):
            offset = 0
            for tokens in token_chunks:
                count = next(iter(tokens.values())).shape[0]
                restore_rng(states[state_index])
                value, chunk_grad = replay(model, tokens, embedding_grad[offset:offset + count])
                mx.eval(value, chunk_grad)
                grads = chunk_grad if grads is None else tree_map(mx.add, grads, chunk_grad)
                mx.eval(grads)
                offset += count
                state_index += 1
        return loss, grads
    finally:
        # Replay consumes no extra randomness from the caller's perspective.
        restore_rng(after_cache_rng)
