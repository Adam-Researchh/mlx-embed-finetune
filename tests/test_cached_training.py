import itertools
import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

import train
from grad_cache import cached_value_and_grad, capture_rng, restore_rng
from objectives import embedding_objective, positive_cross_entropy, ranking_kl
from training_data import make_batch, teacher_groups
from test_training import TinyEncoder, TinyTokenizer, setup_run


ROWS = [
    {"query": "short", "positive": "answer one", "positives": ["second answer"], "negatives": ["wrong one"]},
    {"query": "longer question", "positive": "second answer", "negatives": ["another wrong answer"]},
    {"query": "short", "positive": "third answer", "negatives": ["wrong one", "third wrong"]},
]


class DropoutEncoder(TinyEncoder):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def __call__(self, input_ids, attention_mask):
        hidden = self.dropout(self.layers[0](self.embedding(input_ids)))
        pooled = (hidden * attention_mask[..., None]).sum(axis=1) / attention_mask.sum(axis=1, keepdims=True)
        return SimpleNamespace(text_embeds=pooled)


def token_chunks(texts, size):
    return [train.tokenize_batch(TinyTokenizer(), texts[i:i + size]) for i in range(0, len(texts), size)]


def encode(model, tokens):
    return train.encode_texts(model, **tokens)


def objective_for(batch, hardness=0, dims=None, distill=0):
    positives, valid = mx.array(batch.positive_mask), mx.array(batch.valid_mask)
    scores = mx.array(batch.teacher_scores) if distill else None
    mask = mx.array(batch.teacher_mask) if distill else None

    def loss(q, c, start, stop):
        return embedding_objective(q, c, positives[start:stop], valid[start:stop], .3, hardness, dims,
                                   scores[start:stop] if distill else None,
                                   mask[start:stop] if distill else None, distill, .2)
    return loss


@pytest.mark.parametrize("multi,mrl,hardness,distill,dropout", list(itertools.product(
    (False, True), (False, True), (0, 2), (0, .3), (0, .2))))
def test_cached_loss_and_every_parameter_gradient_match_reference(multi, mrl, hardness, distill, dropout):
    mx.random.seed(17)
    model = DropoutEncoder(dropout)
    model.train()
    mx.eval(model.parameters())
    teacher = {q: {d: i / 10 for i, d in enumerate(docs)} for q, docs in teacher_groups(ROWS).items()}
    batch = make_batch(ROWS, multi_positive=multi, teacher=teacher if distill else None)
    q_chunks, c_chunks = token_chunks(batch.queries, 2), token_chunks(batch.candidates, 3)
    loss = objective_for(batch, hardness, [4, 2] if mrl else None, distill)
    before = capture_rng()

    def reference(m):
        q = mx.concatenate([encode(m, tokens) for tokens in q_chunks])
        c = mx.concatenate([encode(m, tokens) for tokens in c_chunks])
        return loss(q, c, 0, len(ROWS))

    expected, expected_grads = nn.value_and_grad(model, reference)(model)
    mx.eval(expected, expected_grads)
    after_reference = capture_rng()
    restore_rng(before)
    actual, grads = cached_value_and_grad(model, encode, q_chunks, c_chunks, loss, loss_chunk_size=2)
    mx.eval(actual, grads)
    assert float(actual.item()) == pytest.approx(float(expected.item()), rel=2e-5, abs=2e-6)
    assert capture_rng() == after_reference
    for (name, g), (expected_name, eg) in zip(tree_flatten(grads), tree_flatten(expected_grads)):
        assert name == expected_name
        np.testing.assert_allclose(np.array(g), np.array(eg), atol=3e-5, rtol=3e-4, err_msg=name)


def test_cached_deterministic_gradient_matches_one_unsplit_forward():
    mx.random.seed(7)
    model = TinyEncoder()
    batch = make_batch(ROWS, multi_positive=True)
    loss = objective_for(batch)
    q, c = token_chunks(batch.queries, 100)[0], token_chunks(batch.candidates, 100)[0]
    expected, eg = nn.value_and_grad(model, lambda m: loss(encode(m, q), encode(m, c), 0, 3))(model)
    actual, ag = cached_value_and_grad(model, encode, token_chunks(batch.queries, 1),
                                      token_chunks(batch.candidates, 2), loss, 1)
    assert actual.item() == pytest.approx(expected.item(), abs=2e-6)
    for (_, a), (_, e) in zip(tree_flatten(ag), tree_flatten(eg)):
        np.testing.assert_allclose(np.array(a), np.array(e), atol=2e-5, rtol=2e-4)


def test_multi_positive_loss_rewards_the_harder_positive():
    positives = mx.array([[True, True, False]])
    valid = mx.ones((1, 3), dtype=mx.bool_)
    scores = mx.array([[.9, .1, .3]])
    gradient = mx.grad(lambda x: positive_cross_entropy(x, positives, valid, .2, 2))(scores)
    assert gradient[0, 1].item() < 0  # A gradient update raises the hard positive.
    better = mx.array([[.9, .5, .3]])
    assert positive_cross_entropy(better, positives, valid, .2, 2).item() < positive_cross_entropy(scores, positives, valid, .2, 2).item()


@pytest.mark.parametrize("mrl", [False, True])
def test_single_positive_mode_matches_existing_loss_and_gradients(mrl):
    from retrieval import known_positives
    mx.random.seed(91)
    batch = make_batch(ROWS)
    q = train.l2_normalize(mx.random.normal((len(ROWS), 4)))
    c = train.l2_normalize(mx.random.normal((len(batch.candidates), 4)))
    mask = train.build_false_negative_mask(
        [row["positive"] for row in ROWS],
        [text for row in ROWS for text in row["negatives"]],
        [known_positives(ROWS)[row["query"]] for row in ROWS])
    legacy = (lambda q, c: train.matryoshka_loss(q, c, [4, 2], .3, mask, 2)) if mrl else (
        lambda q, c: train.info_nce_loss(q, c, .3, mask, 2))
    objective = objective_for(batch, hardness=2, dims=[4, 2] if mrl else None)
    reference, ref_grads = mx.value_and_grad(legacy, argnums=(0, 1))(q, c)
    actual, grads = mx.value_and_grad(lambda q, c: objective(q, c, 0, len(ROWS)), argnums=(0, 1))(q, c)
    assert actual.item() == pytest.approx(reference.item(), abs=1e-6)
    for expected, value in zip(ref_grads, grads):
        np.testing.assert_allclose(np.array(value), np.array(expected), atol=1e-6)


def test_teacher_kl_improves_ranking_and_ignores_unscored_candidates():
    scores = mx.array([[.2, .9, 0.]])
    mask = mx.array([[True, True, False]])
    student = mx.array([[.6, .4, 1000.]])
    loss_fn = lambda x: ranking_kl(x, scores, mask, .2)
    gradient = mx.grad(loss_fn)(student)
    assert gradient[0, 2].item() == 0
    assert gradient[0, 0].item() > 0 and gradient[0, 1].item() < 0
    assert loss_fn(student - .05 * gradient).item() < loss_fn(student).item()
    assert abs(ranking_kl(scores, scores, mask, .2).item()) < 1e-6


def test_rng_roundtrip_after_random_stream_has_advanced():
    mx.random.seed(19)
    mx.eval(mx.random.normal(shape=(7,)))
    seed = capture_rng()
    expected = mx.random.uniform(shape=(13,))
    mx.eval(expected)
    restore_rng(seed)
    actual = mx.random.uniform(shape=(13,))
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_cache_restores_rng_when_loss_fails():
    mx.random.seed(113)
    model = DropoutEncoder(.2)
    model.train()
    mx.eval(model.parameters())
    q, c = token_chunks(["query"], 1), token_chunks(["positive", "negative"], 1)
    before = capture_rng()
    for tokens in [*q, *c]:
        mx.eval(encode(model, tokens))
    after_forward = capture_rng()
    restore_rng(before)

    def failure(*args):
        raise RuntimeError("intentional objective failure")

    with pytest.raises(RuntimeError, match="intentional"):
        cached_value_and_grad(model, encode, q, c, failure)
    assert capture_rng() == after_forward


def test_validation_respects_encoding_chunk_limit(monkeypatch):
    model, tokenizer = TinyEncoder(), TinyTokenizer()
    model.eval()
    reference = train.evaluate_loss(model, tokenizer, ROWS, 3, .3, 32, True, True,
                                    multi_positive=True)
    original = train.encode_texts
    sizes = []

    def record(m, input_ids, attention_mask, **kwargs):
        sizes.append(input_ids.shape[0])
        return original(m, input_ids, attention_mask, **kwargs)

    monkeypatch.setattr(train, "encode_texts", record)
    actual = train.evaluate_loss(model, tokenizer, ROWS, 3, .3, 32, True, True,
                                 multi_positive=True, encoding_chunk_size=1, loss_chunk_size=1)
    assert max(sizes) == 1
    assert actual == pytest.approx(reference, abs=2e-6)


def test_stateful_encoder_rejected_before_replay():
    model = TinyEncoder()
    model.batch_norm = nn.BatchNorm(4)
    with pytest.raises(ValueError, match="stateful"):
        cached_value_and_grad(model, encode, token_chunks(["q"], 1), token_chunks(["p"], 1), None)


@pytest.mark.parametrize("cache", [0, 1, 3])
def test_all_features_run_through_training_and_checkpoint_export(tmp_path, monkeypatch, cache):
    from retrieval import file_digest
    args, calls, exports = setup_run(tmp_path, monkeypatch, rows=5,
                                    extra=["--multi-positive", "--cache-chunk-size", str(cache),
                                           "--loss-chunk-size", "1", "--matryoshka-dims", "4,2",
                                           "--hardness-strength", "2", "--lora-dropout", ".1"])
    pairs = train.load_pairs(args.train_pairs)
    pairs[0]["positives"] = ["another relevant answer"]
    with open(args.train_pairs, "w") as f:
        for row in pairs:
            f.write(json.dumps(row) + "\n")
    manifest = {"type": "teacher_manifest", "schema_version": 1,
                "train_sha256": file_digest(args.train_pairs)}
    path = tmp_path / "teacher.jsonl"
    records = [manifest, *[{"query": q, "scores": [{"document": doc, "score": i / 10}
                       for i, doc in enumerate(docs)]} for q, docs in teacher_groups(pairs).items()]]
    path.write_text("\n".join(map(json.dumps, records)))
    args.teacher_scores, args.distill_weight = str(path), .2
    train.train(args)
    assert calls and exports[0]["multi_positive"]
    assert exports[0]["cache_chunk_size"] == cache
    assert exports[0]["distill_weight"] == .2
    assert exports[0]["teacher_scores_sha256"] == file_digest(path)


@pytest.mark.parametrize("flags", [["--cache-chunk-size", "-1"], ["--loss-chunk-size", "0"],
                                   ["--distill-weight", ".1"], ["--teacher-scores", "file"],
                                   ["--distill-temperature", "nan"], ["--distill-weight", "-1"]])
def test_invalid_feature_combinations_fail(flags):
    args = train.build_parser().parse_args(["--train-pairs", "unused", *flags])
    with pytest.raises(ValueError):
        train.validate_args(args)
