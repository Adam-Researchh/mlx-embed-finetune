# Adversarial audit — October 12, 2026 UTC

The review started from published main
`8f8aa7a0a2f1fb50d2465bc0fa03bff48ce71aa5`. A fresh independent reviewer inspected
the training-feature changes against `51756d7eb38fd9ecf90763cc330df1ed63728521`
while the primary agent reran validation and checked numerical invariants.
The local work took place on October 11 in New York; this record uses UTC.

## Fixed: unstable teacher-ranking probabilities

**Severity: P2. Introduced by the ranking-distillation feature.**

`ranking_kl` previously computed `logits - logsumexp(logits)` without first
centering the scores. For a large common teacher-score offset, float32 rounding
could erase the normalizing term and produce probabilities that did not sum to
one. Temperature scaling could also overflow finite accepted scores.

With student scores `[0.6, 0.4]`, temperature `0.1`, and equal teacher scores:

| Teacher scores | Previous KL | Previous student gradient |
|---|---:|---|
| `[0, 0]` | 0.433781 | `[3.80797, -3.80797]` |
| `[10000000, 10000000]` | 2.253856 | `[7.61594, -7.61594]` |
| `[3e38, 3e38]` | NaN | Invalid |

All three teachers describe the same uniform distribution. The fix excludes
unscored candidates from the maximum, centers scores before scaling, and uses
the limiting zero contribution for teacher probabilities that underflow to
zero. All three cases now match the first row within floating-point tolerance.
Widely separated finite scores also remain finite for ordinary student scores
and temperatures. Teacher labels are detached from differentiation.

Four new tests reproduce the old failures and pass after the fix. An additional
150 randomized cases compared loss and student gradients against an independent
NumPy float64 reference, varying masks, row counts, score offsets/spreads and
temperatures `0.03`, `0.1`, `1`, and `10`. Maximum absolute differences were
`5.83e-6` for loss and `3.98e-6` for gradients.

## Existing upstream limitation: quantized Gemma padding

**Confirmed before adapters, training, or gradient caching.**

A subsequent priority check found the same mechanism in
[upstream issue 72](https://github.com/Blaizzy/mlx-embeddings/issues/72), reported
on August 23, 2026, with a proposed fix in
[PR 73](https://github.com/Blaizzy/mlx-embeddings/pull/73). Our audit is an
independent reproduction and measurement, not the first discovery of this bug.

On `mlx-embeddings==0.1.0`, MLX `0.32.2`, and mlx-lm `0.31.3`, the untouched
`mlx-community/embeddinggemma-300m-4bit` model embeds a short text differently
when encoded alone versus in a padded batch with a longer text. In the probe,
the cosine between the two vectors for `A cat` was **0.851936**; maximum
coordinate difference across the batch was **0.0701353**.

The upstream `gemma3_text.Model.__call__` casts its additive mask to
`embed_tokens.weight.dtype`. Quantized embedding weights have `uint32` dtype;
casting `[0.0, -inf]` to it yields `[0, 0]`, removing padding exclusion. The
embedding scale calculation also uses that packed-weight dtype. Replacing
only the embedding layer with its dequantized equivalent in an isolated probe
raised the short-text cosine to **0.999998** and reduced maximum coordinate
difference to **0.000287**.

This review does not modify or vendor upstream Gemma internals. Treat this
checkpoint as experimental on the affected stack and use the verified MiniLM
path pending an upstream correction and retest. A cache chunk size of one is
not a complete repair for the model's dtype handling or serving path.

To reproduce the batch-composition check with installed dependencies and cached
model weights, run from the repository root:

```python
import mlx.core as mx
import train
from evaluate import load_encoder

model, tokenizer = load_encoder("mlx-community/embeddinggemma-300m-4bit")
texts = [
    "A cat",
    "A long, detailed paragraph describing research on retrieval training and "
    "using a collection of candidates for every document in an embedding model.",
]
kwarg = train.detect_input_kwarg(model)

def encode(items):
    tokens = train.tokenize_batch(tokenizer, items)
    return train.encode_texts(model, **tokens, input_kwarg=kwarg)

together = encode(texts)
separate = mx.concatenate([encode([text]) for text in texts])
print("Cosines:", mx.sum(together * separate, axis=-1).tolist())
print("Mask cast:", mx.array([0., -float("inf")]).astype(mx.uint32).tolist())
```

## Scope and remaining precision limits

The review covered candidate identity and deduplication, repeated-query positive
labels, teacher cache validation, masked objectives, replay randomness, partial
chunks, validation/checkpoint integration and export behavior. No additional
confirmed regression was found in those paths. This is bounded audit evidence,
not a proof that every input or upstream revision is bug-free.

Additional ModernBERT 4-bit and Gemma 4-bit replay probes preserved exact RNG
progression. Same-chunk cached versus ordinary backward losses differed by
approximately `0.001–0.002`; the worst per-parameter relative gradient-norm
differences were `1.7–2.7%` for ModernBERT and `4.4–5.9%` for Gemma across the
probe configurations. These measurements do not establish bitwise equality or
the tighter MiniLM parity tolerances for those models.

A controlled Gemma probe disabled dropout and compared identical chunks. The
ordinary autodiff forward already differed from the forward without autodiff,
before any replay. Promoting the model's floating tensors to float32 in that
isolated probe reduced global gradient relative L2 error from `2.01%` to
`1.84e-6`, with gradient cosine `1.0` and matching cached/reference loss. This
isolates the observed discrepancy to low-precision arithmetic rather than
cache replay semantics. The audit does not measure its effect on retrieval
quality or change training precision defaults. Gemma also has the distinct
confirmed padding defect above.

The final local suite passed **131 tests**, including real bf16/4-bit MiniLM
adapter, gradient and export checks. The starting revision passed all 127 tests
and its GitHub Linux/macOS checks, illustrating why adversarial cases were
needed in addition to the existing suite. See [validation](validation.md) for
the dependency stack and maintained commands.
