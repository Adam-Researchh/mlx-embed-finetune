"""Shared single/multi-positive objective and optional ranking distillation."""

import mlx.core as mx


def normalize(x):
    x = x.astype(mx.float32)
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def positive_cross_entropy(similarities, positives, valid, temperature, hardness=0.0):
    """Uniform targets over all positives; each query receives equal weight.

    Unlike a log-sum over positive probabilities, this penalizes ignoring a
    difficult positive even when another positive already scores highly.
    """
    logits = similarities / temperature
    if hardness:
        logits = logits + hardness * mx.stop_gradient(similarities) * (valid & ~positives)
    logits = mx.where(valid, logits, -float("inf"))
    log_probs = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    positive_log_probs = mx.where(positives, log_probs, 0.0)
    return -mx.mean(mx.sum(positive_log_probs, axis=1) / mx.sum(positives, axis=1))


def _ranking_log_probs(scores, scored, temperature):
    scores = mx.where(scored, scores.astype(mx.float32), -float("inf"))
    # Center before temperature scaling: an arbitrary teacher-score offset
    # must neither overflow nor erase the log normalizer in float32.
    logits = (scores - mx.max(scores, axis=1, keepdims=True)) / temperature
    return logits - mx.logsumexp(logits, axis=1, keepdims=True)


def ranking_kl(similarities, scores, scored, temperature):
    """KL(teacher || student) on each query's fixed scored candidate group.

    Unscored in-batch candidates still participate in the contrastive loss,
    but do not receive invented teacher labels. No temperature-squared factor.
    """
    teacher_logp = mx.stop_gradient(_ranking_log_probs(scores, scored, temperature))
    student_logp = _ranking_log_probs(similarities, scored, temperature)
    teacher_probability = mx.exp(teacher_logp)
    # Use the 0 * log(0) = 0 limit for unscored entries and teacher
    # probabilities that underflow to zero with widely separated scores.
    active = scored & (teacher_probability > 0)
    delta = mx.where(active, teacher_logp, 0.0) - mx.where(active, student_logp, 0.0)
    return mx.mean(mx.sum(teacher_probability * delta, axis=1))


def embedding_objective(q, c, positives, valid, temperature=.05, hardness=0.0,
                        dims=None, teacher_scores=None, teacher_mask=None,
                        distill_weight=0.0, distill_temperature=.1):
    q, c = q.astype(mx.float32), c.astype(mx.float32)
    if dims:
        contrastive = sum(positive_cross_entropy(
            normalize(q[:, :dim]) @ normalize(c[:, :dim]).T,
            positives, valid, temperature, hardness,
        ) for dim in dims) / len(dims)
    else:
        contrastive = positive_cross_entropy(q @ c.T, positives, valid, temperature, hardness)
    if distill_weight:
        if teacher_scores is None or teacher_mask is None:
            raise ValueError("Distillation requires teacher scores and a scored-candidate mask")
        # Distill only full-dimensional rankings. Matryoshka prefixes retain
        # their own supervised contrastive objectives.
        contrastive = contrastive + distill_weight * ranking_kl(
            q @ c.T, teacher_scores, teacher_mask, distill_temperature,
        )
    return contrastive
