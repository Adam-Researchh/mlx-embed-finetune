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


def ranking_kl(similarities, scores, scored, temperature):
    """KL(teacher || student) on each query's fixed scored candidate group.

    Unscored in-batch candidates still participate in the contrastive loss,
    but do not receive invented teacher labels. No temperature-squared factor.
    """
    teacher_logits = mx.where(scored, scores / temperature, -float("inf"))
    student_logits = mx.where(scored, similarities / temperature, -float("inf"))
    teacher_logp = teacher_logits - mx.logsumexp(teacher_logits, axis=1, keepdims=True)
    student_logp = student_logits - mx.logsumexp(student_logits, axis=1, keepdims=True)
    # Avoid 0 * (-inf - -inf) in sparse candidate groups.
    delta = mx.where(scored, teacher_logp, 0.0) - mx.where(scored, student_logp, 0.0)
    return mx.mean(mx.sum(mx.stop_gradient(mx.exp(teacher_logp)) * delta, axis=1))


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
