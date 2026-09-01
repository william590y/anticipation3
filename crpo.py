"""Contrastive Reinforced Policy Optimization (CRPO) pieces.

Wu et al., *Contrastive Reinforced Policy Optimization via Privileged
Self-Distillation* (arXiv:2607.28026). Notation below follows the paper:

  S_{i,t}   student context  -- rollout i's own prefix at position t
  T_{i,t}   teacher context  -- the same model given *privileged* information
  sim(i,t)  = -KL( pi(.|S_{i,t}) || sg( pi(.|T_{i,t}) ) )                  (Eq. 10)
  E         = sum_y p(y) log p(y)          -- NEGATIVE entropy, as written in
                                              Eqs. 5-7 (so low dE means the
                                              teacher is the more certain one)
  dE_{i,t}  = E^S_{i,t} - E^T_{i,t}                                         (Eq. 7)
  P         = bottom p% of dE over the group  (reflective exploration)      (Eq. 8)
  N         = everything else                (exposure bias)                (Eq. 9)
  L_CRPO    = -log[ sum_P exp(sim/tau) / sum_{P u N} exp(sim/tau) ]         (Eq. 11)
  L_CRPO*   = L_GRPO + lambda * L_CRPO                                      (Eq. 13)

KL and entropy are estimated on the **teacher's** top-K vocabulary with both
distributions renormalised over that truncated set (Appendix C.4, K=100).

**Adaptation to this repo.** The paper's privileged context is the prompt plus a
reference solution and environment feedback, pasted in front of the student's own
response. A packed score/performance window has no room for that -- the layout is
fixed and every position is spoken for. The faithful analogue here is the one this
whole codebase is built around: the teacher is the same model reading the
*ground-truth* prefix (teacher forcing) while the student reads its own rollout.
That is privileged information about the reference trajectory in exactly the
paper's sense, and it is the gap this repo measures directly -- teacher-forced
score-token CE 0.266 versus 10.762 along the model's own rollout.

One consequence of that mapping: T_{i,t} does not depend on the rollout index i
(all rollouts in a group share one ground-truth prefix), so the teacher pass runs
once per group rather than once per rollout. sim(i,t) and dE_{i,t} still vary with
i through the student side, so the group-wise contrast is unaffected.
"""

from __future__ import annotations

import copy

import torch


def teacher_topk_view(student_logits, teacher_logits, k):
    """Renormalise both distributions over the teacher's top-K support.

    `student_logits` is (G, S, V); `teacher_logits` is (1, S, V) when the group
    shares one ground-truth prefix, or (G, S, V) otherwise. Returns
    log-probabilities (G, S, K) for the student and (1 or G, S, K) for the
    teacher, both renormalised over the teacher's top-K support (Appendix C.4).
    The teacher side is left un-expanded -- every downstream op broadcasts, and
    these tensors are the largest ones in the step.
    """
    topk = teacher_logits.topk(k, dim=-1).indices
    teacher_log_probs = torch.log_softmax(teacher_logits.gather(-1, topk), dim=-1)

    student_topk = topk
    if topk.shape[0] == 1 and student_logits.shape[0] != 1:
        student_topk = topk.expand(student_logits.shape[0], -1, -1)
    student_log_probs = torch.log_softmax(student_logits.gather(-1, student_topk), dim=-1)
    return student_log_probs, teacher_log_probs


def negative_entropy(log_probs):
    """E = sum_y p log p, the sign convention of Eqs. 5-6 (i.e. -H)."""
    return (log_probs.exp() * log_probs).sum(dim=-1)


def similarity(student_log_probs, teacher_log_probs):
    """sim(i,t) = -KL( student || teacher ), Eq. 10."""
    student_probs = student_log_probs.exp()
    kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    return -kl


def split_positive_negative(delta_entropy, valid, positive_fraction):
    """P = bottom `positive_fraction` of dE across the whole group (Eqs. 8-9).

    Ranking is over the valid positions of every rollout in the group pooled
    together -- that pooling is what makes the contrast "group-wise".
    """
    flat_delta = delta_entropy[valid]
    if flat_delta.numel() == 0:
        return torch.zeros_like(valid)

    count = max(1, int(round(positive_fraction * flat_delta.numel())))
    count = min(count, flat_delta.numel() - 1) if flat_delta.numel() > 1 else 1
    threshold = torch.kthvalue(flat_delta.float(), count).values

    return valid & (delta_entropy <= threshold)


def contrastive_loss(sim, positive, valid, temperature):
    """Eq. 11, as one scalar over the group.

    Computed with log-sum-exp rather than the ratio of sums: sim is a negative
    KL, so exp(sim/tau) underflows to zero at exactly the drifted positions this
    objective exists to handle.
    """
    if not torch.any(positive) or not torch.any(valid):
        return sim.sum() * 0.0

    scaled = (sim / temperature).flatten()
    neg_inf = torch.finfo(scaled.dtype).min
    all_terms = torch.where(valid.flatten(), scaled, torch.full_like(scaled, neg_inf))
    positive_terms = torch.where(positive.flatten(), scaled, torch.full_like(scaled, neg_inf))
    return -(torch.logsumexp(positive_terms, dim=0) - torch.logsumexp(all_terms, dim=0))


def crpo_group_loss(
    student_logits,
    teacher_logits,
    positions,
    valid,
    *,
    top_k=100,
    temperature=1.0,
    positive_fraction=0.3,
):
    """L_CRPO for one group of rollouts, plus diagnostics.

    `student_logits` (G, L, V) carries gradient; `teacher_logits` (1 or G, L, V)
    must already be detached (the sg(.) in Eq. 10). Both are full-sequence logits;
    the distribution predicting position t is the one at t-1.
    """
    student_at = student_logits[:, positions - 1, :]
    teacher_at = teacher_logits[:, positions - 1, :]

    student_log_probs, teacher_log_probs = teacher_topk_view(student_at, teacher_at, top_k)

    student_entropy = negative_entropy(student_log_probs)
    teacher_entropy = negative_entropy(teacher_log_probs)
    delta_entropy = student_entropy - teacher_entropy

    sim = similarity(student_log_probs, teacher_log_probs)
    positive = split_positive_negative(delta_entropy.detach(), valid, positive_fraction)
    loss = contrastive_loss(sim, positive, valid, temperature)

    with torch.no_grad():
        # The teacher side is (1, S) when the group shares a ground-truth prefix.
        teacher_broadcast = teacher_entropy.expand_as(delta_entropy)
        stats = {
            "kl": float((-sim)[valid].mean().item()),
            "kl_positive": float((-sim)[positive].mean().item()) if torch.any(positive) else float("nan"),
            "kl_negative": (
                float((-sim)[valid & ~positive].mean().item())
                if torch.any(valid & ~positive) else float("nan")
            ),
            "delta_entropy": float(delta_entropy[valid].mean().item()),
            "student_entropy": float(-student_entropy[valid].mean().item()),
            "teacher_entropy": float(-teacher_broadcast[valid].mean().item()),
            "positive_frac": float(positive.sum().item()) / max(1.0, float(valid.sum().item())),
        }
    return loss, stats


class EMATeacher:
    """Trust-region self-teacher of Appendix C.3: theta_ref <- (1-a) theta_ref + a theta.

    Proposition 3's constrained teacher is a geometric mixture of the reference and
    the current policy; the paper implements it as an EMA over weights (alpha = 0.1).
    Holds its own frozen copy of the network, so the policy is never mutated.

    With alpha = 0 there is no copy at all and `module` is the policy itself -- the
    teacher is then the current model under stop-gradient, i.e. plain Eq. 11.
    """

    def __init__(self, module, alpha):
        """`module` is a separate frozen network for alpha > 0, else the policy itself.

        For alpha > 0 build it with `from_pretrained(init_checkpoint)` rather than by
        deep-copying the policy: accelerate rebinds `forward` on the prepared module,
        and deep-copying that bound method drags the whole module graph with it.
        Loading from the init checkpoint is also the right starting point -- theta_ref
        begins at the initial policy.
        """
        self.alpha = alpha
        self.module = module
        self.owns_copy = alpha > 0

    @torch.no_grad()
    def update(self, policy):
        if not self.owns_copy:
            return
        for reference, live in zip(self.module.parameters(), policy.parameters()):
            reference.mul_(1.0 - self.alpha).add_(live.detach(), alpha=self.alpha)
