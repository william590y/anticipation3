"""Token-level PPO pieces: a value head, per-token rewards, and GAE.

The MDP here is one packed window per episode:

  state  s_t  the model's own rollout prefix up to the t-th generated score token
  action a_t  that score token (414 of them: 138 slots x onset/duration/pitch)
  reward r_t  +1 if a_t equals the ground-truth token exactly, -1 otherwise

**The reward is indexed to the action that earned it.** `rewards[:, t]` scores
`positions[t]`, the token the policy just emitted -- not the next one, and not a
lump sum at the end. That is what lets the advantage separate "this token was
wrong" from "this token was fine but the ones after it drifted": with
r_t known per step, GAE's delta_t = r_t + gamma*V(s_{t+1}) - V(s_t) attributes the
immediate hit to t and the downstream consequences through V(s_{t+1}).

The control (performance) triplets that sit between score triplets are *not*
actions -- they are teacher-forced, part of the environment transition. They earn
no reward and take no gradient; V(s_{t+1}) is read at the state that predicts the
next score token, so whatever the environment inserted in between is already
priced into that value.

Contrast with the GRPO arm, which collapses the same information into one scalar
per 414-token rollout and asks the group's variance to sort out which tokens
deserved it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Critic on the policy's final hidden states.

    Deliberately fed *detached* features: the value loss then trains only this
    head and never reshapes the policy trunk. That keeps the DDP graph simple
    (two independently-reduced modules, one backward) and stops a
    randomly-initialised critic from perturbing a converged policy through a
    shared trunk in the first few hundred steps.
    """

    def __init__(self, hidden_size, inner=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Linear(inner, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden):
        return self.net(hidden).squeeze(-1)


def token_rewards(rolled, targets, positions, valid, *, correct=1.0, wrong=-1.0):
    """+1 for an exactly-correct token, -1 otherwise, at the position that produced it.

    Exact token equality, no tolerance on the 10 ms onset/duration bins -- the
    same criterion as `per_rollout_accuracy` and `val/ar_*_accuracy`.
    """
    hit = rolled[:, positions] == targets[:, positions]
    rewards = torch.where(hit, torch.full_like(hit, correct, dtype=torch.float32),
                          torch.full_like(hit, wrong, dtype=torch.float32))
    return rewards * valid


def compute_gae(rewards, values, valid, *, gamma=1.0, lam=0.95):
    """Generalised advantage estimation over the action sequence.

    `values[:, t]` is V(s_t), the value of the state *before* action t, so the
    bootstrap for step t is `values[:, t + 1]` and the final step bootstraps from
    0 (the window ends). Returns (advantages, returns).
    """
    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    running = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
    for t in range(steps - 1, -1, -1):
        next_value = values[:, t + 1] if t + 1 < steps else torch.zeros_like(running)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        running = delta + gamma * lam * running
        advantages[:, t] = running
    return advantages * valid, (advantages + values) * valid


def discounted_returns(rewards, valid, *, gamma=1.0):
    """Monte Carlo returns that do not bootstrap from the value head.

    These are intentionally separate from GAE's lambda-returns: comparing the
    critic with a target that contains its own predictions can make explained
    variance look strong even when it predicts rewards poorly.
    """
    returns = torch.zeros_like(rewards)
    running = torch.zeros(
        rewards.shape[0], device=rewards.device, dtype=rewards.dtype
    )
    for t in range(rewards.shape[1] - 1, -1, -1):
        mask_t = valid[:, t].to(rewards.dtype)
        running = (rewards[:, t] + gamma * running) * mask_t
        returns[:, t] = running
    return returns


def whiten(values, mask, eps=1e-8):
    """Zero-mean unit-variance over the masked entries (PPO advantage whitening)."""
    count = mask.sum()
    if count < 2:
        return values
    mean = (values * mask).sum() / count
    var = (((values - mean) ** 2) * mask).sum() / count
    return ((values - mean) / (var.sqrt() + eps)) * mask


def approximate_kl(logprob, old_logprob, mask):
    """Schulman's non-negative k3 KL estimator, summed over valid actions."""
    log_ratio = logprob - old_logprob
    return ((log_ratio.exp() - log_ratio - 1.0) * mask).sum()


def policy_loss(logprob, old_logprob, advantages, mask, clip_eps):
    """Clipped surrogate; returns (loss_sum, clipped_count, approx_kl_sum)."""
    ratio = torch.exp(logprob - old_logprob)
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    loss = torch.max(unclipped, clipped)
    with torch.no_grad():
        # `mask` arrives as a float (it multiplies losses elsewhere), so combine
        # by product rather than bitwise-and.
        was_clipped = ((clipped > unclipped).to(mask.dtype) * mask).sum()
        approx_kl = approximate_kl(logprob, old_logprob, mask)
    return (loss * mask).sum(), was_clipped, approx_kl


def value_loss(values, old_values, returns, mask, clip_eps):
    """Clipped value loss (sum over masked entries)."""
    clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
    loss = torch.max((values - returns) ** 2, (clipped - returns) ** 2)
    return 0.5 * (loss * mask).sum()


def explained_variance(values, returns, mask):
    """1 - Var(returns - values) / Var(returns); <= 0 means the critic is useless."""
    count = mask.sum()
    if count < 2:
        return float("nan")
    v = values[mask.bool()]
    r = returns[mask.bool()]
    var_r = r.var()
    if var_r == 0:
        return float("nan")
    return float((1.0 - (r - v).var() / var_r).item())
