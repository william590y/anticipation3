"""Planner-regularized LTLM objective.

The previous LTLM runs regularized the oracle posterior toward N(0, I) while
training a separate performance-conditional diffusion as an auxiliary. That
lets q(z|P,S) encode score information the planner cannot recover.

This module implements the intended ELBO term:

    L = E_q[-log p_theta(S | P, z)] + beta * D(q(z|P,S), p_phi(z|P))

with D = KL(q || p_phi) for diagonal Gaussians. beta trades oracle NLL
against forcing the posterior to stay in the planner's support.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


LOG_2PI = math.log(2.0 * math.pi)


def _flatten_latents(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1)


def gaussian_entropy(log_var: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Entropy of a diagonal Gaussian, H(q) = 0.5 * sum(log(2 pi e sigma^2))."""
    entropy = 0.5 * (LOG_2PI + 1.0 + log_var)
    per_example = entropy.reshape(log_var.shape[0], -1).sum(dim=-1)
    return _reduce(per_example, reduction, n_latents=log_var[0].numel())


def kl_diagonal_gaussians(
    mu_q: torch.Tensor,
    log_var_q: torch.Tensor,
    mu_p: torch.Tensor,
    log_var_p: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians.

    KL = 0.5 * sum( log(var_p/var_q) + (var_q + (mu_q-mu_p)^2)/var_p - 1 )
    """
    var_q = log_var_q.exp()
    var_p = log_var_p.exp()
    kl = 0.5 * (
        (log_var_p - log_var_q)
        + (var_q + (mu_q - mu_p).square()) / var_p.clamp_min(1e-8)
        - 1.0
    )
    per_example = kl.reshape(mu_q.shape[0], -1).sum(dim=-1)
    return _reduce(per_example, reduction, n_latents=mu_q[0].numel())


def kl_standard_normal(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL(q || N(0, I)). Kept for ablations / comparison to the old runs."""
    zeros = torch.zeros_like(mu)
    return kl_diagonal_gaussians(mu, log_var, zeros, zeros, reduction=reduction)


def _reduce(per_example: torch.Tensor, reduction: str, n_latents: int) -> torch.Tensor:
    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    if reduction == "mean":
        # Mean over batch and latent coordinates so beta is scale-invariant
        # in z-dimension (matches elbo_reduction=mean on the Cornell runs).
        return per_example.mean() / max(n_latents, 1)
    if reduction == "batchmean":
        return per_example.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def latent_alignment_stats(mu_q: torch.Tensor, mu_p: torch.Tensor) -> dict[str, torch.Tensor]:
    """Cosine / RMSE between oracle posterior mean and planner mean."""
    q = _flatten_latents(mu_q.float())
    p = _flatten_latents(mu_p.float())
    q_norm = F.normalize(q, dim=-1)
    p_norm = F.normalize(p, dim=-1)
    cosine = (q_norm * p_norm).sum(dim=-1).mean()
    rmse = (q - p).square().mean().sqrt()
    q_rms = q.square().mean().sqrt()
    p_rms = p.square().mean().sqrt()
    return {
        "z_cosine": cosine,
        "z_rmse": rmse,
        "q_rms": q_rms,
        "p_rms": p_rms,
    }


def planner_regularized_loss(
    nll: torch.Tensor,
    mu_q: torch.Tensor,
    log_var_q: torch.Tensor,
    mu_p: torch.Tensor,
    log_var_p: torch.Tensor,
    beta: float,
    *,
    isotropic_kl_weight: float = 0.0,
    reduction: str = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """L = NLL + beta * KL(q(z|P,S) || p(z|P)) [+ optional KL(q || N(0,I))].

    ``mu_p`` / ``log_var_p`` are the performance planner. Pass
    ``mu_p.detach()`` during the fast (AdamVI) inner loop so the planner is
    not updated 16x per batch; leave them attached on the slow step so phi
    fits the current posterior.
    """
    planner_kl = kl_diagonal_gaussians(
        mu_q, log_var_q, mu_p, log_var_p, reduction=reduction
    )
    loss = nll + beta * planner_kl
    stats = {
        "nll": nll.detach(),
        "planner_kl": planner_kl.detach(),
        "entropy": gaussian_entropy(log_var_q, reduction=reduction).detach(),
    }
    stats.update(latent_alignment_stats(mu_q.detach(), mu_p.detach()))
    if isotropic_kl_weight:
        iso = kl_standard_normal(mu_q, log_var_q, reduction=reduction)
        loss = loss + isotropic_kl_weight * iso
        stats["isotropic_kl"] = iso.detach()
    stats["loss"] = loss.detach()
    return loss, stats
