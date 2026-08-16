"""Unit tests for planner-regularized KL(q || p_phi)."""

import math

import torch

from anticipation.ltlm_objective import (
    gaussian_entropy,
    kl_diagonal_gaussians,
    kl_standard_normal,
    planner_regularized_loss,
)


def test_kl_identical_gaussians_is_zero():
    mu = torch.randn(4, 3, 8)
    log_var = torch.zeros_like(mu)
    kl = kl_diagonal_gaussians(mu, log_var, mu, log_var, reduction="mean")
    assert float(kl) < 1e-6


def test_kl_standard_normal_shift():
    # KL(N(1, I) || N(0, I)) = 0.5 per coordinate, then mean over coords.
    mu = torch.ones(2, 5)
    log_var = torch.zeros_like(mu)
    kl = kl_standard_normal(mu, log_var, reduction="mean")
    assert abs(float(kl) - 0.5) < 1e-5


def test_kl_batchmean_matches_sum_formula():
    mu_q = torch.tensor([[1.0, 0.0]])
    log_var_q = torch.zeros(1, 2)
    mu_p = torch.zeros(1, 2)
    log_var_p = torch.zeros(1, 2)
    # Per-example sum = 0.5 * (1^2 + 0) = 0.5
    kl = kl_diagonal_gaussians(mu_q, log_var_q, mu_p, log_var_p, reduction="batchmean")
    assert abs(float(kl) - 0.5) < 1e-5


def test_planner_loss_beta_zero_is_nll():
    nll = torch.tensor(1.25)
    mu = torch.randn(3, 4, 2)
    log_var = torch.zeros_like(mu)
    mu_p = torch.randn(3, 4, 2)
    log_var_p = torch.zeros_like(mu_p)
    loss, stats = planner_regularized_loss(
        nll, mu, log_var, mu_p, log_var_p, beta=0.0
    )
    assert abs(float(loss) - 1.25) < 1e-6
    assert "planner_kl" in stats


def test_planner_loss_increases_with_beta_when_q_leaves_p():
    nll = torch.tensor(1.0)
    mu_q = torch.ones(2, 8)
    log_var_q = torch.zeros_like(mu_q)
    mu_p = torch.zeros_like(mu_q)
    log_var_p = torch.zeros_like(mu_q)
    low, _ = planner_regularized_loss(nll, mu_q, log_var_q, mu_p, log_var_p, beta=1.0)
    high, _ = planner_regularized_loss(nll, mu_q, log_var_q, mu_p, log_var_p, beta=10.0)
    assert float(high) > float(low)


def test_entropy_of_unit_gaussian():
    log_var = torch.zeros(1, 1)
    # H = 0.5 * (log(2 pi e)) for 1-D unit Gaussian, then mean/n_latents = same.
    expected = 0.5 * (math.log(2.0 * math.pi) + 1.0)
    got = gaussian_entropy(log_var, reduction="mean")
    assert abs(float(got) - expected) < 1e-5
