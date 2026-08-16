"""Unit tests for the diffusion planner and beta * D(q, p_phi)."""

import torch

from anticipation.ltlm_diffusion import DiffusionPlanner, cosine_beta_schedule
from anticipation.ltlm_objective import planner_regularized_loss


def _tiny_planner(**kwargs):
    defaults = dict(hidden=32, n_thoughts=4, n_heads=4, timesteps=10, n_layers=2)
    defaults.update(kwargs)
    return DiffusionPlanner(**defaults)


def test_cosine_schedule_length_and_range():
    betas = cosine_beta_schedule(20)
    assert betas.shape == (20,)
    assert float(betas.min()) > 0
    assert float(betas.max()) < 1


def test_q_sample_matches_closed_form():
    planner = _tiny_planner()
    z0 = torch.randn(2, 4, 32)
    t = torch.tensor([0, 5], dtype=torch.long)
    noise = torch.randn_like(z0)
    zt = planner.q_sample(z0, t, noise)
    sqrt_ab = planner.sqrt_alphas_cumprod[t].view(2, 1, 1)
    sqrt_om = planner.sqrt_one_minus_alphas_cumprod[t].view(2, 1, 1)
    expected = sqrt_ab * z0 + sqrt_om * noise
    assert torch.allclose(zt, expected, atol=1e-5)


def test_denoise_loss_is_finite_and_scalar():
    planner = _tiny_planner()
    z0 = torch.randn(3, 4, 32)
    cond = torch.randn(3, 4, 32)
    loss = planner.denoise_loss(z0, cond)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_denoise_loss_backprops_to_z_with_frozen_planner():
    planner = _tiny_planner()
    for p in planner.parameters():
        p.requires_grad_(False)
    z0 = torch.randn(2, 4, 32, requires_grad=True)
    cond = torch.randn(2, 4, 32)
    loss = planner.denoise_loss(z0, cond)
    loss.backward()
    assert z0.grad is not None
    assert torch.isfinite(z0.grad).all()
    assert all(p.grad is None for p in planner.parameters())


def test_denoise_loss_trains_phi_when_z_detached():
    planner = _tiny_planner()
    z0 = torch.randn(2, 4, 32)
    cond = torch.randn(2, 4, 32, requires_grad=True)
    loss = planner.denoise_loss(z0, cond)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in planner.parameters())


def test_ddim_sample_shape_and_finite():
    planner = _tiny_planner()
    cond = torch.randn(2, 4, 32)
    z = planner.ddim_sample(cond, steps=4)
    assert z.shape == cond.shape
    assert torch.isfinite(z).all()


def test_beta_scales_diffusion_divergence():
    nll = torch.tensor(1.0)
    mu_q = torch.zeros(2, 4)
    log_var_q = torch.zeros_like(mu_q)
    div = torch.tensor(0.4)
    low, stats = planner_regularized_loss(
        nll, mu_q, log_var_q, beta=1.0, divergence=div
    )
    high, _ = planner_regularized_loss(
        nll, mu_q, log_var_q, beta=10.0, divergence=div
    )
    assert abs(float(low) - 1.4) < 1e-5
    assert abs(float(high) - 5.0) < 1e-5
    assert abs(float(stats["planner_kl"]) - 0.4) < 1e-5


def test_condition_masks_non_control_tokens():
    planner = _tiny_planner()
    embeds = torch.randn(2, 6, 32)
    mask = torch.tensor(
        [
            [True, True, False, False, True, False],
            [False, False, False, False, False, False],
        ]
    )
    cond = planner.condition(embeds, mask)
    assert cond.shape == (2, 4, 32)
    assert torch.isfinite(cond).all()
