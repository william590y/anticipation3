#!/usr/bin/env python
"""Performance-conditioned diffusion over LTLM latent thoughts.

The transformer infers an *oracle* plan ``z`` by Bayesian posterior inference
on the full packed window (score + performance; ``--posterior_score_only``
restricts it to the score triplets instead, but is off by default). At true
inference time the score is unknown, so a second model has to propose ``z``
from the performance alone.

This module is a DDPM whose denoiser is a small Transformer decoder:
noisy thoughts are the queries, and they **cross-attend** to a
``PerformanceEncoder`` over the 170 control notes (the same encoder used by
``plan_vq.py``, at full 10 ms resolution). It is trained on the *same
batches* as the LM, targeting the oracle plan ``z`` the LM was conditioned on
(the posterior *sample*, not ``μ`` -- see ``train_ltlm._infer_z``).

This is the LTM paper's "VAE with amortized inference" slot, filled with a
diffusion model because the user asked for one and because the paper's VAE
collapsed. Cross-attention over performance is the analogue of their
inference network; the training target is the classical-VB local posterior
rather than a jointly-trained encoder, which is the distillation fix the
paper itself recommends.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from plan_vq import NUM_PERF_NOTES, PerformanceEncoder, split_packed_batch


def _broadcast_timestep(values, timesteps, ndim):
    """``values[t]`` -> shape that broadcasts with a (B, ...) tensor of ``ndim``."""
    out = values[timesteps]
    while out.dim() < ndim:
        out = out.unsqueeze(-1)
    return out


def cosine_alpha_bar(timesteps, s=0.008):
    """Nichol & Dhariwal cosine schedule, ``ᾱ_t`` in ``[ε, 1]``."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_bar = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    return alphas_bar.float()


@dataclass
class DiffusionConfig:
    """``d_model`` must not bottleneck ``z_dim``.

    ``z_out`` is a single ``d_model -> z_dim`` linear, so with ``d_model=256``
    and ``z_dim=1024`` the ε-prediction is confined to a 256-dimensional
    subspace of a 1024-dimensional isotropic target. DDIM then carries the
    unexplained three quarters of the noise straight into ``z0``, and the
    sampled plan is mostly noise however well the model trains. Default is
    ``z_dim``-wide; the extra cost is negligible next to 17 GPT-2-medium
    passes per step.
    """

    z_dim: int = 1024
    n_thoughts: int = 96
    d_model: int = 1024
    n_head: int = 8
    n_layer: int = 4
    perf_layers: int = 2
    fourier_bands: int = 32
    dropout: float = 0.0
    timesteps: int = 1000
    sample_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    target_scale_decay: float = 0.99
    clip_z0: float = 4.0

    def to_dict(self):
        return asdict(self)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"time embedding dim must be even, got {dim}")
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs)

    def forward(self, timesteps):
        angles = timesteps.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class PlanDiffusion(nn.Module):
    """ε-prediction DDPM over ``z ∈ R^{n_thoughts × z_dim}``, performance-conditioned."""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        if d % config.n_head != 0:
            raise ValueError(f"d_model {d} not divisible by n_head {config.n_head}")

        encoder_cfg = SimpleNamespace(
            d_model=d,
            fourier_bands=config.fourier_bands,
            n_head=config.n_head,
            perf_layers=config.perf_layers,
            dropout=config.dropout,
        )
        self.note_index = nn.Embedding(NUM_PERF_NOTES, d)
        self.performance = PerformanceEncoder(encoder_cfg)

        self.z_in = nn.Linear(config.z_dim, d)
        self.time_emb = SinusoidalTimeEmbedding(d)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(d, d))
        self.thought_pos = nn.Embedding(config.n_thoughts, d)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.n_head,
            dim_feedforward=4 * d,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.n_layer, norm=nn.LayerNorm(d)
        )
        self.z_out = nn.Linear(d, config.z_dim)

        # Running RMS of the oracle plan. The variational parameters
        # start at the prior (mu = 0) and grow only as far as 16 Adam steps
        # against an already-low NLL push them, so the target's scale is
        # unknown a priori and drifts as the decoder trains. A cosine schedule
        # assumes unit-variance data: at RMS 0.05 every timestep but the first
        # few is pure noise and the model learns eps ~= z_t, which samples back
        # to z ~= 0 -- a diffusion arm that trains cleanly and plans nothing.
        # Normalising by this buffer makes the schedule scale-free. It is a
        # buffer, so DDP's broadcast_buffers keeps every rank on rank 0's value.
        self.register_buffer("target_rms", torch.ones(()))
        self.register_buffer("target_rms_initialized", torch.zeros((), dtype=torch.bool))
        # EMA of the posterior's own sampling scale sigma = exp(0.5*log_var).
        # The denoiser is trained on mu, not on z = mu + eps*sigma: eps is
        # independent of both the window and the performance, so making the
        # denoiser predict it just buries the learnable signal. Measured at
        # kl_weight=10, mu_rms 0.310 against sigma 1.035 -- only 8% of the
        # sample target's variance was signal, and 1.4% at kl_weight=30.
        # sample() re-adds eps*posterior_sigma so the LM still sees the
        # distribution it was trained on.
        self.register_buffer("posterior_sigma", torch.ones(()))

        self._init_schedule(config)

    @torch.no_grad()
    def update_posterior_sigma(self, log_var):
        """EMA-track sigma from the oracle posterior's log_var."""
        sigma = torch.exp(0.5 * log_var.float()).mean()
        decay = float(self.config.target_scale_decay)
        self.posterior_sigma.mul_(decay).add_(sigma, alpha=1.0 - decay)
        return self.posterior_sigma

    @torch.no_grad()
    def _update_target_rms(self, z_target):
        rms = z_target.float().pow(2).mean().clamp(min=1e-12).sqrt()
        if not bool(self.target_rms_initialized):
            self.target_rms.copy_(rms)
            self.target_rms_initialized.fill_(True)
        else:
            decay = float(self.config.target_scale_decay)
            self.target_rms.mul_(decay).add_(rms, alpha=1.0 - decay)
        return self.target_rms.clamp(min=1e-6)

    def _init_schedule(self, config):
        alpha_bar = cosine_alpha_bar(config.timesteps)
        # ᾱ is length T+1 (includes t=0); drop the extra endpoint so indexing
        # by t ∈ {0..T-1} is well defined.
        alpha_bar_t = alpha_bar[1:]
        alpha_bar_prev = alpha_bar[:-1]
        betas = (1.0 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-8)).clamp(1e-5, 0.999)
        alphas = 1.0 - betas
        # Rebuild ᾱ from the *clipped* betas so the two agree. The raw cosine
        # schedule ends at exactly ᾱ_T = 0 and DDIM's z0 estimate divides by
        # sqrt(ᾱ), so the first sampling step divides by the 1e-8 clamp and the
        # "plan" comes back at ~1e7 (measured: diffusion z_rmse 1.9e7 against an
        # oracle whose z_rms is 1.1). Clipping still leaves ᾱ_T near 1e-9, so
        # sample() also clamps z0 -- both are needed.
        alpha_bar_t = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar_t)
        self.register_buffer("alpha_bar_prev", alpha_bar_t[:-1])
        # DDIM uses sqrt(ᾱ) and sqrt(1-ᾱ) directly.
        self.register_buffer("sqrt_alpha_bar", alpha_bar_t.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar_t).sqrt())

    def encode_performance(self, packed_or_perf):
        """Packed windows ``(B, 1020)`` or raw notes ``(B, 170, 3)`` -> ``(B, 170, d)``."""
        if packed_or_perf.dim() == 2:
            perf, _, _ = split_packed_batch(packed_or_perf)
        else:
            perf = packed_or_perf
        index = self.note_index.weight[: perf.shape[1]].unsqueeze(0)
        return self.performance(perf, index)

    def _denoise(self, z_t, timesteps, perf_state):
        """Predict ε given noisy thoughts, diffusion time, and performance."""
        batch, n_thoughts, _ = z_t.shape
        if n_thoughts != self.config.n_thoughts:
            raise ValueError(
                f"z has {n_thoughts} thoughts, diffusion was built for {self.config.n_thoughts}"
            )
        time_feat = self.time_proj(self.time_emb(timesteps)).unsqueeze(1)
        queries = (
            self.z_in(z_t)
            + self.thought_pos.weight.unsqueeze(0)
            + time_feat
        )
        hidden = self.decoder(queries, perf_state)
        return self.z_out(hidden)

    def q_sample(self, z0, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_ab = _broadcast_timestep(self.sqrt_alpha_bar, timesteps, z0.dim())
        sqrt_om = _broadcast_timestep(self.sqrt_one_minus_alpha_bar, timesteps, z0.dim())
        return sqrt_ab * z0 + sqrt_om * noise, noise

    def forward(self, z_target, packed_or_perf):
        """DDP entry point. Calling ``loss`` directly on the *unwrapped* module
        never runs ``prepare_for_backward``, so the reducer's autograd hooks stay
        disarmed and the diffusion gradients are never all-reduced -- each rank
        would quietly train its own copy. Go through the wrapper.

        Autocast is disabled here: bf16 is fine for a 355M LM trained on CE, but
        an ε-MSE at the low-loss end is exactly where bf16's 8-bit mantissa
        starts to matter, and this model is small enough that fp32 is free.
        """
        device_type = z_target.device.type
        if device_type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                return self.loss(z_target, packed_or_perf)
        return self.loss(z_target, packed_or_perf)

    def loss(self, z_target, packed_or_perf):
        """MSE on ε. ``z_target`` is the oracle plan, shape ``(B, n_z, D)``."""
        if z_target.dim() == 4:
            z_target = z_target.reshape(z_target.shape[0], -1, z_target.shape[-1])
        batch = z_target.shape[0]
        device = z_target.device
        z_target = z_target.float()
        if self.training:
            scale = self._update_target_rms(z_target)
        else:
            scale = self.target_rms.clamp(min=1e-6)
        z_target = z_target / scale
        timesteps = torch.randint(0, self.config.timesteps, (batch,), device=device)
        z_t, noise = self.q_sample(z_target, timesteps)
        perf_state = self.encode_performance(packed_or_perf)
        pred = self._denoise(z_t.to(dtype=perf_state.dtype), timesteps, perf_state)
        return F.mse_loss(pred.float(), noise.float())

    @torch.no_grad()
    def sample(self, packed_or_perf, steps=None, generator=None,
               add_posterior_noise=True):
        """DDIM (η=0) sample of the plan given performance only.

        The denoiser models ``mu``; ``add_posterior_noise`` re-adds
        ``eps*posterior_sigma`` on top so the returned object matches the
        ``z = mu + eps*sigma`` the decoder was trained on. Pass ``False`` to get
        the bare ``mu_hat`` for fidelity measurement.
        """
        if packed_or_perf.dim() == 2:
            batch = packed_or_perf.shape[0]
            device = packed_or_perf.device
        else:
            batch = packed_or_perf.shape[0]
            device = packed_or_perf.device
        steps = int(steps or self.config.sample_steps)
        steps = max(1, min(steps, self.config.timesteps))
        perf_state = self.encode_performance(packed_or_perf)
        dtype = perf_state.dtype

        z = torch.randn(
            batch,
            self.config.n_thoughts,
            self.config.z_dim,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        times = torch.linspace(
            self.config.timesteps - 1, 0, steps, device=device
        ).long()
        clip = float(self.config.clip_z0)
        for i, t in enumerate(times):
            t_batch = t.expand(batch)
            eps = self._denoise(z.to(dtype=dtype), t_batch, perf_state).float()
            alpha_bar_t = _broadcast_timestep(self.alpha_bar, t_batch, z.dim())
            z0 = (z - (1.0 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt().clamp(min=1e-8)
            # Targets are unit-RMS by construction (loss() divides by target_rms),
            # so anything past a few sigma is the 1/sqrt(ᾱ) amplification at the
            # noisy end, not signal. Without this the whole trajectory diverges.
            if clip > 0:
                z0 = z0.clamp(-clip, clip)
            if i + 1 == len(times):
                z = z0
                break
            t_prev = times[i + 1].expand(batch)
            alpha_bar_prev = _broadcast_timestep(self.alpha_bar, t_prev, z.dim())
            z = alpha_bar_prev.sqrt() * z0 + (1.0 - alpha_bar_prev).sqrt() * eps
        # Back to the oracle posterior's own scale (loss() trains on mu / rms).
        mu_hat = z * self.target_rms.clamp(min=1e-6)
        if not add_posterior_noise:
            return mu_hat
        # Re-add q's spread so the LM sees the same kind of object it was
        # trained on (mu + eps*sigma), rather than a noise-free mean.
        noise = torch.randn(
            mu_hat.shape, device=mu_hat.device, dtype=mu_hat.dtype, generator=generator
        )
        return mu_hat + noise * self.posterior_sigma


def save_diffusion(module, path):
    torch.save(
        {"config": module.config.to_dict(), "state_dict": module.state_dict()},
        path,
    )


def load_diffusion(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    config = DiffusionConfig(**payload["config"])
    module = PlanDiffusion(config)
    module.load_state_dict(payload["state_dict"])
    return module.to(device)
