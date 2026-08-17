"""Performance-conditional diffusion planner p_phi(z | P).

The training loss is still

    L = E_q[-log p_theta(S | P, z)] + beta * D(q(z|P,S), p_phi(z|P))

Analytic KL is intractable for a diffusion marginal, so D is the DDPM
noise-prediction bound on -log p_phi(z | P):

    D = E_{z~q, t, eps} [ ||eps - eps_phi(z_t, t, P)||^2 ]

That single term both trains phi to cover q (slow step) and, with phi
frozen, pulls q toward the reverse process (fast AdamVI step). This is
the same beta tradeoff as the Gaussian planner; only the parameterization
of p_phi and the tractable D change.

The previous LTLM jobs used KL(q || N(0,I)) plus an *auxiliary* diffusion
loss that only updated phi. That is not this D: q was never scored under
p_phi, so it could encode the score.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule, length `timesteps`."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


def _extract(schedule: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = schedule.gather(0, t)
    return out.reshape(-1, *([1] * (len(shape) - 1)))


class _CrossAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError(f"hidden {hidden} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)

    def forward(self, hidden_states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = hidden_states.shape
        n_ctx = context.shape[1]
        q = self.q(hidden_states).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(bsz, n_ctx, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(bsz, n_ctx, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.o(attn)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb.to(dtype=self.mlp[0].weight.dtype))


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.ln_attn = nn.LayerNorm(hidden)
        self.attn = _CrossAttention(hidden, n_heads)
        self.ln_ff = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_attn(x)
        x = x + self.attn(h, h)
        x = x + self.ff(self.ln_ff(x))
        return x


class PerformanceConditioner(nn.Module):
    """Learned thought queries cross-attend to packed control-token embeddings."""

    def __init__(self, hidden: int, n_thoughts: int, n_heads: int):
        super().__init__()
        self.n_thoughts = n_thoughts
        self.queries = nn.Parameter(torch.randn(n_thoughts, hidden) * 0.02)
        self.cross_attn = _CrossAttention(hidden, n_heads)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, control_embeds: torch.Tensor, control_mask: torch.Tensor) -> torch.Tensor:
        bsz = control_embeds.shape[0]
        queries = self.queries.unsqueeze(0).expand(bsz, -1, -1)
        masked = control_embeds * control_mask.unsqueeze(-1).to(control_embeds.dtype)
        empty = ~control_mask.any(dim=-1)
        if empty.any():
            masked = masked.clone()
            masked[empty, 0] = 1e-3
        return queries + self.cross_attn(self.ln(queries), masked)


class DiffusionPlanner(nn.Module):
    """Conditional DDPM over thought latents z, p_phi(z | P)."""

    def __init__(
        self,
        hidden: int,
        n_thoughts: int,
        n_heads: int,
        *,
        timesteps: int = 1000,
        n_layers: int = 3,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_thoughts = n_thoughts
        self.timesteps = int(timesteps)
        self.conditioner = PerformanceConditioner(hidden, n_thoughts, n_heads)
        self.time_embed = SinusoidalTimeEmbedding(hidden)
        self.input_proj = nn.Linear(hidden, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden, n_heads) for _ in range(n_layers))
        self.ln_out = nn.LayerNorm(hidden)
        self.to_eps = nn.Linear(hidden, hidden)

        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def condition(self, control_embeds: torch.Tensor, control_mask: torch.Tensor) -> torch.Tensor:
        return self.conditioner(control_embeds, control_mask)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = _extract(self.sqrt_alphas_cumprod.to(dtype=z0.dtype), t, z0.shape)
        sqrt_om = _extract(self.sqrt_one_minus_alphas_cumprod.to(dtype=z0.dtype), t, z0.shape)
        return sqrt_ab * z0 + sqrt_om * noise

    def predict_eps(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        temb = self.time_embed(t).unsqueeze(1).to(dtype=z_t.dtype)
        h = self.input_proj(z_t) + cond + temb
        for block in self.blocks:
            h = block(h)
        return self.to_eps(self.ln_out(h))

    def denoise_loss(
        self,
        z0: torch.Tensor,
        cond: torch.Tensor,
        *,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean-reduced DDPM noise-prediction loss. Differentiable in z0 and phi.

        Autocast is off: an ε-MSE at the low-loss end is where bf16's mantissa
        starts to matter, and this module is small enough that fp32 is free.
        """
        if z0.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                return self._denoise_loss_fp32(z0.float(), cond.float(), t=t, noise=noise)
        return self._denoise_loss_fp32(z0, cond, t=t, noise=noise)

    def _denoise_loss_fp32(
        self,
        z0: torch.Tensor,
        cond: torch.Tensor,
        *,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = z0.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (batch,), device=z0.device)
        if noise is None:
            noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        pred = self.predict_eps(z_t, t, cond)
        return F.mse_loss(pred, noise)

    def predict_x0(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        eps = self.predict_eps(z_t, t, cond)
        sqrt_ab = _extract(self.sqrt_alphas_cumprod.to(dtype=z_t.dtype), t, z_t.shape)
        sqrt_om = _extract(self.sqrt_one_minus_alphas_cumprod.to(dtype=z_t.dtype), t, z_t.shape)
        return (z_t - sqrt_om * eps) / sqrt_ab.clamp_min(1e-8)

    @torch.no_grad()
    def ddim_sample(
        self,
        cond: torch.Tensor,
        *,
        steps: int = 50,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Deterministic (eta=0) DDIM sample z ~ p_phi(z | P)."""
        steps = max(1, min(int(steps), self.timesteps))
        device = cond.device
        dtype = cond.dtype
        times = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        z = torch.randn(
            cond.shape[0],
            self.n_thoughts,
            self.hidden,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        for i, t_val in enumerate(times.tolist()):
            t = torch.full((cond.shape[0],), t_val, device=device, dtype=torch.long)
            eps = self.predict_eps(z, t, cond)
            alpha = _extract(self.alphas_cumprod.to(dtype=dtype), t, z.shape)
            x0 = (z - torch.sqrt((1.0 - alpha).clamp_min(0.0)) * eps) / torch.sqrt(alpha.clamp_min(1e-8))
            x0 = x0.clamp(-10.0, 10.0)
            if i + 1 == len(times):
                z = x0
                break
            t_prev = torch.full(
                (cond.shape[0],), int(times[i + 1].item()), device=device, dtype=torch.long
            )
            alpha_prev = _extract(self.alphas_cumprod.to(dtype=dtype), t_prev, z.shape)
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_prev) / (1.0 - alpha).clamp_min(1e-8))
                * (1.0 - alpha / alpha_prev.clamp_min(1e-8))
            )
            dir_xt = torch.sqrt((1.0 - alpha_prev - sigma.square()).clamp_min(0.0)) * eps
            noise = torch.randn(z.shape, device=device, dtype=dtype, generator=generator)
            z = torch.sqrt(alpha_prev) * x0 + dir_xt + sigma * noise
        return z
