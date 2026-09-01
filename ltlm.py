#!/usr/bin/env python
"""Latent Thought Models (Kong et al., ICML 2025) adapted to packed ASAP windows.

The original LTM (https://github.com/mingluzhao/Latent-Thought-Language-Models,
arXiv:2502.01567) is a causal Transformer that conditions every token on a set
of continuous *latent thought* vectors ``z``. ``z`` is not emitted by the LM:
it is a local variational parameter inferred per window by a short Adam run
on the ELBO (classical variational Bayes / "fast learning"), after which the
decoder ("slow learning") is updated to maximise ``p(tokens | z)``.

This module reimplements that inference against HuggingFace GPT-2 rather than
vendoring their LLaMA-style stack. The mapping onto the packed alternating
format:

* ``z`` is injected by **per-layer cross-attention** (Q from tokens, K/V from
  that layer's thought vectors). Sequence length stays 1020, so packed
  position ids remain ``0..1019`` and we never hit the plan_lm.py gotcha of
  extra wpe rows looking like document boundaries. An explicit all-ones
  ``attention_mask`` is still passed, matching train.py / plan_lm.py.
* Thoughts are **not** discrete plan tokens. This is a different object from
  the two-stage VQ plan in ``plan_vq.py`` / ``plan_lm.py``.
* Posterior inference is conditioned on a *label mask*, which is how the three
  validation modes differ:

  - **oracle** — labels over the full window (score + performance), or over
    the score triplets alone under ``--posterior_score_only`` (see
    :func:`score_only_labels`). Either way this is the posterior that cheats
    with the tokens we want to predict, and it is the training target for the
    LM and for the diffusion model.
  - **prefix** — labels only on tokens ``[0, ALTERNATING_START)``, the 32
    lookahead (control, dummy-rest) pairs. The body (interleaved score and
    later performance notes) is ignored. This is ``q(z | early prefix)``:
    what a causal model can infer before any body score slot.
  - **diffusion** — no MCMC; ``z`` is sampled from the performance-conditioned
    diffusion model (see ``ltlm_diffusion.py``).

The paper's adopted inference is AdamVI (not Langevin). Langevin dynamics
are implemented as an optional ``--inference_method langevin`` ablation
matching their Eq. (4).
"""
from __future__ import annotations

import math
import types
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from anticipation.packed_sequence import ALTERNATING_START, is_score_slot_start
from anticipation.score_constraints import constrain_score_token_logits
from anticipation.vocab import VOCAB_SIZE

PACKED_LENGTH = 1020
PREFIX_LENGTH = ALTERNATING_START  # 192: 32 x (control, dummy-rest)
INFERENCE_METHODS = ("adamVI", "langevin")


@dataclass
class LTLMConfig:
    """Latent-thought hyperparameters, scaled from the paper repo's defaults.

    Paper ``config.py`` (GPT-2-small-shaped, 12 layers, dim 768)::

        num_steps = 16
        inference_method = 'adamVI'
        initial_fast_lr = 0.3
        max_z_len = n_layers * 8
        z_dim = dim

    music-medium-800k is GPT-2 medium (24 layers, hidden 1024), so 8 thoughts
    per layer would be 192 vectors of 1024-d. We default to 4 per layer (96
    total) to sit on the paper's larger ``Nz=96`` operating point without
    doubling the already-dominant fast-learning FLOPs. Override with
    ``--thoughts_per_layer``.
    """

    thoughts_per_layer: int = 4
    z_dim: int = 0  # 0 -> use the GPT-2 hidden size
    mcmc_steps: int = 16
    inference_method: str = "adamVI"
    fast_lr: float = 0.3
    final_fast_lr: float = 0.34
    kl_weight: float = 1.0
    elbo_reduction: str = "mean"  # "mean" or "sum"; see infer_posterior
    langevin_step_size: float = 0.1
    prefix_length: int = PREFIX_LENGTH
    dropout: float = 0.0  # cross-attn dropout; GPT-2 keeps its own

    def __post_init__(self):
        if self.thoughts_per_layer < 1:
            raise ValueError(f"thoughts_per_layer must be >= 1, got {self.thoughts_per_layer}")
        if self.inference_method not in INFERENCE_METHODS:
            raise ValueError(
                f"inference_method must be one of {INFERENCE_METHODS}, got {self.inference_method!r}"
            )
        if self.elbo_reduction not in ("mean", "sum"):
            raise ValueError(f"elbo_reduction must be 'mean' or 'sum', got {self.elbo_reduction!r}")

    def to_dict(self):
        return asdict(self)


def unwrap_model(model):
    """Peel DDP / DataParallel so we can touch ``transformer.h`` and config."""
    return model.module if hasattr(model, "module") else model


def n_layers_of(model):
    return len(unwrap_model(model).transformer.h)


def hidden_size_of(model):
    return int(unwrap_model(model).config.n_embd)


def resolve_z_dim(model, config: LTLMConfig):
    return int(config.z_dim) if config.z_dim and config.z_dim > 0 else hidden_size_of(model)


def z_length(model, config: LTLMConfig):
    return n_layers_of(model) * config.thoughts_per_layer


def reshape_z(z, n_layers, thoughts_per_layer):
    """``[B, L*Nz, D]`` -> ``[B, L, Nz, D]``. Accepts already-layered input."""
    if z.dim() == 4:
        if z.shape[1] != n_layers or z.shape[2] != thoughts_per_layer:
            raise ValueError(
                f"layered z has shape {tuple(z.shape)}, expected "
                f"(B, {n_layers}, {thoughts_per_layer}, D)"
            )
        return z
    if z.dim() != 3:
        raise ValueError(f"z must be (B, L*Nz, D) or (B, L, Nz, D), got {tuple(z.shape)}")
    batch, total, dim = z.shape
    expected = n_layers * thoughts_per_layer
    if total != expected:
        raise ValueError(f"z length {total} != n_layers*thoughts_per_layer={expected}")
    return z.view(batch, n_layers, thoughts_per_layer, dim)


def flatten_z(z):
    """``[B, L, Nz, D]`` -> ``[B, L*Nz, D]``. Leaves already-flat input alone."""
    if z.dim() == 3:
        return z
    if z.dim() != 4:
        raise ValueError(f"z must be (B, L, Nz, D) or (B, L*Nz, D), got {tuple(z.shape)}")
    batch, n_layers, n_thoughts, dim = z.shape
    return z.reshape(batch, n_layers * n_thoughts, dim)


# ---------------------------------------------------------------------------
# cross-attention that does not touch GPT-2's KV cache
# ---------------------------------------------------------------------------

class ThoughtCrossAttention(nn.Module):
    """Token queries attend to a short sequence of thought vectors.

    Output projection is zero-initialised so a freshly attached LTLM is
    numerically identical to the pretrained GPT-2 until the thoughts learn
    something; otherwise random cross-attn would immediately wreck the
    fine-tune.
    """

    def __init__(self, hidden_size, n_heads, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = float(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, hidden_states, thoughts):
        batch, seq_len, dim = hidden_states.shape
        n_thoughts = thoughts.shape[1]
        q = self.q(hidden_states).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(thoughts).view(batch, n_thoughts, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(thoughts).view(batch, n_thoughts, self.n_heads, self.head_dim).transpose(1, 2)
        dropout_p = self.attn_dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=False
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.resid_drop(self.out(attended))


def _thought_block_forward(
    self,
    hidden_states,
    past_key_values=None,
    cache_position=None,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    use_cache=False,
    output_attentions=False,
    **kwargs,
):
    """GPT2Block.forward with paper-style cross-attn between self-attn and MLP.

    ``encoder_hidden_states`` from the HF encoder-decoder path is ignored;
    thoughts arrive via ``self._current_z`` set by :func:`set_layer_thoughts`.
    Cross-attn is cache-free: K/V are a handful of thought vectors recomputed
    from ``z`` every step, which is cheaper than teaching GPT-2's
    ``EncoderDecoderCache`` about a new key.
    """
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_output, self_attn_weights = self.attn(
        hidden_states,
        past_key_values=past_key_values,
        cache_position=cache_position,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        **kwargs,
    )
    hidden_states = attn_output + residual

    thoughts = getattr(self, "_current_z", None)
    if thoughts is not None:
        residual = hidden_states
        hidden_states = residual + self.thought_cross_attn(self.thought_ln(hidden_states), thoughts)

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    hidden_states = residual + self.mlp(hidden_states)

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    return outputs


def attach_latent_thoughts(model, config: LTLMConfig, verbose=True):
    """Resize vocab and splice per-layer thought cross-attention onto GPT-2.

    New modules are registered as children of each ``GPT2Block`` so they show
    up in ``state_dict()`` / ``save_pretrained``. The output projection of
    each cross-attn is zero-init, so logits match the pretrained model until
    ``z`` actually moves.
    """
    if model.config.vocab_size != VOCAB_SIZE:
        if verbose:
            print(f"Resizing vocabulary {model.config.vocab_size} -> {VOCAB_SIZE}")
        model.resize_token_embeddings(VOCAB_SIZE)

    model = unwrap_model(model)
    hidden = hidden_size_of(model)
    n_heads = int(model.config.n_head)
    n_layers = n_layers_of(model)
    z_dim = resolve_z_dim(model, config)
    ln_eps = float(model.config.layer_norm_epsilon)

    if z_dim != hidden:
        model.ltlm_z_proj = nn.Linear(z_dim, hidden, bias=False)
        nn.init.normal_(model.ltlm_z_proj.weight, std=0.02)
    else:
        model.ltlm_z_proj = None

    # Shared thought-index embedding (paper uses RoPE on z; a learned table is
    # the GPT-2-native analogue and keeps us off their RoPE stack).
    model.ltlm_thought_pos = nn.Embedding(config.thoughts_per_layer, hidden)
    nn.init.normal_(model.ltlm_thought_pos.weight, std=0.02)

    for layer_idx, block in enumerate(model.transformer.h):
        if not hasattr(block, "thought_cross_attn"):
            block.thought_ln = nn.LayerNorm(hidden, eps=ln_eps)
            block.thought_cross_attn = ThoughtCrossAttention(
                hidden, n_heads, attn_dropout=config.dropout, resid_dropout=config.dropout
            )
        block._ltlm_layer_idx = layer_idx
        block._current_z = None
        block.forward = types.MethodType(_thought_block_forward, block)

    model.ltlm_config = config
    model.config.use_cache = False
    if verbose:
        print(
            f"Attached LTLM cross-attention: {n_layers} layers x "
            f"{config.thoughts_per_layer} thoughts, z_dim={z_dim}, "
            f"inference={config.inference_method}, T_fast={config.mcmc_steps}"
        )
    return model


def ltlm_module_names(model):
    """Parameter names that are *not* part of the pretrained GPT-2."""
    names = []
    for name, _ in model.named_parameters():
        if (
            name.startswith("ltlm_")
            or ".thought_cross_attn." in name
            or ".thought_ln." in name
            or name.startswith("thought_")
        ):
            names.append(name)
    return names


def pretrained_parameter_names(model):
    skip = set(ltlm_module_names(model))
    return [name for name, _ in model.named_parameters() if name not in skip]


def set_layer_thoughts(model, z):
    """Bind a (possibly flat) ``z`` onto each GPT-2 block for the next forward."""
    model = unwrap_model(model)
    config = model.ltlm_config
    n_layers = n_layers_of(model)
    layered = reshape_z(z, n_layers, config.thoughts_per_layer)
    if model.ltlm_z_proj is not None:
        layered = model.ltlm_z_proj(layered)
    pos = model.ltlm_thought_pos.weight.to(dtype=layered.dtype, device=layered.device)
    layered = layered + pos.unsqueeze(0).unsqueeze(0)
    for i, block in enumerate(model.transformer.h):
        block._current_z = layered[:, i]


def clear_layer_thoughts(model):
    for block in unwrap_model(model).transformer.h:
        block._current_z = None


def ones_attention_mask(input_ids):
    """Load-bearing all-ones mask; see plan_lm.py / CLAUDE.md."""
    return torch.ones_like(input_ids)


def ltlm_forward(
    model,
    input_ids,
    z,
    labels=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache=False,
    score_mask=None,
    inputs_embeds=None,
):
    """GPT-2 forward conditioned on latent thoughts ``z``.

    Packed windows keep default position ids ``0..T-1``. We still pass an
    explicit all-ones attention mask so transformers never inspects position
    ids for packed-sequence document boundaries.

    **Only when there is no KV cache.** transformers reads ``attention_mask``
    as a *padding* mask over the full key length (past + current), so an
    all-ones mask shaped like the one-token chunk of a cached decode silently
    truncates the cache to that chunk: every generated token is then produced
    from a 1-token context while the rollout still looks structurally valid.
    ``inference.batched_autoregressive_generate_score`` passes no mask at all
    during cached decode for exactly this reason, and we match it.
    """
    if attention_mask is None and past_key_values is None:
        attention_mask = ones_attention_mask(input_ids)
    inner = unwrap_model(model)
    if score_mask is not None and torch.any(score_mask):
        embed = inner.get_input_embeddings()
        inputs_embeds = embed(input_ids).masked_fill(score_mask.unsqueeze(-1), 0.0)
        input_ids = None
    set_layer_thoughts(inner, z)
    try:
        return model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
    finally:
        clear_layer_thoughts(model)


# ---------------------------------------------------------------------------
# prior / ELBO
# ---------------------------------------------------------------------------

def kl_standard_normal(mu, log_var):
    """KL(q(z|x) || N(0, I)) per element. ``q = N(mu, exp(log_var))``."""
    return -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())


def _count_labeled(labels):
    return (labels[:, 1:] != -100).sum().clamp(min=1)


def _elbo_terms(nll_mean, mu, log_var, labels, kl_weight, reduction):
    kl = kl_standard_normal(mu, log_var)
    if reduction == "sum":
        n_labeled = _count_labeled(labels).to(dtype=nll_mean.dtype)
        nll = nll_mean * n_labeled
        kl_term = kl.sum()
    else:
        nll = nll_mean
        kl_term = kl.mean()
    elbo_loss = nll + kl_weight * kl_term
    return elbo_loss, nll, kl_term, kl


def _freeze_decoder(model):
    frozen = []
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad_(False)
            frozen.append(param)
    return frozen


def _unfreeze(params):
    for param in params:
        param.requires_grad_(True)


def _fast_lr_at_step(step, num_steps, fast_lr):
    """Cosine decay from ``fast_lr`` to ``fast_lr/10`` over the inner loop.

    Matches ``PosteriorOptimizer.get_fast_lr`` in the LTM repo.
    """
    min_lr = fast_lr / 10.0
    if num_steps <= 1:
        return fast_lr
    decay_ratio = min(1.0, step / float(num_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (fast_lr - min_lr)


def score_only_labels(labels, score_token_mask):
    """Restrict the posterior's likelihood term to the real score triplets.

    The variational posterior should explain what the model actually
    *generates*. Control (performance) triplets are the given conditioning:
    they sit in the context either way, so scoring them only spends ``z``'s
    capacity on tokens nothing asks it to produce -- and in a packed window
    they are 170 of the 340 triplets, so they would otherwise dominate the
    ELBO's reconstruction term outright.

    ``score_token_mask`` comes straight off the dataset
    (``packed_dataset._build_score_token_mask``) and marks only *real* score
    triplets, so the prefix's 32 dummy rests are excluded too.
    """
    masked = labels.clone()
    masked[~score_token_mask] = -100
    return masked


def prefix_observation_labels(input_ids, prefix_length=PREFIX_LENGTH):
    """Labels for the prefix posterior, built from the tokens themselves.

    Deliberately **not** derived from the training labels. Under
    ``--loss_mask_performance_tokens`` those carry ``-100`` on every control,
    which would leave this arm observing only the prefix's 32 dummy rests --
    constant placeholders whose gradient w.r.t. ``z`` is ~0, so the posterior
    would never leave the prior and "prefix" would quietly become "no plan"
    while still reporting a full set of numbers.

    What this arm asks is what a plan inferred from the first 32 performance
    notes is worth, so those notes are the observation regardless of what the
    decoder is being trained to predict.
    """
    labels = input_ids.clone()
    labels[:, prefix_length:] = -100
    return labels


def prefix_labels(labels, prefix_length=PREFIX_LENGTH):
    """Keep labels only on the packed prefix ``[0, ALTERNATING_START)``.

    That span is 32 lookahead control triplets interleaved with 32 dummy-rest
    score placeholders. The dummy rests carry no score content; the 32
    controls are the performance notes a causal model has actually seen
    before the first body score slot. Body tokens (later performance *and*
    the score we want to predict) are ``-100``.
    """
    masked = labels.clone()
    masked[:, prefix_length:] = -100
    return masked


def _init_variational(batch, n_z, z_dim, device, dtype):
    """Start at the prior ``N(0, I)``.

    The LTM repo initialises ``log_var ~ N(-5, 0.1^2)`` (a very peaked q).
    That is tuned for training from scratch, where token-sum NLL is huge.
    Here the decoder is a pretrained music LM whose NLL is already ~0.1, so
    a peaked q would let the KL swamp reconstruction on step 1. Starting at
    the prior keeps KL = 0 until the data pull it away.
    """
    mu = torch.zeros(batch, n_z, z_dim, device=device, dtype=dtype)
    log_var = torch.zeros(batch, n_z, z_dim, device=device, dtype=dtype)
    return mu, log_var


@torch.no_grad()
def _posterior_metrics(mu, log_var, z, nll, kl_term, method):
    mu_norm = mu.detach().float().norm(dim=-1).mean()
    z_norm = z.detach().float().norm(dim=-1).mean()
    return {
        "nll": float(nll.detach().float().item()) if torch.is_tensor(nll) else float(nll),
        "kl": float(kl_term.detach().float().item()) if torch.is_tensor(kl_term) else float(kl_term),
        "z_norm": float(z_norm.item()),
        "mu_norm": float(mu_norm.item()),
        "log_var_mean": float(log_var.detach().float().mean().item()),
        "method": method,
    }


def infer_posterior(
    model,
    input_ids,
    labels,
    config: Optional[LTLMConfig] = None,
    attention_mask=None,
    score_mask=None,
    steps=None,
    fast_lr=None,
    method=None,
):
    """Fast learning: local variational parameters for this window.

    Decoder weights are frozen for the duration (gradients still flow *to*
    ``z`` through them). Returns ``(z, mu, log_var, stats)`` with ``z``
    detached, ready to condition the slow decoder update / diffusion target.

    ``labels`` already encode the observation set: pass the training labels
    (optionally through :func:`score_only_labels`) for the oracle posterior, or
    :func:`prefix_observation_labels` for the prefix posterior.

    Fast learning is *itself* an optimization, so it runs under an explicit
    ``torch.enable_grad()``: validation calls this from inside a
    ``@torch.no_grad()`` block, where the ELBO would otherwise have no graph
    and ``backward()`` would raise on the first eval step.
    """
    model = unwrap_model(model)
    config = config or model.ltlm_config
    method = method or config.inference_method
    steps = config.mcmc_steps if steps is None else steps
    fast_lr = config.fast_lr if fast_lr is None else fast_lr
    if attention_mask is None:
        attention_mask = ones_attention_mask(input_ids)

    n_z = z_length(model, config)
    z_dim = resolve_z_dim(model, config)
    batch = input_ids.shape[0]
    device = input_ids.device
    # Latents stay fp32 even under bf16 autocast: Adam on 0.3 lr is noisy
    # enough without also quantising the variational parameters.
    mu, log_var = _init_variational(batch, n_z, z_dim, device, torch.float32)
    eps = torch.randn_like(mu)

    was_training = model.training
    model.eval()
    frozen = _freeze_decoder(model)

    try:
        with torch.enable_grad():
            if method == "adamVI":
                z, mu, log_var, nll, kl_term = _adam_vi(
                    model,
                    input_ids,
                    labels,
                    attention_mask,
                    score_mask,
                    mu,
                    log_var,
                    eps,
                    steps,
                    fast_lr,
                    config,
                )
            elif method == "langevin":
                z, mu, log_var, nll, kl_term = _langevin(
                    model,
                    input_ids,
                    labels,
                    attention_mask,
                    score_mask,
                    mu,
                    steps,
                    config,
                )
            else:
                raise ValueError(f"unknown inference_method {method!r}")
    finally:
        _unfreeze(frozen)
        model.train(was_training)
        clear_layer_thoughts(model)

    stats = _posterior_metrics(mu, log_var, z, nll, kl_term, method)
    stats["steps"] = int(steps)
    stats["fast_lr"] = float(fast_lr)
    return z.detach(), mu.detach(), log_var.detach(), stats


def _adam_vi(
    model, input_ids, labels, attention_mask, score_mask,
    mu, log_var, eps, steps, fast_lr, config,
):
    mu.requires_grad_(True)
    log_var.requires_grad_(True)
    optimizer = torch.optim.AdamW([mu, log_var], lr=fast_lr, betas=(0.9, 0.999), eps=1e-8)
    nll = kl_term = None
    z = None
    for step in range(steps):
        lr_now = _fast_lr_at_step(step, steps, fast_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        optimizer.zero_grad(set_to_none=True)
        z = mu + eps * torch.exp(0.5 * log_var)
        outputs = ltlm_forward(
            model,
            input_ids,
            z.to(dtype=next(model.parameters()).dtype),
            labels=labels,
            attention_mask=attention_mask,
            score_mask=score_mask,
        )
        if outputs.loss is None:
            raise RuntimeError("ELBO forward produced no loss; labels are empty?")
        elbo_loss, nll, kl_term, _ = _elbo_terms(
            outputs.loss, mu, log_var, labels, config.kl_weight, config.elbo_reduction
        )
        elbo_loss.backward()
        optimizer.step()
    with torch.no_grad():
        z = mu + eps * torch.exp(0.5 * log_var)
        outputs = ltlm_forward(
            model,
            input_ids,
            z.to(dtype=next(model.parameters()).dtype),
            labels=labels,
            attention_mask=attention_mask,
            score_mask=score_mask,
        )
        _, nll, kl_term, _ = _elbo_terms(
            outputs.loss, mu, log_var, labels, config.kl_weight, config.elbo_reduction
        )
    return z.detach(), mu.detach(), log_var.detach(), nll.detach(), kl_term.detach()


def _langevin(
    model, input_ids, labels, attention_mask, score_mask,
    z, steps, config,
):
    """Unadjusted Langevin on ``log p(z|x) = log p(x|z) + log p(z)``.

    Paper Eq. (4): ``z <- z + s ∇_z log p(z|x) + sqrt(2s) ε``.
    ``∇_z log p(z) = -z`` for a standard normal prior. We report the final
    ``z`` as both the sample and a degenerate ``mu`` (log_var = 0).
    """
    step_size = float(config.langevin_step_size)
    noise_scale = math.sqrt(2.0 * step_size)
    nll = kl_term = None
    z = z.detach().clone()
    for _ in range(steps):
        z = z.detach().requires_grad_(True)
        outputs = ltlm_forward(
            model,
            input_ids,
            z.to(dtype=next(model.parameters()).dtype),
            labels=labels,
            attention_mask=attention_mask,
            score_mask=score_mask,
        )
        nll = outputs.loss
        # Prior score: -z. Using mean NLL so the step size is batch-stable.
        energy = nll + 0.5 * z.pow(2).mean()
        grad = torch.autograd.grad(energy, z)[0]
        with torch.no_grad():
            z = z - step_size * grad + noise_scale * torch.randn_like(z)
    with torch.no_grad():
        outputs = ltlm_forward(
            model,
            input_ids,
            z.to(dtype=next(model.parameters()).dtype),
            labels=labels,
            attention_mask=attention_mask,
            score_mask=score_mask,
        )
        nll = outputs.loss
        log_var = torch.zeros_like(z)
        _, nll, kl_term, _ = _elbo_terms(
            nll, z, log_var, labels, config.kl_weight, config.elbo_reduction
        )
    return z.detach(), z.detach(), log_var, nll.detach(), kl_term.detach()


# ---------------------------------------------------------------------------
# autoregressive score decode, conditioned on a fixed z
# ---------------------------------------------------------------------------

@torch.inference_mode()
def ltlm_autoregressive_generate_score(
    model,
    tokens_batch,
    z,
    device,
    constrain_score_tokens=True,
    ground_truth_score_tokens_to_feed=0,
    score_start_idx=ALTERNATING_START,
):
    """Greedy score decode with teacher-forced controls, conditioned on ``z``.

    Same slot schedule as ``inference.batched_autoregressive_generate_score``
    (and train.py's validation): body score triplets are emitted left to
    right, each followed by the ground-truth control triplet. ``z`` is held
    fixed for the whole rollout — it is the plan, not something the LM emits.
    """
    if ground_truth_score_tokens_to_feed < 0:
        raise ValueError("ground_truth_score_tokens_to_feed must be non-negative")

    tokens_batch = tokens_batch.to(device)
    z = z.to(device)
    length = tokens_batch.shape[1]
    context_chunks = [tokens_batch[:, :score_start_idx]]
    fed_score_tokens = 0

    def _forward(token_chunk, past=None):
        # No attention_mask once a cache exists -- see ltlm_forward's docstring.
        return ltlm_forward(
            model,
            token_chunk,
            z,
            past_key_values=past,
            use_cache=True,
        )

    primed = _forward(context_chunks[0], past=None)
    past = primed.past_key_values
    next_logits = primed.logits[:, -1, :]

    def append_and_feed(token_col):
        nonlocal past, next_logits
        context_chunks.append(token_col.unsqueeze(1))
        out = _forward(token_col.unsqueeze(1), past=past)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]

    pos = score_start_idx
    while pos + 5 < length:
        if is_score_slot_start(pos, score_start_idx):
            for slot in range(3):
                if fed_score_tokens < ground_truth_score_tokens_to_feed:
                    token_col = tokens_batch[:, pos + slot]
                    fed_score_tokens += 1
                else:
                    logits = next_logits
                    if constrain_score_tokens:
                        logits = constrain_score_token_logits(logits, slot)
                    token_col = logits.argmax(dim=-1)
                append_and_feed(token_col)
            pos += 3
            if pos + 2 < length:
                for k in range(3):
                    append_and_feed(tokens_batch[:, pos + k])
                pos += 3
        else:
            append_and_feed(tokens_batch[:, pos])
            pos += 1

    return torch.cat(context_chunks, dim=1)


def save_ltlm_sidecar(model, path):
    """Write the LTLM config next to a ``save_pretrained`` directory."""
    import json
    from pathlib import Path

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    config = model.ltlm_config
    payload = config.to_dict()
    payload["n_layers"] = n_layers_of(model)
    payload["hidden_size"] = hidden_size_of(model)
    payload["z_dim"] = resolve_z_dim(model, config)
    (path / "ltlm_config.json").write_text(json.dumps(payload, indent=2) + "\n")


def load_ltlm_config(path):
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text())
    known = {f.name for f in LTLMConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in payload.items() if k in known}
    return LTLMConfig(**filtered)
