"""GPT-2 LTLM wrapper: per-layer thought cross-attention + performance planner.

Thoughts z have shape (batch, n_layers * thoughts_per_layer, n_embd). Layer i
cross-attends to thoughts [i*T:(i+1)*T]. The planner p_phi(z|P) is an amortized
diagonal Gaussian whose mean/log-var are produced by attending a set of learned
thought queries to the packed performance (control) tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from anticipation.ltlm_objective import planner_regularized_loss
from anticipation.vocab import CONTROL_OFFSET, SEPARATOR


def control_token_mask(input_ids: torch.Tensor) -> torch.Tensor:
    return (input_ids >= CONTROL_OFFSET) & (input_ids != SEPARATOR)


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

    def forward(self, hidden_states: torch.Tensor, thoughts: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = hidden_states.shape
        n_z = thoughts.shape[1]
        q = self.q(hidden_states).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(thoughts).view(bsz, n_z, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(thoughts).view(bsz, n_z, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.o(attn)


class ThoughtInjectedBlock(nn.Module):
    """Wrap a GPT-2 block and add residual cross-attention to that layer's thoughts."""

    def __init__(self, block: nn.Module, hidden: int, n_heads: int):
        super().__init__()
        self.block = block
        self.ln_cross = nn.LayerNorm(hidden)
        self.cross_attn = _CrossAttention(hidden, n_heads)
        self.thoughts: Optional[torch.Tensor] = None

    def forward(self, hidden_states, *args, **kwargs):
        outputs = self.block(hidden_states, *args, **kwargs)
        is_tuple = isinstance(outputs, tuple)
        h = outputs[0] if is_tuple else outputs
        if self.thoughts is not None:
            h = h + self.cross_attn(self.ln_cross(h), self.thoughts)
        if is_tuple:
            return (h,) + outputs[1:]
        return h


class PerformancePlanner(nn.Module):
    """Amortized diagonal Gaussian p_phi(z | P) over packed control tokens."""

    def __init__(self, hidden: int, n_thoughts: int, n_heads: int):
        super().__init__()
        self.n_thoughts = n_thoughts
        self.queries = nn.Parameter(torch.randn(n_thoughts, hidden) * 0.02)
        self.cross_attn = _CrossAttention(hidden, n_heads)
        self.ln = nn.LayerNorm(hidden)
        self.to_mu = nn.Linear(hidden, hidden)
        self.to_log_var = nn.Linear(hidden, hidden)

    def forward(
        self,
        control_embeds: torch.Tensor,
        control_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # control_embeds: (B, T, D), control_mask: (B, T) True on performance tokens
        bsz = control_embeds.shape[0]
        queries = self.queries.unsqueeze(0).expand(bsz, -1, -1)
        # Mask non-control positions by zeroing them; residual cross-attn then
        # only sees real performance tokens (plus a tiny epsilon on empty rows).
        masked = control_embeds * control_mask.unsqueeze(-1).to(control_embeds.dtype)
        empty = ~control_mask.any(dim=-1)
        if empty.any():
            masked = masked.clone()
            masked[empty, 0] = 1e-3
        thoughts = queries + self.cross_attn(self.ln(queries), masked)
        mu = self.to_mu(thoughts)
        log_var = self.to_log_var(thoughts).clamp(-8.0, 4.0)
        return mu, log_var


@dataclass
class LTLMOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    nll: Optional[torch.Tensor] = None
    planner_kl: Optional[torch.Tensor] = None
    stats: Optional[dict[str, torch.Tensor]] = None
    past_key_values: Any = None


class LTLMCausalLM(nn.Module):
    """Causal LM with per-layer latent thoughts and a performance planner prior."""

    def __init__(
        self,
        base_model: nn.Module,
        thoughts_per_layer: int = 4,
        beta: float = 5.0,
        isotropic_kl_weight: float = 0.0,
        elbo_reduction: str = "mean",
    ):
        super().__init__()
        self.base_model = base_model
        config = base_model.config
        self.n_embd = int(config.n_embd)
        self.n_layer = int(config.n_layer)
        self.n_head = int(config.n_head)
        self.thoughts_per_layer = thoughts_per_layer
        self.n_thoughts = self.n_layer * thoughts_per_layer
        self.beta = beta
        self.isotropic_kl_weight = isotropic_kl_weight
        self.elbo_reduction = elbo_reduction
        self.z_dim = self.n_embd

        transformer = self._transformer()
        blocks = transformer.h
        wrapped = nn.ModuleList(
            ThoughtInjectedBlock(block, self.n_embd, self.n_head) for block in blocks
        )
        transformer.h = wrapped
        self.blocks = wrapped
        self.planner = PerformancePlanner(self.n_embd, self.n_thoughts, self.n_head)

    def _transformer(self):
        if hasattr(self.base_model, "transformer"):
            return self.base_model.transformer
        if hasattr(self.base_model, "model"):
            return self.base_model.model
        raise AttributeError("Could not find GPT-2 transformer trunk on base model")

    def _wte(self):
        return self._transformer().wte

    def set_thoughts(self, z: Optional[torch.Tensor]) -> None:
        if z is None:
            for block in self.blocks:
                block.thoughts = None
            return
        z = z.view(z.shape[0], self.n_layer, self.thoughts_per_layer, self.n_embd)
        for i, block in enumerate(self.blocks):
            block.thoughts = z[:, i]

    def planner_params(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self._wte()(input_ids)
        mask = control_token_mask(input_ids)
        return self.planner(embeds, mask)

    def decode(self, input_ids: torch.Tensor, z: torch.Tensor, **kwargs):
        self.set_thoughts(z)
        try:
            return self.base_model(input_ids=input_ids, **kwargs)
        finally:
            if not kwargs.get("use_cache"):
                self.set_thoughts(None)

    def score_nll(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        z: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.decode(
            input_ids,
            z,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return nll, logits

    def elbo(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        mu_q: torch.Tensor,
        log_var_q: torch.Tensor,
        eps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        detach_planner: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        mu_p, log_var_p = self.planner_params(input_ids)
        if detach_planner:
            mu_p = mu_p.detach()
            log_var_p = log_var_p.detach()
        std = torch.exp(0.5 * log_var_q)
        z = mu_q + eps * std
        nll, logits = self.score_nll(input_ids, labels, z, attention_mask=attention_mask)
        loss, stats = planner_regularized_loss(
            nll,
            mu_q,
            log_var_q,
            mu_p,
            log_var_p,
            beta=self.beta,
            isotropic_kl_weight=self.isotropic_kl_weight,
            reduction=self.elbo_reduction,
        )
        stats["mu_p"] = mu_p
        stats["log_var_p"] = log_var_p
        stats["z"] = z
        return loss, stats, logits

    def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
        """HF-compatible forward. Uses currently set thoughts if present.

        Training should call ``elbo`` / the posterior optimizer rather than this.
        Cached AR decode sets thoughts first, then calls this with use_cache=True.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            **kwargs,
        )
        return LTLMOutput(
            loss=getattr(outputs, "loss", None),
            logits=outputs.logits,
            past_key_values=getattr(outputs, "past_key_values", None),
        )

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(**kwargs)
