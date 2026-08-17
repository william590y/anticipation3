"""AdamVI posterior inference for LTLM, targeting the performance planner.

Fast loop optimizes q(z|P,S) = N(mu, sigma) with the planner-regularized ELBO
while holding p_phi(z|P) fixed. The slow training step then updates theta and
phi with z and q detached.
"""

from __future__ import annotations

from typing import Optional

import torch

from anticipation.ltlm_model import LTLMCausalLM


class PosteriorOptimizer:
    def __init__(
        self,
        model: LTLMCausalLM,
        num_steps: int = 16,
        lr: float = 0.3,
        final_lr: Optional[float] = None,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.model = model
        self.num_steps = num_steps
        self.lr = lr
        self.final_lr = final_lr if final_lr is not None else lr
        self.betas = betas
        self.eps = eps

    @torch.no_grad()
    def _init_q(self, input_ids: torch.Tensor):
        # Start q at the current planner (Gaussian mean, or a short DDIM draw)
        # so D(q, p_phi) begins small.
        return self.model.init_posterior(input_ids)

    def infer(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        """Run AdamVI. Returns detached (z, mu_q, log_var_q) plus last stats."""
        steps = self.num_steps if num_steps is None else num_steps
        base_lr = self.lr if lr is None else lr
        was_training = self.model.training
        self.model.eval()
        # Decoder / planner are a frozen likelihood during fast inference so
        # inner-loop backwards do not leak into the slow parameter update.
        requires_grad_flags = [p.requires_grad for p in self.model.parameters()]
        for param in self.model.parameters():
            param.requires_grad_(False)

        try:
            with torch.enable_grad():
                mu, log_var = self._init_q(input_ids)
                mu.requires_grad_(True)
                log_var.requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    [mu, log_var], lr=base_lr, betas=self.betas, eps=self.eps
                )
                eps_noise = torch.randn_like(mu)

                last_stats: dict[str, torch.Tensor] = {}
                for step in range(steps):
                    for group in optimizer.param_groups:
                        t = step / max(steps - 1, 1)
                        group["lr"] = base_lr * (1.0 - t) + self.final_lr * t
                    optimizer.zero_grad(set_to_none=True)
                    # Call the module so torch.compile(DDP) actually runs.
                    # model.elbo(...) bypasses OptimizedModule.forward.
                    loss, stats, _ = self.model(
                        input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        mu_q=mu,
                        log_var_q=log_var,
                        eps=eps_noise,
                        detach_planner=True,
                    )
                    loss.backward()
                    optimizer.step()
                    last_stats = stats

                with torch.no_grad():
                    log_var.clamp_(-8.0, 4.0)
                    z = mu + eps_noise * torch.exp(0.5 * log_var)
        finally:
            for param, flag in zip(self.model.parameters(), requires_grad_flags):
                param.requires_grad_(flag)
            if was_training:
                self.model.train()
            self.model.zero_grad(set_to_none=True)

        return {
            "z": z.detach(),
            "mu_q": mu.detach(),
            "log_var_q": log_var.detach(),
            "eps": eps_noise.detach(),
            "stats": {k: v.detach() if torch.is_tensor(v) else v for k, v in last_stats.items()
                      if k not in ("mu_p", "log_var_p", "z")},
        }

    def infer_paired(
        self,
        input_ids: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Run AdamVI on two observation sets as one doubled batch.

        Oracle and prefix posteriors do not depend on each other, so stacking
        them is free parallelism on one GPU (same kernels, twice the batch).
        """
        bsz = input_ids.shape[0]
        ids = torch.cat([input_ids, input_ids], dim=0)
        labels = torch.cat([labels_a, labels_b], dim=0)
        mask = None if attention_mask is None else torch.cat([attention_mask, attention_mask], dim=0)
        out = self.infer(ids, labels, attention_mask=mask, **kwargs)

        def _split(key: str):
            tensor = out[key]
            return tensor[:bsz], tensor[bsz:]

        left, right = {}, {}
        for key in ("z", "mu_q", "log_var_q", "eps"):
            left[key], right[key] = _split(key)
        return left, right
