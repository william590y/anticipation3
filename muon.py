"""Muon optimizer (Newton-Schulz orthogonalized momentum) with aux AdamW.

Single torch.optim.Optimizer so accelerate.prepare / LambdaLR treat it like
AdamW: param groups with use_muon=True get Muon updates (2D hidden matrices),
groups with use_muon=False get decoupled-WD Adam (embeddings, head, 1D params).
Weight decay is DECOUPLED in both branches (p *= 1 - lr*wd), which is the form
the "Pre-training under infinite compute" recipe tunes to ~30x standard.
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate UV^T of the SVD of G via quintic Newton-Schulz (bf16)."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class MuonWithAuxAdam(Optimizer):
    def __init__(self, param_groups):
        for g in param_groups:
            assert "use_muon" in g
            if g["use_muon"]:
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0.0)
                g.setdefault("ns_steps", 5)
            else:
                g.setdefault("betas", (0.9, 0.999))
                g.setdefault("eps", 1e-8)
                g.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            if g["use_muon"]:
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(p.grad)
                    buf = st["momentum_buffer"]
                    buf.lerp_(p.grad, 1 - g["momentum"])
                    upd = p.grad.lerp(buf, g["momentum"])       # nesterov
                    upd = zeropower_via_newtonschulz5(
                        upd.reshape(len(upd), -1), g["ns_steps"]
                    ).view_as(p)
                    scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.add_(upd, alpha=-g["lr"] * scale)
            else:
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "exp_avg" not in st:
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                        st["t"] = 0
                    st["t"] += 1
                    b1, b2 = g["betas"]
                    st["exp_avg"].lerp_(p.grad, 1 - b1)
                    st["exp_avg_sq"].mul_(b2).addcmul_(p.grad, p.grad,
                                                       value=1 - b2)
                    bc1 = 1 - b1 ** st["t"]
                    bc2 = 1 - b2 ** st["t"]
                    denom = (st["exp_avg_sq"] / bc2).sqrt_().add_(g["eps"])
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.addcdiv_(st["exp_avg"] / bc1, denom, value=-g["lr"])
        return loss


def build_muon_for_causal_lm(model, muon_lr, muon_wd, adam_lr, emb_wd):
    """Split a HF causal LM: 2D hidden matrices -> Muon; embeddings/head ->
    Adam with (strong) decay; norms and biases -> Adam without decay."""
    muon_p, emb_p, aux_p = [], [], []
    seen = set()
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if any(k in n for k in ("wte", "wpe", "lm_head", "embed")):
            emb_p.append(p)
        elif p.ndim >= 2:
            muon_p.append(p)
        else:
            aux_p.append(p)
    groups = [
        dict(params=muon_p, lr=muon_lr, weight_decay=muon_wd, use_muon=True),
        dict(params=emb_p, lr=adam_lr, weight_decay=emb_wd, use_muon=False),
        dict(params=aux_p, lr=adam_lr, weight_decay=0.0, use_muon=False),
    ]
    n = lambda ps: sum(p.numel() for p in ps)
    print(f"Muon split: {len(muon_p)} matrices ({n(muon_p)/1e6:.1f}M) muon "
          f"lr={muon_lr} wd={muon_wd} | {len(emb_p)} emb ({n(emb_p)/1e6:.1f}M) "
          f"adam wd={emb_wd} | {len(aux_p)} 1D ({n(aux_p)/1e6:.3f}M) adam wd=0")
    return MuonWithAuxAdam(groups)
