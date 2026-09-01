"""Mean-field variational (Bayes-by-Backprop) LoRA.

Turns a PEFT LoRA adapter into an approximate-Bayesian one: instead of a point
estimate of the low-rank update, we keep a fully factorized Gaussian posterior
q(B) = N(mu, sigma^2) over the entries of every ``lora_B`` matrix and optimize
the ELBO

    L = E_{q}[ CE(data) ]  +  KL( q(B) || N(0, prior_std^2) ) / M

with M the number of minibatches per epoch (the standard minibatch weighting of
Blundell et al., "Weight Uncertainty in Neural Networks", ICML 2015). The
expectation is estimated with a single reparameterized sample per forward pass,
    B = mu + softplus(rho) * eps,   eps ~ N(0, I),
which keeps everything differentiable w.r.t. both mu and rho.

WHY ONLY B (and not A):
    The LoRA update is Delta_W = (alpha/r) * B @ A, which is invariant to the
    rescaling (A, B) -> (cA, B/c) and to any invertible mixing of the rank
    directions. Putting a posterior over *both* factors therefore gives a
    non-identifiable model: infinitely many (q_A, q_B) pairs describe the same
    distribution over Delta_W, and the KL term can be driven around by the
    degenerate direction rather than by the data. Parameterizing only B keeps
    the map rho -> distribution over Delta_W injective given A. B is also the
    factor that LoRA zero-initializes, so a small init_sigma makes training
    start from (and stay near) the deterministic LoRA solution, which is exactly
    the behaviour we want for a fine-tuning-scale posterior. A stays a point
    estimate and is trained normally.

Implementation notes:
  * ``mu`` IS the existing ``lora_B.weight`` parameter -- we never replace it.
    That means PEFT checkpointing, ``merge_and_unload()`` and every other code
    path that reads ``lora_B.weight`` transparently sees the posterior MEAN,
    i.e. the deterministic LoRA model. The extra ``rho`` parameter lives on the
    same module and is saved into the adapter checkpoint too (its key contains
    "lora_" so PEFT's state-dict filter keeps it).
  * Sampling happens iff ``module.training`` is True, so ``model.eval()`` gives
    the deterministic posterior-mean model -- validation, autoregressive decode
    and merging are all noise-free by construction.
  * Under DDP each rank draws its own eps; gradients are averaged, which is a
    lower-variance MC estimate of the same ELBO gradient. Do NOT use gradient
    checkpointing with this: a recomputed forward would draw a *different* eps
    than the one used in the original pass.
"""

from __future__ import annotations

import math
import types
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "make_lora_bayesian",
    "lora_kl",
    "lora_sigma_stats",
    "mean_posterior_sigma",
    "is_bayesian",
    "iter_bayesian_lora_modules",
]

_RHO_ATTR = "rho"
_FLAG_ATTR = "_bayes_lora_prior_std"


def _rho_for_sigma(sigma: float) -> float:
    """Inverse softplus: rho such that softplus(rho) == sigma."""
    if sigma <= 0:
        raise ValueError(f"init_sigma must be positive, got {sigma}")
    # log(exp(sigma) - 1), computed stably for tiny sigma.
    return float(torch.log(torch.expm1(torch.tensor(sigma, dtype=torch.float64))).item())


def _iter_lora_b_linears(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    """Yield (qualified_name, module) for every PEFT ``lora_B.<adapter>`` layer.

    PEFT stores ``lora_B`` as an ``nn.ModuleDict`` keyed by adapter name whose
    values are bias-free ``nn.Linear`` layers, so the modules we want are the
    ones whose parent attribute is literally ``lora_B``.
    """
    for name, module in model.named_modules():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] == "lora_B":
            weight = getattr(module, "weight", None)
            if isinstance(weight, nn.Parameter):
                yield name, module


def iter_bayesian_lora_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    """Yield (name, module) for the lora_B layers that carry a posterior."""
    for name, module in model.named_modules():
        if hasattr(module, _FLAG_ATTR) and hasattr(module, _RHO_ATTR):
            yield name, module


def is_bayesian(model: nn.Module) -> bool:
    for _ in iter_bayesian_lora_modules(model):
        return True
    return False


def _bayesian_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Reparameterized forward for a lora_B layer.

    Train: sample W = mu + softplus(rho) * eps with a FRESH eps every call.
    Eval:  use the posterior mean mu (== self.weight), i.e. plain LoRA.
    """
    weight = self.weight
    if self.training:
        sigma = F.softplus(self.rho)
        eps = torch.randn_like(sigma)
        weight = weight + sigma * eps
    return F.linear(x, weight, self.bias)


def make_lora_bayesian(model: nn.Module, prior_std: float = 0.1, init_sigma: float = 1e-4):
    """Give every PEFT ``lora_B`` matrix a mean-field Gaussian posterior.

    The module's existing ``weight`` becomes the posterior mean ``mu``; a new
    ``rho`` parameter of identical shape is registered so that
    ``sigma = softplus(rho)``, initialised so ``sigma == init_sigma`` everywhere
    (tiny by default, so training starts at the deterministic LoRA solution).
    The module's ``forward`` is patched to sample in train mode and to use the
    mean in eval mode.

    Only ``lora_B`` is made stochastic -- see the module docstring for why.

    Returns a dict with ``num_modules``, ``num_parameters``, ``prior_std`` and
    ``init_sigma``. Idempotent: modules that already carry a posterior are left
    alone.
    """
    if prior_std <= 0:
        raise ValueError(f"prior_std must be positive, got {prior_std}")

    rho_init = _rho_for_sigma(init_sigma)
    num_modules = 0
    num_parameters = 0

    for _, module in _iter_lora_b_linears(model):
        if hasattr(module, _RHO_ATTR):
            continue
        rho = nn.Parameter(torch.full_like(module.weight, rho_init))
        rho.requires_grad_(module.weight.requires_grad)
        module.register_parameter(_RHO_ATTR, rho)
        setattr(module, _FLAG_ATTR, float(prior_std))
        # Bind the sampling forward to THIS instance only.
        module.forward = types.MethodType(_bayesian_linear_forward, module)
        num_modules += 1
        num_parameters += rho.numel()

    if num_modules == 0:
        raise RuntimeError(
            "make_lora_bayesian found no PEFT lora_B modules. Call it on the model "
            "returned by get_peft_model()."
        )

    return {
        "num_modules": num_modules,
        "num_parameters": num_parameters,
        "prior_std": float(prior_std),
        "init_sigma": float(init_sigma),
    }


def lora_kl(model: nn.Module) -> torch.Tensor:
    """Analytic KL( N(mu, sigma^2) || N(0, prior_std^2) ), summed over all entries.

    Per element:  log(prior/sigma) + (sigma^2 + mu^2) / (2 prior^2) - 1/2
    """
    total = None
    for _, module in iter_bayesian_lora_modules(model):
        prior_std = float(getattr(module, _FLAG_ATTR))
        mu = module.weight
        sigma = F.softplus(module.rho)
        kl = (
            math.log(prior_std)
            - torch.log(sigma)
            + (sigma.pow(2) + mu.pow(2)) / (2.0 * prior_std ** 2)
            - 0.5
        ).sum()
        total = kl if total is None else total + kl
    if total is None:
        device = next(model.parameters()).device if any(True for _ in model.parameters()) else "cpu"
        return torch.zeros((), device=device)
    return total


@torch.no_grad()
def lora_sigma_stats(model: nn.Module) -> dict:
    """Posterior-width summary, so collapse to a point estimate is visible."""
    total = 0.0
    count = 0
    smallest = math.inf
    largest = -math.inf
    mu_abs = 0.0
    for _, module in iter_bayesian_lora_modules(model):
        sigma = F.softplus(module.rho.detach().float())
        total += sigma.sum().item()
        count += sigma.numel()
        smallest = min(smallest, sigma.min().item())
        largest = max(largest, sigma.max().item())
        mu_abs += module.weight.detach().float().abs().sum().item()
    if count == 0:
        return {"sigma_mean": 0.0, "sigma_min": 0.0, "sigma_max": 0.0,
                "mu_abs_mean": 0.0, "num_parameters": 0}
    return {
        "sigma_mean": total / count,
        "sigma_min": smallest,
        "sigma_max": largest,
        "mu_abs_mean": mu_abs / count,
        "num_parameters": count,
    }


def mean_posterior_sigma(model: nn.Module) -> float:
    """Mean of softplus(rho) over every variational parameter."""
    return lora_sigma_stats(model)["sigma_mean"]


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def _fake_lora_layer(in_features: int, r: int, out_features: int) -> nn.Module:
    """Minimal stand-in with PEFT's ``lora_A``/``lora_B`` ModuleDict layout."""

    class FakeLora(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, r, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(r, out_features, bias=False)})

        def forward(self, x):
            return self.lora_B["default"](self.lora_A["default"](x))

    return FakeLora()


def _self_test() -> None:
    torch.manual_seed(0)
    ok = []

    # ---- (a) train mode is stochastic, eval mode is deterministic ----
    layer = _fake_lora_layer(8, 4, 6)
    # give B a non-zero mean so "close to eval output" is a meaningful check
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(0.0, 0.5)
    info = make_lora_bayesian(layer, prior_std=0.1, init_sigma=0.05)
    assert info["num_modules"] == 1, info
    assert info["num_parameters"] == 4 * 6, info
    x = torch.randn(3, 8)

    layer.train()
    t1, t2 = layer(x), layer(x)
    assert not torch.allclose(t1, t2), "train-mode forwards should differ (fresh eps each call)"

    layer.eval()
    e1, e2 = layer(x), layer(x)
    assert torch.equal(e1, e2), "eval-mode forwards must be identical (posterior mean)"

    # eval output must equal the deterministic mean-weight computation exactly
    mean_out = F.linear(F.linear(x, layer.lora_A["default"].weight), layer.lora_B["default"].weight)
    assert torch.equal(e1, mean_out), "eval mode must use mu exactly"
    ok.append("(a) train-mode sampling / eval-mode determinism")

    # sigma reporting
    stats = lora_sigma_stats(layer)
    assert abs(stats["sigma_mean"] - 0.05) < 1e-6, stats
    assert stats["num_parameters"] == 24, stats
    ok.append(f"(a') sigma reporting: mean sigma = {stats['sigma_mean']:.6g} (init 0.05)")

    # ---- (b) KL >= 0, and matches a hand computation on a one-element case ----
    one = _fake_lora_layer(1, 1, 1)
    make_lora_bayesian(one, prior_std=0.5, init_sigma=0.2)
    b = one.lora_B["default"]
    with torch.no_grad():
        b.weight.fill_(0.3)          # mu = 0.3
    mu, sigma, prior = 0.3, 0.2, 0.5
    expected = math.log(prior / sigma) + (sigma ** 2 + mu ** 2) / (2 * prior ** 2) - 0.5
    got = lora_kl(one).item()
    assert abs(got - expected) < 1e-6, f"KL mismatch: got {got}, expected {expected}"
    assert got >= 0.0
    ok.append(f"(b) analytic KL matches hand computation: {got:.9f} == {expected:.9f}")

    # KL >= 0 over random posteriors, and it is differentiable
    rnd = _fake_lora_layer(5, 3, 7)
    make_lora_bayesian(rnd, prior_std=0.1, init_sigma=1e-3)
    with torch.no_grad():
        rnd.lora_B["default"].weight.normal_(0, 0.3)
        rnd.lora_B["default"].rho.normal_(-3.0, 1.0)
    kl = lora_kl(rnd)
    assert kl.item() >= 0.0, kl.item()
    kl.backward()
    assert rnd.lora_B["default"].rho.grad is not None, "KL must be differentiable w.r.t. rho"
    assert rnd.lora_B["default"].weight.grad is not None, "KL must be differentiable w.r.t. mu"
    # KL is minimized at mu=0, sigma=prior_std -> equals 0 there
    zero = _fake_lora_layer(2, 2, 2)
    make_lora_bayesian(zero, prior_std=0.1, init_sigma=0.1)
    with torch.no_grad():
        zero.lora_B["default"].weight.zero_()
    assert abs(lora_kl(zero).item()) < 1e-6, lora_kl(zero).item()
    ok.append("(b') KL >= 0, differentiable in (mu, rho), and 0 at q == prior")

    # ---- (c) tiny init_sigma => train output ~= eval output ----
    tiny = _fake_lora_layer(8, 4, 6)
    with torch.no_grad():
        tiny.lora_B["default"].weight.normal_(0.0, 0.5)
    make_lora_bayesian(tiny, prior_std=0.1, init_sigma=1e-6)
    tiny.eval()
    ref = tiny(x)
    tiny.train()
    sampled = tiny(x)
    max_dev = (sampled - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert max_dev < 1e-3 * max(scale, 1.0), f"tiny sigma should stay near the mean, dev={max_dev}"
    assert not torch.equal(sampled, ref), "still stochastic, just tiny"
    ok.append(f"(c) init_sigma=1e-6: max|train - eval| = {max_dev:.3e} (output scale {scale:.3f})")

    # ---- (d) end-to-end on a real (tiny) PEFT LoRA model ----
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=2)
    base = GPT2LMHeadModel(cfg)
    peft_model = get_peft_model(
        base,
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.0,
                   target_modules=["c_attn", "c_proj"], bias="none"),
    )
    info = make_lora_bayesian(peft_model, prior_std=0.1, init_sigma=1e-4)
    assert info["num_modules"] == 2 * 3, info  # c_attn + attn.c_proj + mlp.c_proj per layer
    ids = torch.randint(0, 64, (2, 16))

    peft_model.eval()
    with torch.no_grad():
        a1 = peft_model(input_ids=ids, labels=ids).loss.item()
        a2 = peft_model(input_ids=ids, labels=ids).loss.item()
    assert a1 == a2, "PEFT model in eval mode must be deterministic"

    peft_model.train()
    # blow sigma up so the sampling noise is unmistakable through the whole net
    with torch.no_grad():
        for _, m in iter_bayesian_lora_modules(peft_model):
            m.rho.fill_(0.0)  # sigma = softplus(0) = 0.693
    b1 = peft_model(input_ids=ids, labels=ids).loss
    b2 = peft_model(input_ids=ids, labels=ids).loss
    assert b1.item() != b2.item(), "PEFT model in train mode must resample"
    kl = lora_kl(peft_model)
    assert kl.item() > 0
    (b1 + 1e-8 * kl).backward()
    grads = [m.rho.grad for _, m in iter_bayesian_lora_modules(peft_model)]
    assert all(g is not None and torch.isfinite(g).all() for g in grads), "rho must receive gradient"
    ok.append(f"(d) real PEFT model: {info['num_modules']} lora_B posteriors, "
              f"{info['num_parameters']:,} variational params, KL={kl.item():.3f}, rho grads flow")

    print("bayes_lora self-test PASSED")
    for line in ok:
        print("  ok  " + line)


if __name__ == "__main__":
    _self_test()
