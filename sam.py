"""Fisher SAM: sharpness-aware minimization on an information-geometric ball.

Kim et al., *Fisher SAM: Information Geometry and Sharpness Aware Minimisation*
(ICML 2022).  Plain SAM (Foret et al., 2021) ascends inside a Euclidean ball,

    e_w = rho * g / ||g||_2 ,

which treats every coordinate as equally "far".  Fisher SAM replaces the ball
with an ellipsoid induced by the diagonal empirical Fisher (squared gradients),
so flat coordinates are allowed to move further than sharp ones:

    u_i  = g_i / (1 + eta * g_i^2)
    e_w  = rho * u / ||u||_2                      # == rho*(g_i/(1+eta g_i^2))
                                                  #    / sqrt(sum_j g_j^2/(1+eta g_j^2)^2)

``eta = 0`` makes ``u = g`` and the update collapses **exactly** to plain SAM
(the implementation short-circuits that case, so it is exact and not merely
numerically close).

The optimizer is a two-pass scheme per (micro-)batch:

  1. forward/backward at ``w``                      -> g
  2. ``first_step()``  : w <- w + e_w               (ascent to the worst point)
  3. forward/backward at ``w + e_w``                -> g'
  4. ``restore()``     : w <- w                     (undo the ascent)
  5. ``step()``        : the BASE optimizer (AdamW) applies g'

Everything except step 5 lives here; the base optimizer is untouched and keeps
owning the learning rate, so an LR scheduler can stay attached to it.

Design notes that matter for the training loop
----------------------------------------------
* **Not a ``torch.optim.Optimizer`` subclass.**  ``train.py`` hands the base
  optimizer to ``accelerator.prepare`` and to ``LambdaLR``; wrapping it in a
  second Optimizer object would put a stale copy of the param groups in front
  of both.  ``FisherSAM`` therefore *delegates* ``step``/``zero_grad``/
  ``state_dict`` to the (possibly accelerate-wrapped) base optimizer and only
  adds the perturbation machinery.
* **``zero_param_grads()`` exists on purpose.**  ``AcceleratedOptimizer.
  zero_grad()`` is a no-op while ``sync_gradients`` is False -- that is how
  accelerate lets a naive loop call ``zero_grad()`` every micro-batch without
  destroying gradient accumulation.  SAM needs a *real* zero between its two
  passes, so it clears ``p.grad`` directly.
* **Manual accumulation.**  Because the ascent direction must be built from
  THIS micro-batch's gradient alone, ``p.grad`` cannot double as the
  accumulator across micro-batches.  ``accumulate_grads()`` adds the perturbed
  gradient into a private fp32 buffer and ``write_accumulated_grads()`` copies
  the sum back into ``p.grad`` at the accumulation boundary (m-sharpness: the
  sharpness is measured per micro-batch, which is the variant Foret et al.
  report as the better-performing one anyway).
* **``grad_scale``.**  ``accelerator.backward`` divides the loss by
  ``gradient_accumulation_steps``, so ``p.grad`` holds ``g/N``.  Plain SAM is
  scale invariant (the normalisation cancels), but the Fisher metric
  ``1 + eta*g^2`` is NOT: eta is defined against the true micro-batch mean
  gradient.  Pass ``grad_scale=N`` to undo accelerate's division.
* **``exact_restore``** (default): remember the pre-ascent weights and copy
  them back, which is *bitwise* exact.  ``exact_restore=False`` remembers
  ``e_w`` itself and restores with ``p -= e_w`` (the textbook formulation, one
  buffer either way, but ``(w + e) - e != w`` in floating point).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import torch


class FisherSAM:
    """Fisher-SAM perturbation wrapper around an arbitrary base optimizer."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer,
        rho: float = 0.05,
        eta: float = 1.0,
        exact_restore: bool = True,
        eps: float = 1e-12,
    ):
        params = [p for p in params if p.requires_grad]
        if not params:
            raise ValueError("FisherSAM received no trainable parameters.")
        if rho < 0.0:
            raise ValueError(f"rho must be >= 0, got {rho}")
        if eta < 0.0:
            raise ValueError(f"eta must be >= 0, got {eta}")
        if base_optimizer is None:
            raise ValueError("FisherSAM needs a base optimizer to apply the gradient.")

        self.params: List[torch.nn.Parameter] = params
        self.base_optimizer = base_optimizer
        self.rho = float(rho)
        self.eta = float(eta)
        self.exact_restore = bool(exact_restore)
        self.eps = float(eps)

        # Persistent buffers (allocated on first use, reused forever after so
        # the training loop does not churn the caching allocator).
        self._saved: List[Optional[torch.Tensor]] = [None] * len(params)
        self._accum: List[Optional[torch.Tensor]] = [None] * len(params)

        self._perturbed = False
        self._accum_count = 0
        # Diagnostics for the training loop / logs.
        self.last_ascent_norm: Optional[float] = None      # ||u||_2
        self.last_perturbation_norm: Optional[float] = None  # ||e_w||_2 (== rho)

    # ------------------------------------------------------------------ #
    # base-optimizer delegation
    # ------------------------------------------------------------------ #
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def step(self, *args, **kwargs):
        """Apply the (perturbed) gradient with the base optimizer."""
        if self._perturbed:
            raise RuntimeError(
                "FisherSAM.step() called while the weights are still perturbed; "
                "call restore() first."
            )
        return self.base_optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        """Delegate to the base optimizer.

        NOTE: under ``accelerate``'s ``AcceleratedOptimizer`` this is a no-op
        during gradient accumulation.  Use :meth:`zero_param_grads` for the
        zeroing that SAM's two passes actually depend on.
        """
        return self.base_optimizer.zero_grad(*args, **kwargs)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.base_optimizer.load_state_dict(state_dict)

    # ------------------------------------------------------------------ #
    # gradient plumbing
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def zero_param_grads(self, set_to_none: bool = True) -> None:
        """Really clear ``p.grad`` on the tracked parameters."""
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    @torch.no_grad()
    def accumulate_grads(self) -> int:
        """Add the current ``p.grad`` into the private fp32 accumulator."""
        touched = 0
        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue
            # Accumulate at least in fp32: summing N micro-batch gradients in
            # bf16 would lose most of the small ones.
            dtype = (
                torch.float32
                if g.dtype in (torch.float16, torch.bfloat16)
                else g.dtype
            )
            buf = self._accum[i]
            if buf is None or buf.shape != g.shape or buf.dtype != dtype:
                buf = torch.zeros(g.shape, dtype=dtype, device=g.device)
                self._accum[i] = buf
            buf.add_(g.detach().to(dtype))
            touched += 1
        self._accum_count += 1
        return touched

    @torch.no_grad()
    def write_accumulated_grads(self) -> int:
        """Copy the accumulator into ``p.grad`` and clear it.

        Called at the accumulation boundary, BEFORE gradient clipping and the
        base optimizer step, so the existing NaN/Inf gradient guard in the
        training loop still inspects the gradient that will actually be used.
        """
        touched = 0
        for i, p in enumerate(self.params):
            buf = self._accum[i]
            if buf is None:
                continue
            if p.grad is None:
                p.grad = buf.detach().clone().to(p.dtype)
            else:
                p.grad.copy_(buf)
            buf.zero_()
            touched += 1
        self._accum_count = 0
        return touched

    @torch.no_grad()
    def reset_accumulation(self) -> None:
        """Throw away a partially accumulated step (NaN recovery paths)."""
        for buf in self._accum:
            if buf is not None:
                buf.zero_()
        self._accum_count = 0

    @property
    def accumulated_microbatches(self) -> int:
        return self._accum_count

    @property
    def is_perturbed(self) -> bool:
        return self._perturbed

    # ------------------------------------------------------------------ #
    # the Fisher-SAM ascent step
    # ------------------------------------------------------------------ #
    def _u(self, grad: torch.Tensor, grad_scale: float) -> torch.Tensor:
        """u_i = g_i / (1 + eta * g_i^2), computed in fp32 (fp64 stays fp64)."""
        g = grad.detach()
        if g.dtype not in (torch.float32, torch.float64):
            g = g.to(torch.float32)
        if grad_scale != 1.0:
            g = g * grad_scale
        if self.eta == 0.0:
            return g  # plain SAM, exactly
        return g / (1.0 + self.eta * g * g)

    @torch.no_grad()
    def _ascent_norm(self, grad_scale: float) -> float:
        total = None
        for p in self.params:
            if p.grad is None:
                continue
            u = self._u(p.grad, grad_scale)
            # Reduce inside each tensor at its own dtype (torch.sum is pairwise,
            # so fp32 is plenty for a normalising constant) and accumulate the
            # SCALARS in fp64. Casting the gradients themselves to fp64 would
            # allocate a second full-size copy of the largest tensors.
            s = torch.sum(u * u).to(torch.float64)
            total = s if total is None else total + s
        if total is None:
            return 0.0
        return float(total.sqrt().item())

    @torch.no_grad()
    def first_step(self, grad_scale: float = 1.0, zero_grad: bool = False) -> bool:
        """Perturb the weights to ``w + e_w`` using the gradients in ``p.grad``.

        Returns True if a perturbation was applied.  A zero or non-finite
        gradient yields no perturbation (the second pass then simply
        re-evaluates the same point, i.e. the update degrades to plain AdamW
        for that micro-batch) -- this keeps every rank on the same code path
        without an extra collective.
        """
        if self._perturbed:
            raise RuntimeError("first_step() called twice without restore().")

        norm = self._ascent_norm(grad_scale)
        self.last_ascent_norm = norm
        self._perturbed = True

        if norm == 0.0 or not math.isfinite(norm) or self.rho == 0.0:
            self.last_perturbation_norm = 0.0
            for i in range(len(self.params)):
                self._saved[i] = None
            if zero_grad:
                self.zero_param_grads()
            return False

        scale = self.rho / (norm + self.eps)
        applied_sq = None
        for i, p in enumerate(self.params):
            if p.grad is None:
                self._saved[i] = None
                continue
            e_w = (self._u(p.grad, grad_scale) * scale).to(p.dtype)
            buf = self._saved[i]
            if buf is None or buf.shape != p.shape or buf.dtype != p.dtype:
                buf = torch.empty_like(p)
                self._saved[i] = buf
            if self.exact_restore:
                buf.copy_(p)      # remember w  -> bitwise restore
            else:
                buf.copy_(e_w)    # remember e_w -> restore by subtraction
            p.add_(e_w)
            # Diagnostic only (this norm is rho by construction). Accumulated on
            # device and synchronised ONCE: an .item() per tensor would be ~300
            # device syncs per micro-batch.
            sq = torch.sum(e_w * e_w).to(torch.float64)
            applied_sq = sq if applied_sq is None else applied_sq + sq
        self.last_perturbation_norm = (
            0.0 if applied_sq is None else float(applied_sq.sqrt().item())
        )
        if zero_grad:
            self.zero_param_grads()
        return True

    @torch.no_grad()
    def restore(self) -> None:
        """Undo :meth:`first_step`, returning the weights to ``w``."""
        if not self._perturbed:
            raise RuntimeError("restore() called without a preceding first_step().")
        for i, p in enumerate(self.params):
            buf = self._saved[i]
            if buf is None:
                continue
            if self.exact_restore:
                p.copy_(buf)      # bitwise identical to the pre-ascent value
            else:
                p.sub_(buf)
        self._perturbed = False

    # Aliases matching the common SAM naming, for readability at call sites.
    ascent_step = first_step
    descent_step = restore

    def __repr__(self) -> str:
        kind = "plain-SAM (eta=0)" if self.eta == 0.0 else f"Fisher-SAM (eta={self.eta})"
        return (
            f"FisherSAM(rho={self.rho}, {kind}, exact_restore={self.exact_restore}, "
            f"base={type(self.base_optimizer).__name__}, "
            f"tensors={len(self.params)})"
        )


# ---------------------------------------------------------------------- #
# self-test
# ---------------------------------------------------------------------- #
def _reference_perturbation(grads, rho, eta):
    """Hand-computed e_w over the flattened gradient, in float64."""
    flat = torch.cat([g.reshape(-1).to(torch.float64) for g in grads])
    u = flat if eta == 0.0 else flat / (1.0 + eta * flat * flat)
    return rho * u / u.norm()


def _selftest():
    torch.manual_seed(0)

    from train import make_adamw  # the base optimizer the training loop uses

    # ---- a tiny, well-posed regression problem, in float64 ----------------
    n, d = 64, 5
    X = torch.randn(n, d, dtype=torch.float64)
    w_true = torch.randn(d, 1, dtype=torch.float64)
    y = X @ w_true + 0.01 * torch.randn(n, 1, dtype=torch.float64)

    def make_model():
        torch.manual_seed(1)
        return torch.nn.Sequential(
            torch.nn.Linear(d, 4, dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 1, dtype=torch.float64),
        )

    def loss_of(model):
        return torch.nn.functional.mse_loss(model(X), y)

    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    # ---- (a) eta = 0 reproduces the hand-computed plain-SAM perturbation --
    print("test (a): eta=0 == plain SAM, and eta>0 == the Fisher formula")
    for eta in (0.0, 1.0, 25.0):
        model = make_model()
        params = list(model.parameters())
        sam = FisherSAM(params, make_adamw(params, 1e-3), rho=0.05, eta=eta)
        loss_of(model).backward()
        grads = [p.grad.detach().clone() for p in params]
        before = [p.detach().clone() for p in params]
        sam.first_step()
        applied = torch.cat(
            [(p.detach() - b).reshape(-1) for p, b in zip(params, before)]
        )
        expected = _reference_perturbation(grads, rho=0.05, eta=eta)
        err = float((applied - expected).abs().max())
        # relative to the largest component; the only difference from the
        # reference is float64 summation order (per-tensor vs one flat vector)
        rel = err / float(expected.abs().max())
        check(
            f"eta={eta}: e_w matches closed form (max|err|={err:.3e}, rel={rel:.1e})",
            rel < 1e-11,
        )
        check(
            f"eta={eta}: ||e_w||_2 == rho ({sam.last_perturbation_norm:.12f})",
            abs(sam.last_perturbation_norm - 0.05) < 1e-12,
        )
        sam.restore()

    # eta=0 must be plain SAM *exactly*, and eta>0 must actually differ.
    model = make_model()
    params = list(model.parameters())
    loss_of(model).backward()
    grads = [p.grad.detach().clone() for p in params]
    plain = _reference_perturbation(grads, 0.05, 0.0)
    fisher = _reference_perturbation(grads, 0.05, 25.0)
    check(
        "eta>0 bends the direction away from plain SAM",
        float((plain - fisher).abs().max()) > 1e-6,
        f"max|delta|={float((plain - fisher).abs().max()):.3e}",
    )

    # ---- (a2) plain SAM is invariant to the gradient scale ---------------
    for scale in (1.0, 8.0):
        model = make_model()
        params = list(model.parameters())
        sam = FisherSAM(params, make_adamw(params, 1e-3), rho=0.05, eta=0.0)
        loss_of(model).backward()
        before = [p.detach().clone() for p in params]
        sam.first_step(grad_scale=scale)
        applied = torch.cat(
            [(p.detach() - b).reshape(-1) for p, b in zip(params, before)]
        )
        expected = _reference_perturbation(
            [p.grad for p in params], rho=0.05, eta=0.0
        )
        check(
            f"eta=0 invariant to grad_scale={scale}",
            float((applied - expected).abs().max()) < 1e-14,
        )
        sam.restore()

    # ---- (b) restore() is bitwise exact ----------------------------------
    print("test (b): restore() returns the weights bitwise")
    model = make_model()
    params = list(model.parameters())
    sam = FisherSAM(params, make_adamw(params, 1e-3), rho=0.1, eta=1.0)
    loss_of(model).backward()
    before = [p.detach().clone() for p in params]
    sam.first_step()
    moved = all(not torch.equal(p.detach(), b) for p, b in zip(params, before))
    check("first_step() actually moved every tensor", moved)
    sam.restore()
    bitwise = all(torch.equal(p.detach(), b) for p, b in zip(params, before))
    check("exact_restore=True -> torch.equal on every tensor", bitwise)

    # the textbook subtract-e_w variant: correct, but only to round-off
    model = make_model()
    params = list(model.parameters())
    sam = FisherSAM(
        params, make_adamw(params, 1e-3), rho=0.1, eta=1.0, exact_restore=False
    )
    loss_of(model).backward()
    before = [p.detach().clone() for p in params]
    sam.first_step()
    sam.restore()
    worst = max(
        float((p.detach() - b).abs().max()) for p, b in zip(params, before)
    )
    check(
        "exact_restore=False -> restores to round-off",
        worst < 1e-15,
        f"max|drift|={worst:.3e}",
    )

    # ---- (c) the manual accumulator sums micro-batch gradients ------------
    print("test (c): manual gradient accumulation")
    model = make_model()
    params = list(model.parameters())
    sam = FisherSAM(params, make_adamw(params, 1e-3), rho=0.05, eta=1.0)
    chunks = [(X[:32], y[:32]), (X[32:], y[32:])]
    manual = None
    for xb, yb in chunks:
        sam.zero_param_grads()
        torch.nn.functional.mse_loss(model(xb), yb).backward()
        g = torch.cat([p.grad.reshape(-1).clone() for p in params])
        manual = g if manual is None else manual + g
        sam.accumulate_grads()
    check("accumulated 2 micro-batches", sam.accumulated_microbatches == 2)
    sam.zero_param_grads()
    sam.write_accumulated_grads()
    written = torch.cat([p.grad.reshape(-1) for p in params])
    check(
        "write_accumulated_grads() == sum of micro-batch grads",
        float((written - manual).abs().max()) < 1e-15,
    )
    check("accumulator emptied after write", sam.accumulated_microbatches == 0)
    sam.zero_param_grads()
    sam.accumulate_grads()  # nothing to add (grads are None)
    sam.reset_accumulation()

    # ---- (d) end-to-end: it fits the toy problem -------------------------
    print("test (d): the full two-pass loop fits the toy problem")
    for eta in (0.0, 1.0):
        model = make_model()
        params = list(model.parameters())
        sam = FisherSAM(params, make_adamw(params, 5e-2), rho=0.05, eta=eta)
        start = float(loss_of(model).detach())
        for _ in range(300):
            sam.zero_param_grads()
            loss_of(model).backward()          # pass 1 at w
            sam.first_step()
            sam.zero_param_grads()
            loss_of(model).backward()          # pass 2 at w + e_w
            sam.restore()
            sam.accumulate_grads()
            sam.zero_param_grads()
            sam.write_accumulated_grads()
            sam.step()
        end = float(loss_of(model).detach())
        check(
            f"eta={eta}: loss {start:.4f} -> {end:.4f}",
            end < 0.1 * start and math.isfinite(end),
        )

    # ---- (e) guard rails --------------------------------------------------
    print("test (e): API guards")
    model = make_model()
    params = list(model.parameters())
    sam = FisherSAM(params, make_adamw(params, 1e-3), rho=0.05, eta=1.0)
    try:
        sam.restore()
        check("restore() before first_step() raises", False)
    except RuntimeError:
        check("restore() before first_step() raises", True)
    loss_of(model).backward()
    sam.first_step()
    try:
        sam.first_step()
        check("double first_step() raises", False)
    except RuntimeError:
        check("double first_step() raises", True)
    try:
        sam.step()
        check("step() while perturbed raises", False)
    except RuntimeError:
        check("step() while perturbed raises", True)
    sam.restore()

    # zero gradients must not produce NaNs
    model = make_model()
    params = list(model.parameters())
    sam = FisherSAM(params, make_adamw(params, 1e-3), rho=0.05, eta=1.0)
    for p in params:
        p.grad = torch.zeros_like(p)
    before = [p.detach().clone() for p in params]
    moved = sam.first_step()
    check("zero gradient -> no perturbation, no NaN", (not moved) and all(
        torch.equal(p.detach(), b) for p, b in zip(params, before)))
    sam.restore()
    for p in params:
        p.grad = torch.full_like(p, float("nan"))
    moved = sam.first_step()
    check("NaN gradient -> no perturbation", not moved)
    sam.restore()
    check("weights finite after a NaN gradient",
          all(torch.isfinite(p).all() for p in params))

    print()
    if failures:
        raise SystemExit(f"FisherSAM self-test FAILED: {failures}")
    print("FisherSAM self-test: all checks passed.")


if __name__ == "__main__":
    _selftest()
