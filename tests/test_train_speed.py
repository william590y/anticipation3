"""Guards for algorithm-neutral train.py speedups.

The L2-on-last-microbatch scaling must match adding the penalty on every
microbatch (weights do not change during accumulation). Checkpoints must not
go through ``accelerator.unwrap_model`` once the GPT-2 is torch.compiled.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

import train


def test_l2_last_microbatch_matches_every_microbatch_grads():
    torch.manual_seed(0)
    init = torch.randn(8, 8)
    reference = init.detach() + 0.1
    coef = 50.0
    accum = 4
    xs = [torch.randn(8, 8) for _ in range(accum)]

    param_every = nn.Parameter(init.clone())
    param_once = nn.Parameter(init.clone())

    def l2(param):
        return ((param.float() - reference.float()) ** 2).sum() / param.numel()

    param_every.grad = None
    for x in xs:
        loss = (param_every * x).sum() + coef * l2(param_every)
        (loss / accum).backward()

    param_once.grad = None
    for i, x in enumerate(xs):
        loss = (param_once * x).sum()
        if i == accum - 1:
            loss = loss + train.l2_anchor_loss_addend(l2(param_once), coef, accum)
        (loss / accum).backward()

    assert torch.allclose(param_every.grad, param_once.grad, rtol=1e-5, atol=1e-6)


def test_l2_addend_is_coefficient_times_accum():
    penalty = torch.tensor(0.25)
    addend = train.l2_anchor_loss_addend(penalty, 1e5, 4)
    assert torch.allclose(addend, torch.tensor(1e5 * 0.25 * 4))


def test_unwrap_ddp_leaves_plain_module():
    module = nn.Linear(2, 2)
    assert train.unwrap_ddp(module) is module


def test_compiled_root_walks_orig_mod():
    inner = nn.Linear(2, 2)

    class _Compiled(nn.Module):
        def __init__(self, orig):
            super().__init__()
            self._orig_mod = orig

    wrapped = _Compiled(inner)
    assert train.compiled_root(wrapped) is inner
    assert train.saveable_model(wrapped) is inner


def test_compile_causal_lm_disabled_is_identity():
    module = nn.Linear(2, 2)
    assert train.compile_causal_lm(module, enabled=False, mode="default") is module


def test_train_py_does_not_unwrap_compiled_ddp():
    src = inspect.getsource(train)
    assert "accelerator.unwrap_model(" not in src
    assert "saveable_model(" in src
    assert "torch.compile" in src
    assert "enable_fast_kernels" in src
    assert "attn_implementation" in src
    main_src = inspect.getsource(train.main)
    assert "l2_anchor_loss_addend" in main_src
    assert "accelerator.sync_gradients" in main_src
