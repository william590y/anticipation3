"""Guards against the DDP desync that killed LTLM planner job 69260.

That job called ``unwrap_model(model).elbo(...)`` for the slow step, so
``DDP.forward`` / ``prepare_for_backward`` never ran. Ranks trained independent
copies; rank 0 lagged on wandb.log; ranks 1-2 hit ``run_validation`` first and
NCCL-timeouted on ``wait_for_everyone``. Rank-0-only eval between barriers is
the same failure mode even with working DDP if validation exceeds the timeout.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

import train_ltlm
from anticipation.ltlm_model import LTLMCausalLM


class _Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0
        self.elbo_calls = 0
        self.weight = nn.Parameter(torch.ones(1))

    def elbo(self, *args, **kwargs):
        self.elbo_calls += 1
        loss = self.weight.sum()
        return loss, {"loss": loss.detach()}, None

    def forward(self, input_ids=None, labels=None, attention_mask=None, mu_q=None, **kwargs):
        self.forward_calls += 1
        if mu_q is not None:
            return self.elbo(input_ids, labels, mu_q, **kwargs)
        return self.weight.sum()


def test_slow_elbo_goes_through_module_forward():
    model = _Recorder()
    dummy = torch.zeros(1, 1)
    loss, stats, _ = train_ltlm.slow_elbo(model, dummy, dummy, dummy, dummy, dummy)
    assert model.forward_calls == 1
    assert model.elbo_calls == 1
    loss.backward()
    assert model.weight.grad is not None


def test_ltlm_forward_dispatches_to_elbo_when_mu_q_is_set():
    called = {}

    def fake_elbo(self, *args, **kwargs):
        called["yes"] = True
        return torch.tensor(0.0), {}, None

    obj = LTLMCausalLM.__new__(LTLMCausalLM)
    obj.elbo = fake_elbo.__get__(obj, LTLMCausalLM)
    result = LTLMCausalLM.forward(
        obj,
        input_ids=torch.zeros(1, 1),
        labels=torch.zeros(1, 1),
        mu_q=torch.zeros(1),
        log_var_q=torch.zeros(1),
        eps=torch.zeros(1),
    )
    assert called["yes"]
    assert result[0].item() == 0.0


def test_train_ltlm_source_does_not_bypass_ddp():
    src = inspect.getsource(train_ltlm)
    assert "raw_model.elbo(" not in src
    assert "slow_elbo(" in src
    eval_src = inspect.getsource(train_ltlm.evaluate_paths)
    assert "accelerator.reduce" in eval_src
    main_src = inspect.getsource(train_ltlm.main)
    assert "InitProcessGroupKwargs" in main_src
    # Validation and checkpointing are entered by every rank.
    assert "if accelerator.is_main_process:\n            max_batches" not in main_src
    assert "accelerator.wait_for_everyone()" in inspect.getsource(train_ltlm.main)
