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
    assert "infer_paired" in eval_src
    assert "batch_idx % " not in eval_src
    main_src = inspect.getsource(train_ltlm.main)
    assert "InitProcessGroupKwargs" in main_src
    assert "torch.compile" in src
    assert "broadcast_offsets" in src
    assert "attn_implementation" in src
    assert "val_loader" in main_src
    # Validation and checkpointing are entered by every rank.
    assert "if accelerator.is_main_process:\n            max_batches" not in main_src
    assert "accelerator.wait_for_everyone()" in inspect.getsource(train_ltlm.main)


def test_compile_ltlm_disabled_is_identity():
    module = nn.Linear(2, 2)
    assert train_ltlm.compile_ltlm(module, enabled=False, mode="default") is module


def test_broadcast_offsets_single_process_passthrough():
    class _OneRank:
        num_processes = 1

    offsets = [0, 10, 20]
    got, seqlen = train_ltlm.broadcast_offsets(_OneRank(), offsets, 1020)
    assert got is offsets
    assert seqlen == 1020


def test_infer_paired_splits_doubled_batch():
    from anticipation.ltlm_posterior import PosteriorOptimizer

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        def init_posterior(self, input_ids):
            batch = input_ids.shape[0]
            return torch.zeros(batch, 2, 4), torch.zeros(batch, 2, 4)

        def forward(self, input_ids=None, labels=None, mu_q=None, log_var_q=None, eps=None, **kwargs):
            loss = (mu_q.float() ** 2).mean() + 0.0 * self.weight.sum()
            return loss, {"loss": loss.detach()}, None

    model = Tiny()
    posterior = PosteriorOptimizer(model, num_steps=2, lr=0.05, final_lr=0.05)
    ids = torch.zeros(3, 5, dtype=torch.long)
    labels_a = torch.zeros(3, 5, dtype=torch.long)
    labels_b = torch.ones(3, 5, dtype=torch.long)
    left, right = posterior.infer_paired(ids, labels_a, labels_b)
    assert left["mu_q"].shape[0] == 3
    assert right["mu_q"].shape[0] == 3
    assert left["z"].shape == right["z"].shape


def test_tokenized_dataset_reuses_broadcast_offsets(tmp_path):
    from train import TokenizedDataset

    path = tmp_path / "toy.txt"
    path.write_text("1 2 3 |\n4 5 6 |\n")
    scanned = TokenizedDataset(path, is_training=False)
    reused = TokenizedDataset(
        path,
        is_training=False,
        offsets=scanned.offsets,
        sequence_length=scanned.sequence_length,
    )
    assert reused.offsets == scanned.offsets
    assert len(reused) == len(scanned)
