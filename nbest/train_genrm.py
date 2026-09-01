#!/usr/bin/env python
"""Generative verifier (GenRM, arXiv:2408.15240) for N-best selection.

Reward modelling as NEXT-TOKEN PREDICTION on the generator's own backbone,
rather than a discriminative head. The verifier is the FT AMT model itself,
its vocabulary extended by three tokens:

    ASK  -- appended after the window-with-candidate, the verdict position
    YES  -- target when the candidate is good
    NO   -- target when it is not

    score(x, y) = p_theta(YES | window-with-candidate, ASK)          (Eq. 4)

trained with the paper's JOINT objective (Eq. 5)

    L = L_verify + lambda * L_generate,     lambda = 1/3

where L_generate is ordinary next-token loss on GROUND-TRUTH score tokens --
their D_correct term, which their ablation shows is not optional (lambda = 0
is consistently worse, and the generation data also improves the model's own
Pass@N).

Two adaptations, both forced by the domain and documented here because they
are where this departs from the paper:

1. BINARY LABELS FROM A GRADED METRIC. Their labels come from an exact-match
   answer checker; F1 is continuous. We binarise WITHIN each window at its
   own median F1 -- top half YES, bottom half NO. That is balanced 50/50 by
   construction (which the paper requires) and, more importantly, makes the
   label a WITHIN-POOL judgement: a global F1 threshold would mostly encode
   window difficulty, the exact failure we diagnosed in the fitted alpha/beta
   objective, whose features were between-window proxies with near-zero
   within-pool signal.
2. NO CoT VARIANT. GenRM's gains come mostly from chain-of-thought rationales;
   ours would have to be a synthesised symbolic verification trace (the
   paper's algorithmic tasks show templated traces work), which is a larger
   project. This is direct GenRM, which the paper reports as roughly matching
   a well-tuned discriminative RM -- so this row is a like-for-like test of
   the generative parameterisation, not of CoT.

  torchrun --standalone --nproc_per_node=1 -m nbest.train_genrm \
      --shards 'nbest_data/unf32_train_shard*.pt' \
      --run_dir run_nbest_reranker/genrm_<stamp>
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import contextlib

import torch

# Training dtype is a MEASURED CONFOUND in this repo, not a free choice: a bf16
# weight cast and a bf16 autocast have been measured to move F1 in OPPOSITE
# directions (0.3533->0.3208 over 208 windows; 22.71->24.94 over 24), so a
# run's dtype must be recorded and controllable rather than hardcoded.
# Default is unchanged (bf16 autocast); --fp32 makes the forward full precision.
_FP32 = False


def _AC():
    if _FP32:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16,
                          enabled=torch.cuda.is_available())
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from anticipation.vocab import VOCAB_SIZE
from evaluate_muster import load_model
from nbest.reranker import substitute_candidates
from nbest.train_reranker_pairwise import (cosine_lr, load_shards,
                                           split_windows)
from onpolicy_rollout import score_token_positions

WANDB_PROJECT = "anticipation-asap"
ASK, YES, NO = VOCAB_SIZE, VOCAB_SIZE + 1, VOCAB_SIZE + 2
NEW_VOCAB = VOCAB_SIZE + 3


def window_labels(f1: torch.Tensor) -> torch.Tensor:
    """Within-window median split -> 1.0 (YES) / 0.0 (NO)."""
    return (f1 > f1.median()).float()


def verify_batch(model, wins, cands, flat_pos, device):
    """-> (B,) log p(YES) - log p(NO) at the verdict position, and log p(YES)."""
    toks = substitute_candidates(wins, cands, flat_pos).to(device)
    ask = torch.full((toks.shape[0], 1), ASK, dtype=torch.long, device=device)
    seq = torch.cat([toks, ask], dim=1)
    logits = model(seq, use_cache=False).logits[:, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    return logp[:, YES], logp[:, NO]


@torch.no_grad()
def selection_metrics(model, data, hold_rows, flat_pos, device,
                      max_windows=150, batch=8):
    model.eval()
    sel, greedy, oracle, accs = [], [], [], []
    for r in hold_rows[:max_windows]:
        cs = data["cands_of_row"][r]
        idx = torch.tensor(cs)
        f1 = data["cand_f1"][idx]
        kind = data["cand_kind"][idx].numpy()
        scores = torch.empty(len(cs))
        for lo in range(0, len(cs), batch):
            part = idx[lo:lo + batch]
            wins = data["win_tok"][data["cand_win_row"][part]]
            with _AC():
                y, _ = verify_batch(model, wins, data["cand_tok"][part],
                                    flat_pos, device)
            scores[lo:lo + len(part)] = y.float().cpu()
        sel.append(float(f1[int(scores.argmax())]))
        g = f1.numpy()[kind == 0]
        greedy.append(float(g[0]) if len(g) else float("nan"))
        oracle.append(float(f1.max()))
        lab = window_labels(f1)
        accs.append(float(((scores > np.log(0.5)).float() == lab).float().mean()))
    model.train()
    return {"sel_f1": float(np.nanmean(sel)),
            "greedy_f1": float(np.nanmean(greedy)),
            "oracle_f1": float(np.nanmean(oracle)),
            "verdict_acc": float(np.mean(accs)),
            "n_windows": len(sel)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default="run_paper_split_v2/checkpoint-2500")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--cands_per_step", type=int, default=4,
                    help="verification examples per optimizer step")
    ap.add_argument("--lam_generate", type=float, default=1 / 3,
                    help="paper's lambda on the D_correct generation loss")
    ap.add_argument("--lr", type=float, default=2e-6,
                    help="the paper's verifier LR (this is a 420M generator "
                         "being fine-tuned, not a small head being trained)")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb_mode", default="online")
    ap.add_argument("--fp32", action="store_true",
                    help="train and evaluate in full precision (no bf16 "
                         "autocast). The SHARDS' decode dtype is a separate "
                         "axis -- match them deliberately.")
    a = ap.parse_args()
    global _FP32
    _FP32 = bool(a.fp32)
    print(f"forward dtype: {'fp32' if _FP32 else 'bf16 autocast'}", flush=True)

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
    is_main = rank == 0
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed + rank)
    gen = torch.Generator().manual_seed(a.seed + rank)

    data = load_shards(a.shards)
    train_rows, hold_rows = split_windows(data)
    flat_pos = score_token_positions(data["win_tok"].shape[1])

    model, _ = load_model(a.checkpoint)
    model.resize_token_embeddings(NEW_VOCAB, mean_resizing=False)
    with torch.no_grad():   # plan_lm.py's lesson: default mean-resizing makes
        emb = model.get_input_embeddings().weight   # new rows ~265x too small
        std = emb[:VOCAB_SIZE].std()
        emb[VOCAB_SIZE:].normal_(0.0, float(std))
        out = model.get_output_embeddings().weight
        if out is not emb:
            out[VOCAB_SIZE:].normal_(0.0, float(out[:VOCAB_SIZE].std()))
    model = model.to(device)
    net = DDP(model, device_ids=[local], output_device=local) \
        if world > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=a.weight_decay)
    if is_main:
        os.makedirs(a.run_dir, exist_ok=True)
        print(f"GenRM backbone: {sum(p.numel() for p in model.parameters())/1e6:.0f}M "
              f"params, vocab {VOCAB_SIZE} -> {NEW_VOCAB}", flush=True)
        wandb.init(project=WANDB_PROJECT, mode=a.wandb_mode,
                   name=f"nbest_{Path(a.run_dir).name}",
                   config={**vars(a), "loss": "genrm_joint_next_token"})

    def save(step, name=None):
        path = os.path.join(a.run_dir, name or f"ckpt_step{step:07d}.pt")
        torch.save({"model": model.state_dict(), "step": step,
                    "cfg": vars(a), "special": {"ASK": ASK, "YES": YES,
                                                "NO": NO}}, path)
        return path

    rows_t = torch.tensor(train_rows, dtype=torch.long)
    best_sel, t_last, s_last = -1.0, time.time(), 0
    model.train()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, a.lr, a.warmup, a.steps)
        r = int(rows_t[torch.randint(0, len(rows_t), (1,),
                                     generator=gen)].item())
        cs = data["cands_of_row"][r]
        idx_all = torch.tensor(cs)
        f1_all = data["cand_f1"][idx_all]
        lab_all = window_labels(f1_all)
        # Balanced 50/50 draw, as the paper requires.
        pos = [i for i in range(len(cs)) if lab_all[i] > 0.5]
        neg = [i for i in range(len(cs)) if lab_all[i] <= 0.5]
        half = max(1, a.cands_per_step // 2)
        pick = ([pos[int(torch.randint(0, len(pos), (1,), generator=gen))]
                 for _ in range(half)] if pos else []) + \
               ([neg[int(torch.randint(0, len(neg), (1,), generator=gen))]
                 for _ in range(half)] if neg else [])
        sel = idx_all[torch.tensor(pick)]
        wins = data["win_tok"][data["cand_win_row"][sel]]
        target = lab_all[torch.tensor(pick)].to(device)

        with _AC():
            y, n = verify_batch(net, wins, data["cand_tok"][sel], flat_pos,
                                device)
            # CE over the two verdict tokens = next-token loss restricted to
            # the only two legal continuations.
            two = torch.stack([n, y], dim=-1)
            l_verify = F.cross_entropy(two, target.long())
            # D_correct: plain next-token loss on the window's own GT tokens.
            gt = data["win_tok"][data["cand_win_row"][sel[:1]]].to(device).long()
            out = net(gt, use_cache=False).logits
            l_gen = F.cross_entropy(
                out[:, :-1].reshape(-1, out.shape[-1]).float(),
                gt[:, 1:].reshape(-1))
            loss = l_verify + a.lam_generate * l_gen
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()

        if is_main and step % a.log_every == 0:
            now = time.time()
            rate = (step - s_last) / max(now - t_last, 1e-6)
            t_last, s_last = now, step
            wandb.log({"train/loss": float(loss.detach()),
                       "train/l_verify": float(l_verify.detach()),
                       "train/l_generate": float(l_gen.detach()),
                       "train/p_yes": float(y.exp().mean().detach()),
                       "train/grad_norm": float(gn),
                       "train/lr": opt.param_groups[0]["lr"],
                       "train/steps_per_sec": rate}, step=step)
            print(f"step {step:6d}  verify={float(l_verify.detach()):.4f}  "
                  f"gen={float(l_gen.detach()):.4f}  {rate:.2f} it/s",
                  flush=True)
        if is_main and step > 0 and step % a.eval_every == 0:
            mtr = selection_metrics(model, data, hold_rows, flat_pos, device)
            wandb.log({f"eval/{k}": v for k, v in mtr.items()}, step=step)
            print(f"  eval @ {step}: {mtr}", flush=True)
            if mtr["sel_f1"] > best_sel:
                best_sel = mtr["sel_f1"]
                save(step, "best.pt")
                print(f"  new best sel_f1={best_sel:.4f} -> best.pt",
                      flush=True)
        if is_main and step > 0 and step % a.ckpt_every == 0:
            save(step)
    if is_main:
        save(a.steps, "final.pt")
        print(f"final metrics: "
              f"{selection_metrics(model, data, hold_rows, flat_pos, device)}",
              flush=True)
        wandb.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
