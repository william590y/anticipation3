#!/usr/bin/env python
"""Train the N-best reranker q_phi(x, y) ~ F1 by MSE regression.

Data: shard .pt files from `nbest.generate_nbest`. 5% of TRAIN WINDOWS
(by window line index hash, never by candidate -- candidates of one window
must not straddle the split) are held out for ranking metrics: pairwise
candidate-order accuracy and mean per-window Spearman rho between q_phi and
true F1 (candidate pairs/windows with tied F1 are skipped).

DDP over 2 GPUs:
  torchrun --standalone --nproc_per_node=2 -m nbest.train_reranker \
      --shards 'nbest_data/train_shard*.pt' --run_dir run_nbest_reranker/<stamp>
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import contextlib

import torch

# See train_reranker_pairwise.py: dtype is a measured confound here, so it is
# a recorded flag rather than a hardcoded choice. Default unchanged.
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

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from onpolicy_rollout import score_token_positions
from nbest.reranker import Reranker, RerankerConfig, substitute_candidates

WANDB_PROJECT = "anticipation-asap"


def load_shards(pattern):
    paths = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") \
        else [pattern]
    if not paths:
        raise SystemExit(f"no shards match {pattern}")
    win_line, win_tok, c_line, c_tok, c_f1 = [], [], [], [], []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        win_line.append(d["window_line_idx"])
        win_tok.append(d["window_tokens"])
        c_line.append(d["cand_line_idx"])
        c_tok.append(d["cand_tokens"])
        c_f1.append(d["cand_f1"])
    win_line = torch.cat(win_line)
    win_tok = torch.cat(win_tok)
    c_line = torch.cat(c_line)
    row_of_line = {int(l): i for i, l in enumerate(win_line.tolist())}
    c_win_row = torch.tensor([row_of_line[int(l)] for l in c_line.tolist()],
                             dtype=torch.long)
    return {"win_tok": win_tok, "win_line": win_line,
            "cand_tok": torch.cat(c_tok), "cand_f1": torch.cat(c_f1),
            "cand_line": c_line, "cand_win_row": c_win_row}


def split_holdout(data, holdout_mod=20):
    """Hold out every 20th WINDOW (by rank among sorted unique line indices).

    Never split by raw ``line % k`` -- stride-sampled line indices are all
    multiples of the stride, which aliases with k and degenerates the split.
    """
    uniq = sorted(set(data["cand_line"].tolist()))
    held_lines = set(uniq[::holdout_mod])
    hold = torch.tensor([int(l) in held_lines
                         for l in data["cand_line"].tolist()])
    return (~hold).nonzero().flatten(), hold.nonzero().flatten()


def batch_inputs(data, idx, flat_pos):
    wins = data["win_tok"][data["cand_win_row"][idx]]
    return substitute_candidates(wins, data["cand_tok"][idx], flat_pos)


@torch.no_grad()
def ranking_metrics(model, data, hold_idx, flat_pos, device, batch=64,
                    max_pairs=6000):
    model.eval()
    idx = hold_idx[:max_pairs]
    preds = torch.empty(len(idx))
    for lo in range(0, len(idx), batch):
        chunk = idx[lo:lo + batch]
        toks = batch_inputs(data, chunk, flat_pos).to(device)
        with _AC():
            preds[lo:lo + len(chunk)] = model(toks).float().cpu()
    f1 = data["cand_f1"][idx]
    lines = data["cand_line"][idx]
    mse = float(((preds - f1) ** 2).mean())

    pair_ok = pair_tot = 0
    rhos = []
    for line in lines.unique().tolist():
        sel = (lines == line).nonzero().flatten()
        if len(sel) < 2:
            continue
        p, r = preds[sel].numpy(), f1[sel].numpy()
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                if r[i] == r[j]:
                    continue
                pair_tot += 1
                pair_ok += int((p[i] - p[j]) * (r[i] - r[j]) > 0)
        if np.unique(r).size > 1:
            pr = np.argsort(np.argsort(p)).astype(float)
            rr = np.argsort(np.argsort(r)).astype(float)
            denom = pr.std() * rr.std()
            if denom > 0:
                rhos.append(float(((pr - pr.mean()) * (rr - rr.mean())).mean()
                                  / denom))
    model.train()
    return {"holdout_mse": mse,
            "pairwise_acc": pair_ok / max(pair_tot, 1),
            "spearman": float(np.mean(rhos)) if rhos else float("nan"),
            "n_pairs": pair_tot, "n_windows": len(rhos)}


def cosine_lr(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default="nbest_data/train_shard*.pt")
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=64, help="per GPU")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb_mode", default="online")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fp32", action="store_true",
                    help="train/eval in full precision (no bf16 autocast)")
    a = ap.parse_args()
    global _FP32
    _FP32 = bool(a.fp32)
    print(f"forward dtype: {'fp32' if _FP32 else 'bf16 autocast'}", flush=True)
    if a.smoke:
        a.steps, a.batch_size, a.warmup = 40, 8, 5
        a.dim, a.depth, a.heads = 128, 2, 4
        a.eval_every, a.ckpt_every, a.log_every = 20, 20, 10
        a.wandb_mode = "disabled"
    if not a.run_dir:
        a.run_dir = f"run_nbest_reranker/{datetime.now():%m%d_%H%M%S}"

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
    is_main = rank == 0
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed + rank)
    gen = torch.Generator().manual_seed(a.seed + 1000 * rank)

    data = load_shards(a.shards)
    train_idx, hold_idx = split_holdout(data)
    flat_pos = score_token_positions(data["win_tok"].shape[1])
    if is_main:
        print(f"{len(data['cand_f1'])} candidates over "
              f"{len(data['win_line'])} windows; train pairs "
              f"{len(train_idx)}, holdout {len(hold_idx)}", flush=True)

    cfg = RerankerConfig(dim=a.dim, depth=a.depth, heads=a.heads,
                         dim_feedforward=4 * a.dim,
                         seq_len=data["win_tok"].shape[1])
    model = Reranker(cfg).to(device)
    net = DDP(model, device_ids=[local], output_device=local) \
        if world > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=a.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        os.makedirs(a.run_dir, exist_ok=True)
        print(f"reranker params: {n_params/1e6:.1f}M", flush=True)
        wandb.init(project=WANDB_PROJECT, mode=a.wandb_mode,
                   name=f"nbest_reranker_{Path(a.run_dir).name}",
                   config={**vars(a), **cfg.to_dict(), "params": n_params})

    def save(step, name=None):
        path = os.path.join(a.run_dir, name or f"ckpt_step{step:07d}.pt")
        torch.save({"model": model.state_dict(), "model_cfg": cfg.to_dict(),
                    "step": step, "cfg": vars(a)}, path)
        return path

    model.train()
    t_last, s_last = time.time(), 0
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, a.lr, a.warmup, a.steps)
        pick = train_idx[torch.randint(0, len(train_idx), (a.batch_size,),
                                       generator=gen)]
        toks = batch_inputs(data, pick, flat_pos).to(device)
        target = data["cand_f1"][pick].to(device)
        with _AC():
            pred = net(toks)
            loss = F.mse_loss(pred.float(), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()

        if is_main and step % a.log_every == 0:
            now = time.time()
            rate = (step - s_last) / max(now - t_last, 1e-6)
            t_last, s_last = now, step
            wandb.log({"train/mse": float(loss.detach()),
                       "train/grad_norm": float(gn),
                       "train/lr": opt.param_groups[0]["lr"],
                       "train/steps_per_sec": rate}, step=step)
            print(f"step {step:6d}  mse={float(loss.detach()):.5f}  "
                  f"{rate:.1f} it/s", flush=True)
        if is_main and step > 0 and step % a.eval_every == 0:
            m = ranking_metrics(model, data, hold_idx, flat_pos, device,
                                batch=a.batch_size)
            wandb.log({f"eval/{k}": v for k, v in m.items()}, step=step)
            print(f"  eval @ {step}: {m}", flush=True)
        if is_main and step > 0 and step % a.ckpt_every == 0:
            save(step)

    if world > 1:
        dist.barrier()
    if is_main:
        m = ranking_metrics(model, data, hold_idx, flat_pos, device,
                            batch=a.batch_size)
        wandb.log({f"eval/{k}": v for k, v in m.items()}, step=a.steps)
        final = save(a.steps, "final.pt")
        print(f"final metrics: {m}\nfinal -> {final}", flush=True)
        wandb.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
