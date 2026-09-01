#!/usr/bin/env python
"""Train the N-best reranker with a margin-weighted PAIRWISE ranking loss.

For candidates i, j of the SAME window with F1 gap Delta_ij = F1_i - F1_j:

    L_ij = |Delta_ij| * log(1 + exp(-sgn(Delta_ij) * (q_i - q_j)))

(BRIO/SLiC-style sequence calibration), skipping nearly-tied pairs with
|Delta_ij| < --min_gap. Unlike the pointwise MSE trainer
(`nbest.train_reranker`), there is no reward for estimating window
difficulty -- only within-pool candidate order matters, which is the signal
N-best selection actually uses.

q is the PRE-sigmoid logit of the same ~45M `Reranker`; checkpoints keep the
pointwise format (inner model state_dict + model_cfg), so
`build_reranker_from_ckpt` loads them unchanged and its sigmoid output is a
monotone transform of q (identical argmax selection).

Batching: --windows_per_batch windows per GPU per step, ALL candidates of
each window forwarded together, loss averaged over valid in-window pairs.
Holdout: every 20th window; model selection by holdout argmax-q selected-F1
(`best.pt`).

DDP over 4 GPUs:
  torchrun --standalone --nproc_per_node=4 -m nbest.train_reranker_pairwise \
      --shards 'nbest_data/unf_train_shard*.pt' \
      --run_dir run_nbest_reranker/pairwise_<stamp>
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
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from onpolicy_rollout import score_token_positions
from nbest.reranker import Reranker, RerankerConfig, substitute_candidates

WANDB_PROJECT = "anticipation-asap"


class LogitReranker(nn.Module):
    """Reranker minus the output sigmoid; DDP-wrap THIS so the pairwise
    forward goes through the DDP module (gradient all-reduce hooks)."""

    def __init__(self, inner: Reranker):
        super().__init__()
        self.inner = inner

    def forward(self, tokens: torch.Tensor,
                feats: torch.Tensor | None = None) -> torch.Tensor:
        m = self.inner
        h = m.norm(m.encoder(m.embed_tokens(tokens, feats))).mean(dim=1)
        return m.head(h).squeeze(-1)


def load_shards(pattern):
    paths = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") \
        else [pattern]
    if not paths:
        raise SystemExit(f"no shards match {pattern}")
    win_line, win_tok, c_line, c_tok, c_f1, c_kind = [], [], [], [], [], []
    c_lp = []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        if "cand_tok_logp_ft" in d:
            c_lp.append(torch.stack([d["cand_tok_logp_ft"],
                                     d["cand_tok_logp_base"]], dim=-1))
        win_line.append(d["window_line_idx"])
        win_tok.append(d["window_tokens"])
        c_line.append(d["cand_line_idx"])
        c_tok.append(d["cand_tokens"])
        c_f1.append(d["cand_f1"])
        c_kind.append(d["cand_kind"])
    win_line = torch.cat(win_line)
    win_tok = torch.cat(win_tok)
    c_line = torch.cat(c_line)
    row_of_line = {int(l): i for i, l in enumerate(win_line.tolist())}
    c_win_row = torch.tensor([row_of_line[int(l)] for l in c_line.tolist()],
                             dtype=torch.long)
    cands_of_row = {}
    for i, r in enumerate(c_win_row.tolist()):
        cands_of_row.setdefault(r, []).append(i)
    out_feats = torch.cat(c_lp) if c_lp else None
    if c_lp and len(c_lp) != len(paths):
        raise SystemExit("some shards carry per-token features and some do "
                         "not; regenerate the missing ones")
    return {"cand_feats": out_feats,
            "win_tok": win_tok, "win_line": win_line,
            "cand_tok": torch.cat(c_tok), "cand_f1": torch.cat(c_f1),
            "cand_kind": torch.cat(c_kind), "cand_line": c_line,
            "cand_win_row": c_win_row, "cands_of_row": cands_of_row}


def split_windows(data, holdout_mod=20):
    """Window-level split: every 20th window (by rank among sorted unique
    line indices) held out. Same rule as the pointwise trainer."""
    uniq = sorted(set(data["win_line"].tolist()))
    held_lines = set(uniq[::holdout_mod])
    train_rows, hold_rows = [], []
    for r, l in enumerate(data["win_line"].tolist()):
        (hold_rows if int(l) in held_lines else train_rows).append(r)
    return train_rows, hold_rows


@torch.no_grad()
def selection_metrics(net, data, hold_rows, flat_pos, device, batch=72,
                      max_windows=400, min_gap=0.01):
    """Holdout: argmax-q selected F1 vs greedy/oracle, pairwise acc, rho."""
    net.eval()
    rows = hold_rows[:max_windows]
    cand_idx = [i for r in rows for i in data["cands_of_row"][r]]
    cand_idx_t = torch.tensor(cand_idx, dtype=torch.long)
    preds = torch.empty(len(cand_idx))
    for lo in range(0, len(cand_idx), batch):
        chunk = cand_idx_t[lo:lo + batch]
        wins = data["win_tok"][data["cand_win_row"][chunk]]
        toks = substitute_candidates(wins, data["cand_tok"][chunk],
                                     flat_pos).to(device)
        ft = (data["cand_feats"][chunk].to(device)
              if data.get("cand_feats") is not None else None)
        with _AC():
            preds[lo:lo + len(chunk)] = net(toks, ft).float().cpu()
    pos_of = {ci: k for k, ci in enumerate(cand_idx)}

    sel_f1, greedy_f1, oracle_f1 = [], [], []
    pair_ok = pair_tot = 0
    rhos = []
    for r in rows:
        cs = data["cands_of_row"][r]
        q = np.array([preds[pos_of[i]] for i in cs])
        f1 = data["cand_f1"][torch.tensor(cs)].numpy()
        kind = data["cand_kind"][torch.tensor(cs)].numpy()
        sel_f1.append(float(f1[int(q.argmax())]))
        g = f1[kind == 0]
        greedy_f1.append(float(g[0]) if len(g) else float("nan"))
        oracle_f1.append(float(f1.max()))
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if abs(f1[i] - f1[j]) < min_gap:
                    continue
                pair_tot += 1
                pair_ok += int((q[i] - q[j]) * (f1[i] - f1[j]) > 0)
        if np.unique(f1).size > 1:
            pr = np.argsort(np.argsort(q)).astype(float)
            rr = np.argsort(np.argsort(f1)).astype(float)
            denom = pr.std() * rr.std()
            if denom > 0:
                rhos.append(float(((pr - pr.mean()) * (rr - rr.mean())).mean()
                                  / denom))
    net.train()
    return {"sel_f1": float(np.nanmean(sel_f1)),
            "greedy_f1": float(np.nanmean(greedy_f1)),
            "oracle_f1": float(np.nanmean(oracle_f1)),
            "pairwise_acc": pair_ok / max(pair_tot, 1),
            "spearman": float(np.mean(rhos)) if rhos else float("nan"),
            "n_pairs": pair_tot, "n_windows": len(rows)}


def cosine_lr(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default="nbest_data/unf_train_shard*.pt")
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--windows_per_batch", type=int, default=8, help="per GPU")
    ap.add_argument("--min_gap", type=float, default=0.01,
                    help="skip pairs with |F1_i - F1_j| below this")
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
                    help="train and evaluate in full precision (no bf16 "
                         "autocast). The SHARDS' decode dtype is a separate "
                         "axis -- match them deliberately.")
    a = ap.parse_args()
    global _FP32
    _FP32 = bool(a.fp32)
    print(f"forward dtype: {'fp32' if _FP32 else 'bf16 autocast'}", flush=True)
    if a.smoke:
        a.steps, a.windows_per_batch, a.warmup = 40, 2, 5
        a.dim, a.depth, a.heads = 128, 2, 4
        a.eval_every, a.ckpt_every, a.log_every = 20, 20, 10
        a.wandb_mode = "disabled"
    if not a.run_dir:
        a.run_dir = f"run_nbest_reranker/pairwise_{datetime.now():%m%d_%H%M%S}"

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
    train_rows, hold_rows = split_windows(data)
    flat_pos = score_token_positions(data["win_tok"].shape[1])
    if is_main:
        print(f"{len(data['cand_f1'])} candidates over "
              f"{len(data['win_line'])} windows; train windows "
              f"{len(train_rows)}, holdout {len(hold_rows)}", flush=True)

    n_feat = 0 if data.get("cand_feats") is None \
        else int(data["cand_feats"].shape[-1])
    cfg = RerankerConfig(dim=a.dim, depth=a.depth, heads=a.heads,
                         dim_feedforward=4 * a.dim,
                         seq_len=data["win_tok"].shape[1],
                         token_features=n_feat)
    if is_main:
        print(f"per-token generative features: {n_feat}"
              f"{' (log p_FT, log p_base)' if n_feat == 2 else ''}",
              flush=True)
    inner = Reranker(cfg).to(device)
    model = LogitReranker(inner)
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
                   config={**vars(a), **cfg.to_dict(), "params": n_params,
                           "loss": "pairwise_margin_weighted"})

    def save(step, name=None):
        # Same checkpoint schema as the pointwise trainer: inner Reranker
        # weights + model_cfg, so build_reranker_from_ckpt loads it as-is.
        path = os.path.join(a.run_dir, name or f"ckpt_step{step:07d}.pt")
        torch.save({"model": inner.state_dict(), "model_cfg": cfg.to_dict(),
                    "step": step, "cfg": vars(a)}, path)
        return path

    train_rows_t = torch.tensor(train_rows, dtype=torch.long)
    best_sel = -1.0
    model.train()
    t_last, s_last = time.time(), 0
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, a.lr, a.warmup, a.steps)
        wrows = train_rows_t[torch.randint(0, len(train_rows_t),
                                           (a.windows_per_batch,),
                                           generator=gen)].tolist()
        cand_idx = [i for r in wrows for i in data["cands_of_row"][r]]
        bounds = np.cumsum([0] + [len(data["cands_of_row"][r])
                                  for r in wrows])
        chunk = torch.tensor(cand_idx, dtype=torch.long)
        wins = data["win_tok"][data["cand_win_row"][chunk]]
        toks = substitute_candidates(wins, data["cand_tok"][chunk],
                                     flat_pos).to(device)
        f1 = data["cand_f1"][chunk].to(device)
        feats = (data["cand_feats"][chunk].to(device)
                 if data.get("cand_feats") is not None else None)
        with _AC():
            q = net(toks, feats).float()
            terms, weights = [], []
            for w in range(len(wrows)):
                lo, hi = int(bounds[w]), int(bounds[w + 1])
                qw, fw = q[lo:hi], f1[lo:hi]
                delta = fw.unsqueeze(1) - fw.unsqueeze(0)   # (k, k)
                dq = qw.unsqueeze(1) - qw.unsqueeze(0)
                mask = torch.triu(torch.ones_like(delta, dtype=torch.bool),
                                  diagonal=1) & (delta.abs() >= a.min_gap)
                if mask.any():
                    terms.append(
                        (delta.abs() * F.softplus(-torch.sign(delta) * dq))[mask])
                    weights.append(delta.abs()[mask])
            if not terms:
                continue
            flat = torch.cat(terms)
            loss = flat.mean()
            # The raw loss scale tracks the batch's |Delta| mass (hard-window
            # batches carry several-fold more weight), so the raw curve
            # oscillates by construction. The |Delta|-weighted mean softplus
            # is batch-comparable: log 2 at init, -> 0 as ranking improves.
            with torch.no_grad():
                norm_loss = float(flat.sum() / torch.cat(weights).sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()

        if is_main and step % a.log_every == 0:
            now = time.time()
            rate = (step - s_last) / max(now - t_last, 1e-6)
            t_last, s_last = now, step
            wandb.log({"train/pair_loss": float(loss.detach()),
                       "train/norm_pair_loss": norm_loss,
                       "train/grad_norm": float(gn),
                       "train/lr": opt.param_groups[0]["lr"],
                       "train/steps_per_sec": rate}, step=step)
            print(f"step {step:6d}  pair_loss={float(loss.detach()):.5f}  "
                  f"{rate:.1f} it/s", flush=True)
        if is_main and step > 0 and step % a.eval_every == 0:
            m = selection_metrics(model, data, hold_rows, flat_pos, device,
                                  min_gap=a.min_gap)
            wandb.log({f"eval/{k}": v for k, v in m.items()}, step=step)
            print(f"  eval @ {step}: {m}", flush=True)
            if m["sel_f1"] > best_sel:
                best_sel = m["sel_f1"]
                save(step, "best.pt")
                print(f"  new best sel_f1={best_sel:.4f} -> best.pt",
                      flush=True)
        if is_main and step > 0 and step % a.ckpt_every == 0:
            save(step)

    if world > 1:
        dist.barrier()
    if is_main:
        m = selection_metrics(model, data, hold_rows, flat_pos, device,
                              min_gap=a.min_gap)
        wandb.log({f"eval/{k}": v for k, v in m.items()}, step=a.steps)
        if m["sel_f1"] > best_sel:
            save(a.steps, "best.pt")
        final = save(a.steps, "final.pt")
        print(f"final metrics: {m}\nfinal -> {final}", flush=True)
        wandb.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
