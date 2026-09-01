#!/usr/bin/env python
"""Train the ListT5-style listwise Fusion-in-Decoder reranker.

Each step samples `m` candidates from a window, feeds them as ONE list (m
encoder passes fused into a single decoder memory), and trains the decoder to
emit the full permutation worst-first with token-level cross-entropy -- the
paper's supervision, which beat emitting only the winner by 3 NDCG.

Holdout metric is the m-ary TOURNAMENT SORT champion over all 33 candidates,
i.e. the quantity the method is for, alongside greedy and the pool oracle.

  torchrun --standalone --nproc_per_node=1 -m nbest.train_listt5 \
      --shards 'nbest_data/unf32_train_shard*.pt' \
      --run_dir run_nbest_reranker/listt5_<stamp>
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nbest.listt5 import ListwiseFiD, ListwiseFiDConfig, tournament_sort
from nbest.reranker import substitute_candidates
from nbest.train_reranker_pairwise import (cosine_lr, load_shards,
                                           split_windows)
from onpolicy_rollout import score_token_positions

WANDB_PROJECT = "anticipation-asap"


def build_group(data, rows, flat_pos, device, m, feats_wanted):
    """(B, m, L) tokens for one candidate group per row + (B, m) worst->best
    target. Candidate slots are SHUFFLED before identifiers are assigned, so
    no identifier correlates with quality (ListT5 does the same)."""
    toks, targets, feats = [], [], []
    for cs in rows:
        cs = list(cs)
        f1 = data["cand_f1"][torch.tensor(cs)]
        idx = torch.tensor(cs)
        wins = data["win_tok"][data["cand_win_row"][idx]]
        t = substitute_candidates(wins, data["cand_tok"][idx], flat_pos)
        if len(cs) < m:                       # pad by repeating slot 0
            pad = m - len(cs)
            t = torch.cat([t, t[:1].expand(pad, -1)], dim=0)
            f1 = torch.cat([f1, f1[:1].expand(pad)])
        toks.append(t[:m])
        targets.append(torch.argsort(f1[:m]))   # ascending F1 = worst first
        if feats_wanted:
            fe = data["cand_feats"][idx]
            if len(cs) < m:
                fe = torch.cat([fe, fe[:1].expand(m - len(cs), -1, -1)], 0)
            feats.append(fe[:m])
    out_f = torch.stack(feats).to(device) if feats_wanted else None
    return (torch.stack(toks).to(device), torch.stack(targets).to(device),
            out_f)


@torch.no_grad()
def selection_metrics(model, data, hold_rows, flat_pos, device, m,
                      max_windows=150, use_feats=False):
    model.eval()
    sel, greedy, oracle, passes = [], [], [], []
    for r in hold_rows[:max_windows]:
        cs = data["cands_of_row"][r]
        f1 = data["cand_f1"][torch.tensor(cs)].numpy()
        kind = data["cand_kind"][torch.tensor(cs)].numpy()

        def pick_best(group, _cs=cs):
            toks, _, fe = build_group(data, [[_cs[g] for g in group]],
                                      flat_pos, device, m, use_feats)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                perm = model.generate(toks[:, :len(group)], fe)
            return group[int(perm[0, -1])]     # best is emitted LAST

        champ, n_pass = tournament_sort(len(cs), m, pick_best)
        sel.append(float(f1[champ]))
        g = f1[kind == 0]
        greedy.append(float(g[0]) if len(g) else float("nan"))
        oracle.append(float(f1.max()))
        passes.append(n_pass)
    model.train()
    return {"sel_f1": float(np.nanmean(sel)),
            "greedy_f1": float(np.nanmean(greedy)),
            "oracle_f1": float(np.nanmean(oracle)),
            "passes_per_window": float(np.mean(passes)),
            "n_windows": len(sel)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--windows_per_batch", type=int, default=4)
    ap.add_argument("--m", type=int, default=5, help="list size per pass")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dec_depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb_mode", default="online")
    a = ap.parse_args()

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
    use_feats = data.get("cand_feats") is not None
    n_feat = int(data["cand_feats"].shape[-1]) if use_feats else 0
    flat_pos = score_token_positions(data["win_tok"].shape[1])
    if is_main:
        print(f"{len(data['win_line'])} windows; train {len(train_rows)}, "
              f"holdout {len(hold_rows)}; m={a.m}; token_features={n_feat}",
              flush=True)

    cfg = ListwiseFiDConfig(dim=a.dim, depth=a.depth, heads=a.heads,
                            dim_feedforward=4 * a.dim, m=a.m,
                            dec_depth=a.dec_depth,
                            seq_len=data["win_tok"].shape[1],
                            token_features=n_feat)
    model = ListwiseFiD(cfg).to(device)
    net = DDP(model, device_ids=[local], output_device=local) \
        if world > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=a.weight_decay)
    if is_main:
        os.makedirs(a.run_dir, exist_ok=True)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"ListT5-style FiD params: {n_params/1e6:.1f}M", flush=True)
        wandb.init(project=WANDB_PROJECT, mode=a.wandb_mode,
                   name=f"nbest_{Path(a.run_dir).name}",
                   config={**vars(a), **cfg.to_dict(), "params": n_params,
                           "loss": "listwise_permutation_ce"})

    def save(step, name=None):
        path = os.path.join(a.run_dir, name or f"ckpt_step{step:07d}.pt")
        torch.save({"model": model.state_dict(), "model_cfg": cfg.to_dict(),
                    "step": step, "cfg": vars(a)}, path)
        return path

    rows_t = torch.tensor(train_rows, dtype=torch.long)
    best_sel, t_last, s_last = -1.0, time.time(), 0
    model.train()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, a.lr, a.warmup, a.steps)
        wrows = rows_t[torch.randint(0, len(rows_t), (a.windows_per_batch,),
                                     generator=gen)].tolist()
        groups = []
        for r in wrows:
            cs = data["cands_of_row"][r]
            pick = torch.randperm(len(cs), generator=gen)[:a.m].tolist()
            groups.append([cs[i] for i in pick])
        toks, target, feats = build_group(data, groups, flat_pos, device,
                                          a.m, use_feats)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            logits = net(toks, target, feats)
            loss = F.cross_entropy(logits.reshape(-1, a.m).float(),
                                   target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()

        if is_main and step % a.log_every == 0:
            now = time.time()
            rate = (step - s_last) / max(now - t_last, 1e-6)
            t_last, s_last = now, step
            wandb.log({"train/perm_ce": float(loss.detach()),
                       "train/grad_norm": float(gn),
                       "train/lr": opt.param_groups[0]["lr"],
                       "train/steps_per_sec": rate}, step=step)
            print(f"step {step:6d}  perm_ce={float(loss.detach()):.4f}  "
                  f"{rate:.1f} it/s", flush=True)
        if is_main and step > 0 and step % a.eval_every == 0:
            mtr = selection_metrics(model, data, hold_rows, flat_pos, device,
                                    a.m, use_feats=use_feats)
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
        mtr = selection_metrics(model, data, hold_rows, flat_pos, device, a.m,
                                use_feats=use_feats)
        print(f"final metrics: {mtr}", flush=True)
        wandb.log({f"eval/{k}": v for k, v in mtr.items()}, step=a.steps)
        wandb.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
