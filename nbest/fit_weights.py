#!/usr/bin/env python
"""Fit the decode-objective weights (alpha, beta, gamma) on VAL N-best shards.

Selection rule (the interface the viz beam-search inference consumes):

    score(y) = alpha * z(logp_ft) + beta * z(logp_base) + gamma * z(q_phi)
    y_hat    = argmax over the window's candidates

where z(.) is per-feature standardisation with the mean/std saved in the
output json (computed over all val candidates). Weights are found by a coarse
grid over each coefficient followed by coordinate refinement, maximising the
mean F1 of the selected candidate per window. Baselines reported: random
candidate, greedy candidate, each single feature alone, alpha+beta (no
reranker), the full objective, and the per-window oracle.

Output schema (nbest_data/decode_weights.json):
{
  "alpha": float, "beta": float, "gamma": float,
  "feature_stats": {"logp_ft": {"mean","std"}, "logp_base": {...},
                    "q_phi": {...}},
  "reranker_checkpoint": str, "ft_checkpoint": str,
  "base_model": "stanford-crfm/music-medium-800k",
  "metrics": {"selected_mean_f1", "oracle_mean_f1", "greedy_mean_f1",
              "random_mean_f1", "alpha_only_mean_f1", "beta_only_mean_f1",
              "gamma_only_mean_f1", "alpha_beta_mean_f1", "n_windows"},
  "rule": "argmax_y alpha*z(logp_ft)+beta*z(logp_base)+gamma*z(q_phi)"
}
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from onpolicy_rollout import score_token_positions
from nbest.reranker import build_reranker_from_ckpt, substitute_candidates


def load_val(pattern):
    paths = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") \
        else [pattern]
    if not paths:
        raise SystemExit(f"no shards match {pattern}")
    out = {}
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        for k in ("window_line_idx", "window_tokens", "cand_line_idx",
                  "cand_tokens", "cand_logp_ft", "cand_logp_base", "cand_f1",
                  "cand_kind"):
            out.setdefault(k, []).append(d[k])
        out.setdefault("checkpoint", d["checkpoint"])
    for k in list(out):
        if isinstance(out[k], list):
            out[k] = torch.cat(out[k])
    return out


@torch.no_grad()
def compute_q(data, ckpt, device, batch=64):
    model = build_reranker_from_ckpt(ckpt, device)
    row_of_line = {int(l): i for i, l in
                   enumerate(data["window_line_idx"].tolist())}
    win_rows = torch.tensor([row_of_line[int(l)] for l in
                             data["cand_line_idx"].tolist()])
    flat_pos = score_token_positions(data["window_tokens"].shape[1])
    q = torch.empty(len(win_rows))
    for lo in range(0, len(win_rows), batch):
        hi = min(lo + batch, len(win_rows))
        wins = data["window_tokens"][win_rows[lo:hi]]
        toks = substitute_candidates(wins, data["cand_tokens"][lo:hi],
                                     flat_pos).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            q[lo:hi] = model(toks).float().cpu()
    return q


def group_by_window(data, q):
    lines = data["cand_line_idx"].numpy()
    order = np.argsort(lines, kind="stable")
    feats = np.stack([data["cand_logp_ft"].numpy(),
                      data["cand_logp_base"].numpy(),
                      q.numpy()], axis=1)[order]
    f1 = data["cand_f1"].numpy()[order]
    kind = data["cand_kind"].numpy()[order]
    lines = lines[order]
    uniq, starts = np.unique(lines, return_index=True)
    groups = np.split(np.arange(len(lines)), starts[1:])
    counts = {len(g) for g in groups}
    if len(counts) == 1:
        c = counts.pop()
        idx = np.stack(groups)
        return feats[idx], f1[idx], kind[idx]
    raise SystemExit(f"non-uniform candidate counts per window: {counts}")


def mean_f1_for(weights, zfeats, f1):
    scores = (zfeats * np.asarray(weights)[None, None, :]).sum(-1)
    pick = scores.argmax(axis=1)
    return float(f1[np.arange(len(f1)), pick].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-shards", default="nbest_data/val_shard*.pt")
    ap.add_argument("--reranker-ckpt", required=True)
    ap.add_argument("--output", default="nbest_data/decode_weights.json")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_val(a.val_shards)
    q = compute_q(data, a.reranker_ckpt, device, a.batch)
    feats, f1, kind = group_by_window(data, q)   # (W, C, 3), (W, C), (W, C)
    W, C, _ = feats.shape
    mean = feats.reshape(-1, 3).mean(0)
    std = feats.reshape(-1, 3).std(0) + 1e-8
    z = (feats - mean) / std

    rng = np.random.default_rng(0)
    metrics = {
        "n_windows": W,
        "oracle_mean_f1": float(f1.max(axis=1).mean()),
        "random_mean_f1": float(
            f1[np.arange(W), rng.integers(0, C, W)].mean()),
        "greedy_mean_f1": float(f1[kind == 0].mean()),
        "alpha_only_mean_f1": mean_f1_for((1, 0, 0), z, f1),
        "beta_only_mean_f1": mean_f1_for((0, 1, 0), z, f1),
        "gamma_only_mean_f1": mean_f1_for((0, 0, 1), z, f1),
    }
    grid = (-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0)
    best_ab, best_ab_f1 = (1, 0, 0), -1
    for al, be in itertools.product(grid, grid):
        v = mean_f1_for((al, be, 0.0), z, f1)
        if v > best_ab_f1:
            best_ab, best_ab_f1 = (al, be, 0.0), v
    metrics["alpha_beta_mean_f1"] = best_ab_f1
    best, best_f1 = best_ab, best_ab_f1
    for al, be, ga in itertools.product(grid, grid, grid):
        v = mean_f1_for((al, be, ga), z, f1)
        if v > best_f1:
            best, best_f1 = (al, be, ga), v
    for _ in range(3):  # coordinate refinement
        for axis in range(3):
            for delta in (-0.3, -0.1, -0.03, 0.03, 0.1, 0.3):
                cand = list(best)
                cand[axis] += delta
                v = mean_f1_for(cand, z, f1)
                if v > best_f1:
                    best, best_f1 = tuple(cand), v
    metrics["selected_mean_f1"] = best_f1

    out = {
        "alpha": best[0], "beta": best[1], "gamma": best[2],
        "feature_stats": {
            "logp_ft": {"mean": float(mean[0]), "std": float(std[0])},
            "logp_base": {"mean": float(mean[1]), "std": float(std[1])},
            "q_phi": {"mean": float(mean[2]), "std": float(std[2])}},
        "reranker_checkpoint": str(a.reranker_ckpt),
        "ft_checkpoint": data["checkpoint"],
        "base_model": "stanford-crfm/music-medium-800k",
        "metrics": metrics,
        "rule": "argmax_y alpha*z(logp_ft)+beta*z(logp_base)+gamma*z(q_phi)",
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
