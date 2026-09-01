#!/usr/bin/env python
"""Score each N-best window's GROUND-TRUTH score under both models.

For the GT-in-pool experiment: add the GT itself to every candidate pool and
ask (1) how often the base model ranks the GT above every generated candidate,
P(B_GT > max_j B_j), and (2) how B and F correlate within pools. This script
computes the missing quantities B_GT and F_GT per window, with the exact same
conventions as the stored candidate features (nbest/generate_nbest.py):

  * B_GT = ``logp_base_batch`` on the window's 414 GT score tokens (so_c:
    AUTOREGRESS-primed flat layout, constrained, summed log-prob);
  * F_GT = teacher-forced constrained FT log-prob of the GT score tokens in
    the interleaved packed window at temperature 1.0 (``score_token_logprob``,
    same as the greedy candidate's rescoring).

Shardable over GPUs; outputs a .pt with window_line_idx, gt_logp_base,
gt_logp_ft aligned to the shard's window order slice.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval_base_score_ppl import load_base_model, slot_logit_masks  # noqa: E402
from evaluate_muster import load_model  # noqa: E402
from nbest.generate_nbest import logp_base_batch  # noqa: E402
from onpolicy_rollout import score_token_logprob, score_token_positions  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-file", default="nbest_data/unf_val_shard00.pt")
    ap.add_argument("--checkpoint", default="run_paper_split_v2/checkpoint-2500")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(a.shard_file, map_location="cpu", weights_only=False)
    windows = data["window_tokens"][a.shard_index::a.num_shards].long()
    line_idx = data["window_line_idx"][a.shard_index::a.num_shards]
    n = windows.shape[0]
    print(f"{a.shard_file}: slice {a.shard_index}/{a.num_shards} -> {n} windows")

    ft, _ = load_model(a.checkpoint)
    ft = ft.to(device).eval()
    base = load_base_model(device)
    masks = slot_logit_masks(device)
    positions = score_token_positions(windows.shape[1], device=device)

    autocast = lambda: torch.autocast(  # noqa: E731
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available())

    gt_lp_ft = torch.empty(n)
    gt_lp_base = torch.empty(n)
    t0 = time.time()
    for lo in range(0, n, a.batch):
        hi = min(lo + a.batch, n)
        w = windows[lo:hi].to(device)
        with autocast():
            logits = ft(w).logits
        lp = score_token_logprob(logits, w, positions,
                                 temperature=1.0, constrain=True)
        gt_lp_ft[lo:hi] = lp.sum(dim=1).float().cpu()
        del logits
        gt_flat = w[:, positions]
        gt_lp_base[lo:hi] = logp_base_batch(base, masks, gt_flat,
                                            a.batch).float().cpu()
        if (lo // a.batch) % 10 == 0:
            rate = hi / max(time.time() - t0, 1e-6)
            print(f"  {hi}/{n}  {rate:.1f} win/s", flush=True)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"shard_file": a.shard_file, "checkpoint": a.checkpoint,
                "shard_index": a.shard_index, "num_shards": a.num_shards,
                "window_line_idx": line_idx,
                "gt_logp_ft": gt_lp_ft, "gt_logp_base": gt_lp_base}, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
