#!/usr/bin/env python
"""Does re-enabling dropout break PPO's importance ratio?

PPO's clipped surrogate assumes the ratio pi_theta/pi_old is 1 at the sampling
parameters, so the first inner epoch is an unbiased policy-gradient step. That
holds only because the post-training runs keep the model in eval() -- with
dropout on in the update pass but the behaviour policy scored without it, the
ratio is exp(log pi_dropout - log pi_clean), which is pure noise.

If that noise routinely lands outside [1-clip_eps, 1+clip_eps], the clipped
surrogate flattens and the gradient dies on most tokens: PPO would look like it
is running while learning nothing. This measures it directly on real unfiltered
windows before committing GPUs to the full run.

Reports, per placement:
  |log ratio|   the dropout-induced perturbation per score token
  clip fraction the share of tokens outside the clip band at eps
  ratio spread  quantiles of the ratio itself
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from evaluate_muster import load_model  # noqa: E402
from onpolicy_rollout import (  # noqa: E402
    rollout_score_slots,
    score_token_positions,
    score_token_logprob,
)


def read_windows(path, count):
    windows = []
    with open(path) as stream:
        for line in stream:
            windows.append([int(tok) for tok in line.split("|", 1)[0].split()])
            if len(windows) >= count:
                break
    return torch.tensor(windows, dtype=torch.long)


def logprob_pass(model, rolled, positions, temperature, autocast_ctx):
    with torch.no_grad(), autocast_ctx():
        logits = model(input_ids=rolled, use_cache=False).logits
    return score_token_logprob(
        logits, rolled, positions, temperature=temperature, constrain=True
    )


def summarise(name, delta, valid, clip_eps):
    flat = delta[valid.bool()].abs().float()
    ratio = delta[valid.bool()].float().exp()
    outside = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean()
    quantiles = torch.tensor([0.5, 0.9, 0.99]).to(flat.device)
    q = torch.quantile(flat, quantiles)
    print(
        f"  {name:<34} mean |log r| {flat.mean():.4f}  "
        f"p50 {q[0]:.4f}  p90 {q[1]:.4f}  p99 {q[2]:.4f}  max {flat.max():.3f}  "
        f"outside +-{clip_eps:.0%}: {100 * outside:.1f}%"
    )
    return float(outside)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="run_paper_split_v2/checkpoint-2500")
    ap.add_argument("--data", default="data/val_paper_unfiltered.txt")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.checkpoint, config_source=None)
    model.to(device)

    def autocast_ctx():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    input_ids = read_windows(args.data, args.windows).to(device)
    print(f"{input_ids.shape[0]} unfiltered windows from {args.data}")

    # Behaviour policy: eval mode, exactly as the rollout is collected today.
    model.eval()
    rollout = rollout_score_slots(
        model, input_ids, targets=input_ids, temperature=args.temperature,
        constrain=True, collect_logprobs=True, collect_gt_ce=False,
        autocast_ctx=autocast_ctx,
    )
    rolled = rollout["rolled"]
    positions = rollout["positions"]
    valid = rollout["valid"]

    clean = logprob_pass(model, rolled, positions, args.temperature, autocast_ctx)

    print("\nBaseline: does a second clean forward reproduce the first?")
    clean2 = logprob_pass(model, rolled, positions, args.temperature, autocast_ctx)
    summarise("eval vs eval (numerical floor)", clean2 - clean, valid, args.clip_eps)

    print("\nDropout on in the update pass (behaviour scored clean):")
    fractions = []
    model.train()
    for repeat in range(args.repeats):
        noisy = logprob_pass(model, rolled, positions, args.temperature, autocast_ctx)
        fractions.append(
            summarise(f"train() vs eval()  [draw {repeat + 1}]",
                      noisy - clean, valid, args.clip_eps)
        )

    print("\nDropout on in BOTH passes (independent masks):")
    noisy_reference = logprob_pass(model, rolled, positions, args.temperature, autocast_ctx)
    for repeat in range(args.repeats):
        noisy = logprob_pass(model, rolled, positions, args.temperature, autocast_ctx)
        summarise(f"train() vs train() [draw {repeat + 1}]",
                  noisy - noisy_reference, valid, args.clip_eps)

    model.eval()
    print(
        f"\nVERDICT: with dropout in the update pass, "
        f"{100 * statistics.mean(fractions):.1f}% of score tokens fall outside the "
        f"clip band at eps={args.clip_eps} before any parameter has moved."
    )
    if statistics.mean(fractions) > 0.25:
        print(
            "  -> The surrogate is clipped on most tokens by noise alone; the "
            "first-epoch gradient would be largely destroyed."
        )
    else:
        print("  -> Ratio noise is tolerable; dropout can stay on in the update pass.")


if __name__ == "__main__":
    main()
