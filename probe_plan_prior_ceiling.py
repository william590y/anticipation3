#!/usr/bin/env python
"""How well could ANY unconditional prior predict the plan?

Stage 2 at ``placement=front`` has no context before the plan, so the best it can
possibly do on code position j is the marginal mode, ``max_c p(c_j)``. This runs
the frozen stage-1 encoder over validation windows, builds the empirical code
distribution per position, and reports that ceiling next to what the LM actually
achieves -- turning "the plan looks unpredictable" into a number.

It also reports the conditional ceiling that a *performance-conditioned* prior
would be aiming at, via the entropy of the code distribution.
"""
from __future__ import annotations

import argparse
from collections import Counter

import torch

from packed_dataset import TokenizedDataset
from pianobart_encoder import load_pianobart, score_features
from plan_vq import load_plan_vq, split_packed_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vq_checkpoint", required=True)
    parser.add_argument("--val_file", default="data/val_paper.txt")
    parser.add_argument("--windows", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plan_vq = load_plan_vq(args.vq_checkpoint, device=device)
    plan_vq.eval()
    pianobart, meta = load_pianobart(device=device)

    dataset = TokenizedDataset(args.val_file, is_training=False)
    total = min(args.windows, len(dataset))
    print(f"scoring {total} validation windows from {args.val_file}\n")

    per_position = None
    with torch.no_grad():
        for start in range(0, total, args.batch_size):
            stop = min(start + args.batch_size, total)
            windows = torch.stack(
                [dataset[i]["input_ids"] for i in range(start, stop)]
            ).to(device)
            perf, score, valid = split_packed_batch(windows)
            features = score_features(pianobart, meta, score, valid)
            codes = plan_vq.encode_codes(perf, features, valid).cpu()
            if per_position is None:
                per_position = [Counter() for _ in range(codes.shape[1])]
            for j in range(codes.shape[1]):
                per_position[j].update(codes[:, j].tolist())

    print(f"{'pos':>4}  {'distinct':>8}  {'mode share':>11}  {'entropy(bits)':>13}")
    ceilings = []
    for j, counter in enumerate(per_position):
        n = sum(counter.values())
        probabilities = [c / n for c in counter.values()]
        mode = max(probabilities)
        entropy = -sum(p * torch.log2(torch.tensor(p)).item() for p in probabilities)
        ceilings.append(mode)
        print(f"{j:>4}  {len(counter):>8}  {100 * mode:>10.2f}%  {entropy:>13.2f}")

    mean_ceiling = sum(ceilings) / len(ceilings)
    print(
        f"\nBest possible accuracy for an UNCONDITIONAL prior (mean over positions): "
        f"{100 * mean_ceiling:.2f}%"
    )
    print(
        "That is the hard ceiling for placement=front, where the LM has no context\n"
        "before the plan. Compare it against the run's val/plan_code_accuracy."
    )


if __name__ == "__main__":
    main()
