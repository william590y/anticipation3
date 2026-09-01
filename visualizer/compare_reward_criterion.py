#!/usr/bin/env python
"""Is the PPO-F1 reward the same thing as any column of the results table?

The table reports three criteria (compute_f1.py):
    onset_pitch        pitch exact, onset exact
    onset_pitch_dur    pitch exact, onset exact, duration exact
    onset_pitch_tol1   pitch exact, onset within +-1 bin, DURATION IGNORED

The PPO-F1 reward trains against onset_pitch_tol1 -- the third column. This
scores the same unfiltered rollouts under all three table criteria plus the
reward's own online matcher, to confirm they agree. Any residual gap on the
third column is the matching ORDER, not the criterion: the reward matches in
emission order (required for the increments to telescope) while compute_f1 sorts
predictions by onset first.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from f1_reward import IncrementalF1  # noqa: E402

GROUPS = [
    ("base (ckpt 2500)", "rollouts_valloss"),
    ("GRPO", "rollouts_grpo"),
    ("PPO (token +-1)", "rollouts_ppo"),
]


def load_payload(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text[text.find("{") : text.rfind("}") + 1])


def notes(entries):
    return [
        (int(n["t"]), int(n["d"]), int(n["p"]))
        for n in (entries or [])
        if n and n.get("p") is not None
    ]


def reward_criterion_f1(pred, gt):
    """Exactly the reward's own matcher: online, one-to-one, pitch+dur exact, onset +-1."""
    matcher = IncrementalF1(gt)
    for note in pred:
        matcher.add(note)
    return matcher.f1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--stream", default="raw", help="raw = unfiltered input")
    args = ap.parse_args()

    payload = load_payload(args.data)
    examples = payload["examples"]
    order = list(payload.get("example_order") or examples)

    print(f"{args.stream} stream, {len(order)} windows, macro-averaged over pieces\n")
    print(
        f"  {'model':<18} {'onset+pitch':>12} {'+duration':>11} {'±1 bin':>9} "
        f"| {'REWARD matcher':>17}"
    )
    print(f"  {'':<18} {'(col 1)':>12} {'(col 2)':>11} {'(col 3)':>9} "
          f"| {'should match col 3':>17}")

    for label, group in GROUPS:
        by_piece_table = defaultdict(lambda: defaultdict(list))
        by_piece_reward = defaultdict(list)
        found = 0
        for eid in order:
            block = (examples[eid].get(group) or {}).get(args.stream)
            if not isinstance(block, dict):
                continue
            found += 1
            piece = examples[eid].get("piece") or eid
            for crit, scores in (block.get("f1") or {}).items():
                by_piece_table[piece][crit].append(scores["f1"])
            by_piece_reward[piece].append(
                reward_criterion_f1(
                    notes(block.get("pred_score")), notes(examples[eid].get("gt_score"))
                )
            )
        if not found:
            print(f"  {label:<18} (absent)")
            continue

        def macro(per_piece):
            means = [sum(v) / len(v) for v in per_piece.values() if v]
            return sum(means) / len(means) if means else 0.0

        cols = {}
        for crit in ("onset_pitch", "onset_pitch_dur", "onset_pitch_tol1"):
            cols[crit] = macro(
                {piece: vals[crit] for piece, vals in by_piece_table.items() if crit in vals}
            )
        reward = macro(by_piece_reward)
        print(
            f"  {label:<18} {100 * cols['onset_pitch']:>11.1f}% "
            f"{100 * cols['onset_pitch_dur']:>10.1f}% "
            f"{100 * cols['onset_pitch_tol1']:>8.1f}% "
            f"| {100 * reward:>16.1f}%"
        )

    print(
        "\nThe reward matcher should equal the '±1 bin' column; any small residual "
        "is\nemission-order vs onset-order greedy matching, not a different criterion."
    )


if __name__ == "__main__":
    main()
