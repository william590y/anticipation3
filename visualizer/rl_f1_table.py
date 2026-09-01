#!/usr/bin/env python
"""Macro-average note-level F1 per checkpoint, read from data.js inline metrics.

The mean-F1 table protocol: unfiltered (``raw``) performance input, no
ground-truth seeding, greedy AR rollout, macro-averaged over pieces so every
piece carries equal weight regardless of how many windows it contributed.

Every rollout block in data.js already carries its own ``f1`` dict (written by
the precompute chain), so this reads them rather than re-rolling the models --
which is also why the numbers reproduce ckpt_f1_compare.json exactly. Pass
``--verify`` to assert that.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

VARIANTS = ("onset_pitch", "onset_pitch_dur", "onset_pitch_tol1")

# Display order is the table's row order. `compare_key` names the matching entry
# in ckpt_f1_compare.json, where one exists, for the --verify cross-check.
ROWS = [
    {
        "key": "base_loss",
        "group": "rollouts_valloss",
        "model": "Ours (FT)",
        "label": "best val loss",
        "compare_key": "base_loss",
    },
    {
        "key": "base_pitch",
        "group": "rollouts",
        "model": "Ours (FT)",
        "label": "best AR pitch",
        "compare_key": "base_pitch",
    },
    {
        "key": "lora_loss",
        "group": "rollouts_lora_valloss",
        "model": "Ours (LoRA)",
        "label": "best val loss",
        "compare_key": "lora_loss",
    },
    {
        "key": "lora_pitch",
        "group": "rollouts_lora",
        "model": "Ours (LoRA)",
        "label": "best AR pitch",
        "compare_key": "lora_pitch",
    },
    {
        "key": "grpo",
        "group": "rollouts_grpo",
        "model": "Ours (GRPO)",
        "label": "best val reward",
        "compare_key": None,
    },
    {
        "key": "ppo",
        "group": "rollouts_ppo",
        "model": "Ours (PPO)",
        "label": "best val reward",
        "compare_key": None,
    },
    # The two reference models are note-aligned to our filtered input notes, so
    # `filtered` is the only stream they have; overriding the variant here keeps
    # them in the same table rather than silently dropping them.
    {
        "key": "paper1",
        "group": "rollouts_paper1",
        "model": "Paper 1",
        "label": "Zeng+ (joint-apt-epr)",
        "compare_key": None,
        "variant": "filtered",
    },
    {
        "key": "paper2",
        "group": "rollouts_paper2",
        "model": "Paper 2",
        "label": "Beyer & Dai (MIDI2ScoreTF)",
        "compare_key": None,
        "variant": "filtered",
    },
]


def load_payload(path):
    text = Path(path).read_text(encoding="utf-8")
    left = text.find("{")
    right = text.rfind("}")
    if left < 0 or right < left:
        raise SystemExit(f"{path} is not a window.* JSON assignment")
    return json.loads(text[left : right + 1])


def macro_over_pieces(payload, group, variant="raw"):
    """Mean over pieces of (mean over that piece's windows) of each F1 variant."""
    examples = payload["examples"]
    order = list(payload.get("example_order") or examples)
    by_piece = defaultdict(list)
    for eid in order:
        block = (examples[eid].get(group) or {}).get(variant)
        if not isinstance(block, dict):
            return None, 0, 0
        scores = block.get("f1")
        if not isinstance(scores, dict):
            return None, 0, 0
        by_piece[examples[eid].get("piece") or eid].append(scores)
    out = {}
    for crit in VARIANTS:
        piece_means = []
        for windows in by_piece.values():
            vals = [w[crit]["f1"] for w in windows if crit in w]
            if vals:
                piece_means.append(sum(vals) / len(vals))
        out[crit] = (sum(piece_means) / len(piece_means)) if piece_means else None
    return out, len(order), len(by_piece)


def checkpoint_for(payload, row):
    sets = payload.get("checkpoint_sets") or {}
    if row["key"] == "base_loss":
        return payload.get("checkpoint_val_loss")
    if row["key"] == "base_pitch":
        return payload.get("checkpoint")
    if row["key"] == "lora_loss":
        return payload.get("lora_checkpoint_val_loss")
    if row["key"] == "lora_pitch":
        return payload.get("lora_checkpoint")
    if row["key"] == "grpo":
        return payload.get("grpo_checkpoint") or (sets.get("grpo") or {}).get("checkpoint")
    if row["key"] == "ppo":
        return payload.get("ppo_checkpoint") or (sets.get("ppo") or {}).get("checkpoint")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--output", default="visualizer/rl_f1_table.json")
    ap.add_argument("--compare", default="visualizer/ckpt_f1_compare.json")
    ap.add_argument("--variant", default="raw")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Fail if a row disagrees with ckpt_f1_compare.json beyond 1e-9.",
    )
    args = ap.parse_args()

    payload = load_payload(args.data)
    reference = {}
    compare_path = Path(args.compare)
    if compare_path.is_file():
        reference = json.loads(compare_path.read_text(encoding="utf-8"))

    results = []
    for row in ROWS:
        variant = row.get("variant", args.variant)
        scores, n_windows, n_pieces = macro_over_pieces(payload, row["group"], variant)
        if scores is None:
            print(f"skip {row['key']}: {row['group']}/{variant} not in data.js")
            continue
        entry = {
            **{k: row[k] for k in ("key", "group", "model", "label")},
            "variant": variant,
            "checkpoint": checkpoint_for(payload, row),
            "n_windows": n_windows,
            "n_pieces": n_pieces,
            "macro_f1": scores,
        }
        if row["key"] == "ppo":
            entry["best_step"] = payload.get("ppo_best_step")
            entry["val_reward"] = payload.get("ppo_best_val_reward")
        elif row["key"] == "grpo":
            tail = str(entry["checkpoint"] or "").rsplit("checkpoint-", 1)
            entry["best_step"] = int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else None
        ref = reference.get(row["compare_key"] or "")
        if ref:
            deltas = {
                crit: scores[crit] - ref["macro_f1"][crit]
                for crit in VARIANTS
                if ref["macro_f1"].get(crit) is not None
            }
            entry["delta_vs_compare"] = deltas
            worst = max(abs(v) for v in deltas.values())
            if args.verify and worst > 1e-9:
                raise SystemExit(
                    f"{row['key']} disagrees with {args.compare} by {worst:.3e}"
                )
        results.append(entry)
        pct = {c: f"{100 * scores[c]:.2f}%" for c in VARIANTS}
        print(
            f"{row['key']:<11} {str(entry['checkpoint']):<48} "
            f"onset+pitch {pct['onset_pitch']:>7}  "
            f"+dur {pct['onset_pitch_dur']:>7}  "
            f"tol1 {pct['onset_pitch_tol1']:>7}   "
            f"({n_pieces} pieces / {n_windows} windows)"
        )

    Path(args.output).write_text(
        json.dumps({"variant": args.variant, "rows": results}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
