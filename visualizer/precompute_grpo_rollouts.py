#!/usr/bin/env python
"""Add the GRPO model's rollouts to data.js, plus multi-note GT-seeded variants.

Writes, per window:

    examples[id].rollouts_grpo          full variant set for the GRPO checkpoint,
                                        including the legacy one-note `*_seeded`
                                        spelling plus `*_seed2` through `*_seed5`
    examples[id].rollouts_seed_patch    the same seed2..5 variants for the base
                                        model already in `rollouts`, so the UI's
                                        seeded-note control works for it too

The GRPO checkpoint is a plain HF directory (full fine-tune, not an adapter), so it
loads through `evaluate_muster.load_model` exactly like the base model does.

Sharded for a SLURM array, same pattern as precompute_valloss_rollouts.py:

    python visualizer/precompute_grpo_rollouts.py \\
        --data visualizer/data.js --shard-index 0 --num-shards 4 \\
        --output visualizer/grpo_shards/shard_00.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from anticipation.config import CONTEXT_SIZE  # noqa: E402
from evaluate_muster import load_model  # noqa: E402
from precompute_visualizer import compute_rollout_set, tokens_from_controls  # noqa: E402
from compute_sequence_ppl import load_payload  # noqa: E402
from atomic_json import atomic_dump_json  # noqa: E402

DEFAULT_GRPO = "run_grpo_acc_reward/checkpoint-250"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--output", required=True)
    ap.add_argument("--grpo-checkpoint", default=DEFAULT_GRPO)
    ap.add_argument("--base-checkpoint", default=None,
                    help="Model behind the existing `rollouts` key; defaults to the "
                         "data.js `checkpoint` field. Used only for the seed patch.")
    ap.add_argument("--seed-counts", default="1,2,3,4,5")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--topk-onset", type=int, default=5)
    ap.add_argument("--topk-dur", type=int, default=4)
    ap.add_argument("--topk-pitch", type=int, default=8)
    ap.add_argument("--max-candidates", type=int, default=40)
    ap.add_argument("--skip-base-seeds", action="store_true")
    args = ap.parse_args()

    seed_counts = sorted({int(s) for s in args.seed_counts.split(",") if s.strip()})
    if any(n < 1 or n > 5 for n in seed_counts):
        raise SystemExit("--seed-counts must contain only values from 1 through 5")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid --shard-index / --num-shards")

    payload, _ = load_payload(args.data)
    examples = payload["examples"]
    order = list(payload.get("example_order") or list(examples))
    order = order[args.shard_index :: args.num_shards]
    base_checkpoint = args.base_checkpoint or payload.get("checkpoint")
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(order)} windows {order}")
    print(f"GRPO: {args.grpo_checkpoint}   base (seed patch): {base_checkpoint}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    roll_args = SimpleNamespace(
        topk_onset=args.topk_onset, topk_dur=args.topk_dur, topk_pitch=args.topk_pitch,
        max_candidates=args.max_candidates, slot_progress=False,
    )

    shard = {
        "format": 2,
        "seed_alignment": "raw_notes.j",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "grpo_checkpoint": args.grpo_checkpoint,
        "seed_counts": seed_counts,
        "base_seed_patch": bool(not args.skip_base_seeds and base_checkpoint),
        "example_order": order,
        "examples": {eid: {} for eid in order},
    }

    def window_inputs(eid):
        ex = examples[eid]
        perf = ex.get("perf_notes") or []
        return (
            tokens_from_controls(perf, CONTEXT_SIZE - 4),
            ex.get("raw_notes"),
            ex.get("gt_score") or [],
        )

    jobs = [("rollouts_grpo", args.grpo_checkpoint, seed_counts)]
    if not args.skip_base_seeds and base_checkpoint:
        jobs.append(("rollouts_seed_patch", base_checkpoint,
                     [n for n in seed_counts if n != 1]))

    for key, checkpoint, counts in jobs:
        if not counts and key == "rollouts_seed_patch":
            continue
        print(f"\n=== {key}: loading {checkpoint} ===", flush=True)
        model, loaded = load_model(checkpoint, config_source=None)
        target = device if args.device else (
            loaded if isinstance(loaded, torch.device) else torch.device(loaded)
        )
        model.to(target)
        model.eval()

        started = time.perf_counter()
        for eid in tqdm(order, desc=key):
            tokens, raw, gt = window_inputs(eid)
            t_win = time.perf_counter()
            rollouts, _ = compute_rollout_set(
                model, target, tokens, raw, gt, roll_args, seed_counts=counts,
            )
            if key == "rollouts_seed_patch":
                # only the extra seeded variants; the plain/1-seeded ones already
                # exist under `rollouts` and must not be recomputed or overwritten
                rollouts = {
                    k: v for k, v in rollouts.items()
                    if any(k.endswith(f"_seed{n}") for n in counts) and v
                }
            shard["examples"][eid][key] = rollouts
            tqdm.write(f"  {eid}: {len(rollouts)} variants ({time.perf_counter() - t_win:.1f}s)")
        print(f"{key} done in {time.perf_counter() - started:.1f}s")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.output)
    atomic_dump_json(out, shard)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
