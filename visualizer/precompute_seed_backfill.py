#!/usr/bin/env python
"""Compute seed-2..5 patches for one existing visualization model/window shard."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from atomic_json import atomic_dump_json  # noqa: E402
from compute_sequence_ppl import load_payload  # noqa: E402
from seed_pipeline_common import (  # noqa: E402
    BACKFILL_GROUPS,
    EXTRA_SEED_COUNTS,
    attach_inline_metrics,
    build_seed_rollout,
    canonical_seed_variant,
    checkpoint_for_group,
    group_is_lora,
    load_checkpoint_model,
    roll_args,
    valid_rollout,
)


NUM_SHARDS = 4


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--output", required=True)
    ap.add_argument("--group", required=True, choices=BACKFILL_GROUPS)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--topk-onset", type=int, default=5)
    ap.add_argument("--topk-dur", type=int, default=4)
    ap.add_argument("--topk-pitch", type=int, default=8)
    ap.add_argument("--max-candidates", type=int, default=40)
    args = ap.parse_args()
    if not 0 <= args.shard_index < NUM_SHARDS:
        raise SystemExit(f"--shard-index must be 0..{NUM_SHARDS - 1}")

    payload, _prefix = load_payload(args.data)
    checkpoint = args.checkpoint or checkpoint_for_group(payload, args.group)
    if not checkpoint:
        raise SystemExit(f"no checkpoint metadata for {args.group}")
    examples = payload["examples"]
    full_order = list(payload.get("example_order") or examples)
    order = full_order[args.shard_index::NUM_SHARDS]
    for eid in order:
        if not isinstance(examples[eid].get(args.group), dict):
            raise SystemExit(f"{eid} has no existing {args.group} block")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_checkpoint_model(
        checkpoint, device, is_lora=group_is_lora(args.group)
    )
    rollout_args = roll_args(
        args.topk_onset,
        args.topk_dur,
        args.topk_pitch,
        args.max_candidates,
    )
    shard = {
        "format": 1,
        "group": args.group,
        "checkpoint": checkpoint,
        "seed_counts": list(EXTRA_SEED_COUNTS),
        "seed_alignment": "raw_notes.j",
        "shard_index": args.shard_index,
        "num_shards": NUM_SHARDS,
        "example_order": order,
        "examples": {},
    }
    started = time.perf_counter()
    for eid in tqdm(order, desc=f"{args.group} shard {args.shard_index}"):
        ex = examples[eid]
        block = {}
        for count in EXTRA_SEED_COUNTS:
            for stream in ("filtered", "raw"):
                variant = canonical_seed_variant(stream, count)
                rollout = build_seed_rollout(
                    model, device, ex, stream, count, rollout_args
                )
                if rollout is None:
                    continue
                attach_inline_metrics(model, device, ex, variant, rollout)
                if not valid_rollout(rollout):
                    raise SystemExit(f"inline metrics failed for {eid}/{args.group}/{variant}")
                block[variant] = rollout
        shard["examples"][eid] = {args.group: block}
        tqdm.write(f"{eid}: {len(block)} seeded variants")

    atomic_dump_json(args.output, shard)
    print(
        f"Atomically wrote {args.output} in {time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
