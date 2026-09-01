#!/usr/bin/env python
"""AR rollouts for the 40k masked-loss paper-split run into visualizer shards.

Shades BOTH checkpoints across a global rank so 3090 and A6000 jobs can run
in parallel without mixing GRES:

  checkpoint-7500  -> examples[id].rollouts_masked_40k
  checkpoint-40000 -> examples[id].rollouts_masked_40k_final

Never writes ``rollouts_masked`` (the 20k run). Format 4: four AR variants
(filtered/raw × plain/GT-seed) via batched KV-cached decode.

  python visualizer/precompute_masked_40k_rollouts.py \\
      --data visualizer/data.js \\
      --rank 0 --world-size 6 \\
      --output visualizer/masked_40k_shards/shard_00.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from anticipation.config import CONTEXT_SIZE  # noqa: E402
from compute_sequence_ppl import load_payload  # noqa: E402
from fast_rollout import (  # noqa: E402
    compute_rollout_sets_batched,
    default_roll_args,
    load_and_compile_model,
    warmup_compile,
)
from precompute_visualizer import tokens_from_controls  # noqa: E402

PROTECTED_GROUPS = frozenset({"rollouts_masked"})

DEFAULT_CKPTS = [
    ("run_paper_split_v2_masked_40k/checkpoint-7500", "rollouts_masked_40k"),
    ("run_paper_split_v2_masked_40k/checkpoint-40000", "rollouts_masked_40k_final"),
]


def parse_ckpt_groups(values):
    out = []
    for raw in values:
        if ":" not in raw:
            raise SystemExit(f"checkpoint spec must be path:group, got {raw!r}")
        path, group = raw.rsplit(":", 1)
        if group in PROTECTED_GROUPS:
            raise SystemExit(f"refusing to write protected group {group}")
        if not group.startswith("rollouts_"):
            raise SystemExit(f"group must start with rollouts_, got {group}")
        out.append((path, group))
    return out


def plan_units(example_order, ckpt_groups):
    """(checkpoint, group, example_id) in checkpoint-major order."""
    return [(ckpt, group, eid) for ckpt, group in ckpt_groups for eid in example_order]


def chunk_for_rank(units, rank, world):
    """Consecutive chunks so a rank typically loads one checkpoint."""
    if world < 1 or rank < 0 or rank >= world:
        raise ValueError(f"invalid rank/world: {rank}/{world}")
    n = len(units)
    base, rem = divmod(n, world)
    sizes = [base + (1 if i < rem else 0) for i in range(world)]
    start = sum(sizes[:rank])
    return units[start:start + sizes[rank]]


def gpu_window_batch():
    if not torch.cuda.is_available():
        return 1
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    # 48GB A6000: two packed windows (8 variants) still fit; 24GB 3090: one.
    return 2 if mem_gb >= 40 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--checkpoint",
        action="append",
        dest="checkpoints",
        default=None,
        help="path:group (repeatable). Default: the two 40k masked checkpoints.",
    )
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--topk-onset", type=int, default=5)
    ap.add_argument("--topk-dur", type=int, default=4)
    ap.add_argument("--topk-pitch", type=int, default=8)
    ap.add_argument("--max-candidates", type=int, default=40)
    ap.add_argument("--window-batch", type=int, default=0, help="0 = auto from GPU memory")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--no-tensorrt", action="store_true")
    args = ap.parse_args()

    ckpt_groups = parse_ckpt_groups(args.checkpoints) if args.checkpoints else list(DEFAULT_CKPTS)
    payload, _ = load_payload(args.data)
    examples = payload["examples"]
    order = list(payload.get("example_order") or list(examples))
    units = plan_units(order, ckpt_groups)
    mine = chunk_for_rank(units, args.rank, args.world_size)
    print(
        f"Rank {args.rank}/{args.world_size}: {len(mine)} window-units "
        f"(of {len(units)} total, {len(ckpt_groups)} checkpoints × {len(order)} windows)"
    )
    for ckpt, group, eid in mine:
        print(f"  {group} {eid} <- {ckpt}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    roll_args = default_roll_args(
        topk_onset=args.topk_onset,
        topk_dur=args.topk_dur,
        topk_pitch=args.topk_pitch,
        max_candidates=args.max_candidates,
    )
    window_batch = args.window_batch or gpu_window_batch()
    print(f"device={device} window_batch={window_batch}")

    shard = {"checkpoints": {}, "rank": args.rank, "world_size": args.world_size, "examples": {}}
    t_all = time.perf_counter()
    loaded_ckpt = None
    model = None
    compile_backend = "eager"

    i = 0
    while i < len(mine):
        ckpt, group, _ = mine[i]
        same = []
        while i < len(mine) and mine[i][0] == ckpt and mine[i][1] == group:
            same.append(mine[i])
            i += 1
        if group in PROTECTED_GROUPS:
            raise SystemExit(f"refusing to write protected group {group}")

        if loaded_ckpt != ckpt:
            print(f"Loading {ckpt} on {device}...")
            model, compile_backend = load_and_compile_model(
                ckpt,
                device,
                compile_model=not args.no_compile,
                compile_mode=args.compile_mode,
                try_tensorrt=not args.no_tensorrt,
            )
            loaded_ckpt = ckpt
            first_eid = same[0][2]
            warmup_tokens = tokens_from_controls(
                examples[first_eid].get("perf_notes") or [], CONTEXT_SIZE - 4
            )
            t_w = time.perf_counter()
            warmup_compile(
                model, device, warmup_tokens,
                batch_size=max(4, window_batch * 4),
            )
            print(f"  warmup {time.perf_counter() - t_w:.1f}s backend={compile_backend}")
        shard["checkpoints"][group] = ckpt

        for start in range(0, len(same), window_batch):
            chunk = same[start:start + window_batch]
            wins = []
            for _, _, eid in chunk:
                ex = examples[eid]
                perf = ex.get("perf_notes") or []
                raw = ex.get("raw_notes")
                gt = ex.get("gt_score") or []
                wins.append({
                    "eid": eid,
                    "tokens": tokens_from_controls(perf, CONTEXT_SIZE - 4),
                    "raw_notes": raw,
                    "gt_by_slot": gt,
                })
            t_win = time.perf_counter()
            packed = compute_rollout_sets_batched(model, device, wins, roll_args)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_win
            per = elapsed / max(len(chunk), 1)
            for _, _, eid in chunk:
                rollouts = packed[eid]
                shard["examples"].setdefault(eid, {})[group] = rollouts
                raw_n = len((rollouts.get("raw") or {}).get("pred_score") or [])
                filt_n = len((rollouts.get("filtered") or {}).get("pred_score") or [])
                tqdm.write(
                    f"  {eid} {group}: filtered={filt_n} raw={raw_n} "
                    f"({per:.1f}s/window, backend={compile_backend})"
                )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(shard, fh)
    print(
        f"Wrote {out} ({time.perf_counter() - t_all:.1f}s, "
        f"{len(mine)} units, backend={compile_backend})"
    )


if __name__ == "__main__":
    main()
