#!/usr/bin/env python
"""Recompute visualizer AR rollouts for best-val-loss checkpoints into data.js.

Reads an existing format-4 ``data.js`` (keeps windows / paper rollouts / pitch-AR
``rollouts`` / ``rollouts_lora`` intact) and writes::

    examples[id].rollouts_valloss
    examples[id].rollouts_lora_valloss

using the same ``compute_rollout_set`` path as ``precompute_visualizer.py``,
conditioned on the stored filtered/raw control notes.

Default checkpoints (saved-ckpt metrics from paper_split training logs):
  base  run_paper_split_v2/checkpoint-2500
  LoRA  run_paper_split_lora_r512/checkpoint-10000

Designed for SLURM array sharding::

  python visualizer/precompute_valloss_rollouts.py \\
      --data visualizer/data.js \\
      --shard-index 0 --num-shards 7 \\
      --output visualizer/valloss_shards/shard_00.json
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
from precompute_visualizer import (  # noqa: E402
    LORA_BASE_MODEL,
    compute_rollout_set,
    load_lora_model,
    tokens_from_controls,
)
from compute_sequence_ppl import load_payload  # noqa: E402

DEFAULT_BASE = "run_paper_split_v2/checkpoint-2500"
DEFAULT_LORA = "run_paper_split_lora_r512/checkpoint-10000"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_BASE)
    ap.add_argument("--lora-checkpoint", default=DEFAULT_LORA)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--topk-onset", type=int, default=5)
    ap.add_argument("--topk-dur", type=int, default=4)
    ap.add_argument("--topk-pitch", type=int, default=8)
    ap.add_argument("--max-candidates", type=int, default=40)
    ap.add_argument("--skip-lora", action="store_true")
    args = ap.parse_args()

    if args.num_shards < 1 or args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("invalid --shard-index / --num-shards")

    payload, _ = load_payload(args.data)
    examples = payload["examples"]
    order = list(payload.get("example_order") or list(examples))
    order = order[args.shard_index :: args.num_shards]
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(order)} windows {order}")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Loading base {args.checkpoint} on {device}...")
    model, loaded = load_model(args.checkpoint, config_source=None)
    if args.device is None:
        device = loaded if isinstance(loaded, torch.device) else torch.device(loaded)
    model.to(device)
    model.eval()

    lora_model = None
    if not args.skip_lora and args.lora_checkpoint:
        print(f"Loading LoRA {args.lora_checkpoint} (base {LORA_BASE_MODEL})...")
        lora_model = load_lora_model(args.lora_checkpoint)
        lora_model.to(device)
        lora_model.eval()

    roll_args = SimpleNamespace(
        topk_onset=args.topk_onset,
        topk_dur=args.topk_dur,
        topk_pitch=args.topk_pitch,
        max_candidates=args.max_candidates,
        slot_progress=False,
    )

    shard = {
        "checkpoint_val_loss": args.checkpoint,
        "lora_checkpoint_val_loss": None if args.skip_lora else args.lora_checkpoint,
        "examples": {},
    }

    t0 = time.perf_counter()
    for eid in tqdm(order, desc="windows"):
        ex = examples[eid]
        perf = ex.get("perf_notes") or []
        raw = ex.get("raw_notes")
        gt = ex.get("gt_score") or []
        tokens = tokens_from_controls(perf, CONTEXT_SIZE - 4)
        t_win = time.perf_counter()
        rollouts, _ = compute_rollout_set(
            model, device, tokens, raw, gt, roll_args,
        )
        entry = {"rollouts_valloss": rollouts}
        if lora_model is not None:
            rollouts_lora, _ = compute_rollout_set(
                lora_model, device, tokens, raw, gt, roll_args,
            )
            entry["rollouts_lora_valloss"] = rollouts_lora
        shard["examples"][eid] = entry
        tqdm.write(
            f"  {eid}: filtered={len(rollouts['filtered']['pred_score'])} "
            f"raw={len((rollouts.get('raw') or {}).get('pred_score') or [])} "
            f"({time.perf_counter() - t_win:.1f}s)"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(shard, fh)
    print(f"Wrote {out} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
