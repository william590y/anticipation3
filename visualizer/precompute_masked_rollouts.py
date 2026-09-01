#!/usr/bin/env python
"""AR rollouts for the masked-loss paper-split run into visualizer shards.

Reads the existing format-4 ``data.js`` (windows / FT / LoRA / paper rollouts
stay intact) and writes ``examples[id].rollouts_masked`` using the same
``compute_rollout_set`` path as ``precompute_visualizer.py``.

Default checkpoint is ``checkpoint-20000`` (final saved step of the masked
run). AR pitch is 100% at every save, so this is chosen by AR onset
(16.6%) rather than ``pick_best_checkpoint.py``.

  python visualizer/precompute_masked_rollouts.py \\
      --data visualizer/data.js \\
      --shard-index 0 --num-shards 2 \\
      --output visualizer/masked_shards/shard_00.json
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

DEFAULT_CKPT = "run_paper_split_v2_masked/checkpoint-20000"
GROUP = "rollouts_masked"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--topk-onset", type=int, default=5)
    ap.add_argument("--topk-dur", type=int, default=4)
    ap.add_argument("--topk-pitch", type=int, default=8)
    ap.add_argument("--max-candidates", type=int, default=40)
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
    print(f"Loading {args.checkpoint} on {device}...")
    model, loaded = load_model(args.checkpoint, config_source=None)
    if args.device is None:
        device = loaded if isinstance(loaded, torch.device) else torch.device(loaded)
    model.to(device)
    model.eval()

    roll_args = SimpleNamespace(
        topk_onset=args.topk_onset,
        topk_dur=args.topk_dur,
        topk_pitch=args.topk_pitch,
        max_candidates=args.max_candidates,
        slot_progress=False,
    )

    shard = {"checkpoint_masked": args.checkpoint, "examples": {}}
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
        shard["examples"][eid] = {GROUP: rollouts}
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
