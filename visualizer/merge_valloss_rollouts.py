#!/usr/bin/env python
"""Merge best-val-loss rollout shards into visualizer/data.js."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_sequence_ppl import load_payload  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload, prefix = load_payload(args.data)
    n = 0
    ckpt = lora = None
    for path in args.shards:
        shard = json.loads(Path(path).read_text(encoding="utf-8"))
        ckpt = shard.get("checkpoint_val_loss") or ckpt
        lora = shard.get("lora_checkpoint_val_loss") or lora
        for eid, patch in shard.get("examples", {}).items():
            ex = payload["examples"].get(eid)
            if ex is None:
                print(f"WARNING: unknown example {eid}")
                continue
            if "rollouts_valloss" in patch:
                ex["rollouts_valloss"] = patch["rollouts_valloss"]
                n += 1
            if "rollouts_lora_valloss" in patch:
                ex["rollouts_lora_valloss"] = patch["rollouts_lora_valloss"]
                n += 1

    if ckpt:
        payload["checkpoint_val_loss"] = ckpt
    if lora:
        payload["lora_checkpoint_val_loss"] = lora
    payload["checkpoint_sets"] = {
        "pitch_ar": {
            "label": "best pitch AR",
            "checkpoint": payload.get("checkpoint"),
            "lora_checkpoint": payload.get("lora_checkpoint"),
        },
        "val_loss": {
            "label": "best val loss",
            "checkpoint": payload.get("checkpoint_val_loss"),
            "lora_checkpoint": payload.get("lora_checkpoint_val_loss"),
        },
    }
    print(
        f"Attached valloss rollouts ({n} group patches); "
        f"base={payload.get('checkpoint_val_loss')} "
        f"lora={payload.get('lora_checkpoint_val_loss')}"
    )

    if args.dry_run:
        print("dry-run: not writing")
        return

    out = Path(args.data)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(prefix)
        json.dump(payload, fh)
        fh.write(";\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
