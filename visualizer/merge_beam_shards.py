#!/usr/bin/env python
"""Merge beam-search shards into visualizer/data.js.

Each shard is produced by ``precompute_beams.py`` and has the shape::

    {"examples": {ex_id: {rollouts|rollouts_lora: {variant: beams_dict}}}}

This script attaches each ``beams`` dict onto the matching rollout object in
``data.js`` without touching candidates / paper rollouts / etc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from compute_sequence_ppl import load_payload  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument(
        "--shards", nargs="+", required=True,
        help="One or more shard JSON files from precompute_beams.py",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload, prefix = load_payload(args.data)
    n_attached = 0
    n_missing_ex = 0
    n_missing_roll = 0

    for shard_path in args.shards:
        shard = json.loads(Path(shard_path).read_text(encoding="utf-8"))
        for ex_id, groups in shard.get("examples", {}).items():
            ex = payload["examples"].get(ex_id)
            if ex is None:
                n_missing_ex += 1
                print(f"WARNING: shard example {ex_id} not in data.js")
                continue
            for group_name, variants in groups.items():
                block = ex.get(group_name)
                if not isinstance(block, dict):
                    n_missing_roll += 1
                    print(f"WARNING: {ex_id}.{group_name} missing")
                    continue
                for variant, beams in variants.items():
                    roll = block.get(variant)
                    if not isinstance(roll, dict):
                        n_missing_roll += 1
                        print(f"WARNING: {ex_id}.{group_name}.{variant} missing")
                        continue
                    roll["beams"] = beams
                    n_attached += 1

    payload["beam_widths"] = sorted(
        {
            int(w)
            for shard_path in args.shards
            for groups in json.loads(Path(shard_path).read_text(encoding="utf-8"))
            .get("examples", {})
            .values()
            for variants in groups.values()
            for beams in variants.values()
            for w in (beams or {})
        }
    ) or [5]
    print(
        f"Attached beams to {n_attached} rollout(s); "
        f"missing examples={n_missing_ex} missing rollouts={n_missing_roll}; "
        f"beam_widths={payload['beam_widths']}"
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
