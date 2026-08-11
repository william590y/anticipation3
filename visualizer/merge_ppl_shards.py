#!/usr/bin/env python
"""Merge sequence/beam perplexity metric shards into visualizer/data.js."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compute_sequence_ppl import load_payload, merge_metrics_into_roll


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload, prefix = load_payload(args.data)
    n_windows = 0
    n_rolls = 0
    for path in args.shards:
        shard = json.loads(Path(path).read_text(encoding="utf-8"))
        for eid, groups in shard.get("examples", {}).items():
            ex = payload["examples"].get(eid)
            if ex is None:
                print(f"WARNING: unknown example {eid} in {path}")
                continue
            n_windows += 1
            for group_name, variants in groups.items():
                block = ex.get(group_name)
                if not isinstance(block, dict):
                    continue
                for variant, roll_patch in variants.items():
                    roll = block.get(variant)
                    if not isinstance(roll, dict):
                        continue
                    merge_metrics_into_roll(roll, roll_patch)
                    n_rolls += 1

    print(f"Merged metrics into {n_rolls} rollout(s) across {n_windows} window refs")
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
