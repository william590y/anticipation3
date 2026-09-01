#!/usr/bin/env python
"""Merge masked-loss rollout shards into visualizer/data.js."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_sequence_ppl import load_payload  # noqa: E402

GROUP = "rollouts_masked"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload, prefix = load_payload(args.data)
    n = 0
    ckpt = None
    for path in args.shards:
        shard = json.loads(Path(path).read_text(encoding="utf-8"))
        ckpt = shard.get("checkpoint_masked") or ckpt
        for eid, patch in shard.get("examples", {}).items():
            ex = payload["examples"].get(eid)
            if ex is None:
                print(f"WARNING: unknown example {eid}")
                continue
            if GROUP in patch:
                ex[GROUP] = patch[GROUP]
                n += 1

    if ckpt:
        payload["checkpoint_masked"] = ckpt
    print(f"Attached {n} {GROUP} windows; checkpoint={payload.get('checkpoint_masked')}")

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
