#!/usr/bin/env python
"""Merge 40k masked-loss rollout shards into visualizer/data.js.

Attaches ``rollouts_masked_40k`` / ``rollouts_masked_40k_final`` only.
Never overwrites ``rollouts_masked`` (the 20k run).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atomic_json import atomic_dump_data_js  # noqa: E402
from compute_sequence_ppl import load_payload  # noqa: E402

PROTECTED_GROUPS = frozenset({"rollouts_masked"})
DEFAULT_GROUPS = ("rollouts_masked_40k", "rollouts_masked_40k_final")
META_KEYS = {
    "rollouts_masked_40k": "checkpoint_masked_40k",
    "rollouts_masked_40k_final": "checkpoint_masked_40k_final",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload, prefix = load_payload(args.data)
    attached = {g: 0 for g in DEFAULT_GROUPS}
    extra = {}
    ckpts = dict(payload.get("checkpoints_masked_40k") or {})

    for path in args.shards:
        shard = json.loads(Path(path).read_text(encoding="utf-8"))
        for group, ckpt in (shard.get("checkpoints") or {}).items():
            if group in PROTECTED_GROUPS:
                print(f"WARNING: skipping protected group {group} from {path}")
                continue
            ckpts[group] = ckpt
        for eid, patch in shard.get("examples", {}).items():
            ex = payload["examples"].get(eid)
            if ex is None:
                print(f"WARNING: unknown example {eid}")
                continue
            for group, block in patch.items():
                if group in PROTECTED_GROUPS:
                    print(f"WARNING: skip {eid}.{group} (protected)")
                    continue
                if not group.startswith("rollouts_"):
                    continue
                ex[group] = block
                if group in attached:
                    attached[group] += 1
                else:
                    extra[group] = extra.get(group, 0) + 1

    for group, meta in META_KEYS.items():
        if group in ckpts:
            payload[meta] = ckpts[group]
    payload["checkpoints_masked_40k"] = ckpts

    bits = [f"{g}={n}" for g, n in attached.items()]
    bits += [f"{g}={n}" for g, n in extra.items()]
    print(f"Attached {', '.join(bits)}; ckpts={ckpts}")

    if args.dry_run:
        print("dry-run: not writing")
        return

    atomic_dump_data_js(args.data, prefix, payload)
    print(f"Wrote {args.data}")


if __name__ == "__main__":
    main()
