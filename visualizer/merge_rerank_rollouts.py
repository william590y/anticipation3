#!/usr/bin/env python
"""Merge discriminative-rerank decode shards into ``visualizer/data.js``.

Reads shards written by ``rerank_viz_rollout.py`` and attaches, per example::

    rollouts_rerank.<variant> = {"pred_score": [...], "rerank_meta": {...}}

plus the top-level metadata key ``checkpoint_rerank`` (the FT checkpoint the
beam ran with) and ``rerank_decode_weights`` (alpha/beta/gamma actually used),
so the UI can label the row. Atomic write via ``atomic_json``.

  python visualizer/merge_rerank_rollouts.py \
      --shards visualizer/rerank_shards/*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atomic_json import atomic_dump_data_js  # noqa: E402
from compute_sequence_ppl import load_payload  # noqa: E402

GROUP = "rollouts_rerank"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--group", default=GROUP,
                    help="rollout group to attach (e.g. rollouts_rerank_ab30)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    group = args.group

    payload, prefix = load_payload(args.data)
    attached = 0
    for path in args.shards:
        shard = json.loads(Path(path).read_text(encoding="utf-8"))
        variant = shard.get("variant", "filtered")
        payload[f"checkpoint_{group}"] = shard["checkpoint"]
        weights = shard.get("weights") or {}
        payload[f"{group}_decode_weights"] = {
            k: weights.get(k) for k in ("alpha", "beta", "gamma")}
        for eid, entry in shard.get("examples", {}).items():
            ex = payload["examples"].get(eid)
            if ex is None:
                print(f"WARNING: unknown example {eid}")
                continue
            ex.setdefault(group, {})[variant] = {
                "pred_score": entry["pred_score"],
                "rerank_meta": entry.get("rerank_meta"),
            }
            attached += 1

    print(f"Attached {group} to {attached} example/variant slots")
    if args.dry_run:
        print("dry-run: not writing")
        return
    atomic_dump_data_js(args.data, prefix, payload)
    print(f"Wrote {args.data}")


if __name__ == "__main__":
    main()
