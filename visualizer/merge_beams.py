"""Merge beam-search shards into visualizer/data.js.

Usage:
  python visualizer/merge_beams.py \\
      --data visualizer/data.js \\
      --shards visualizer/beam_shards/shard_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from compute_sequence_ppl import load_payload  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--shards", nargs="+", required=True,
                    help="Shard JSON paths (shell globs ok if expanded)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    shard_paths = []
    for pattern in args.shards:
        matched = sorted(glob.glob(pattern))
        if matched:
            shard_paths.extend(matched)
        else:
            shard_paths.append(pattern)
    shard_paths = sorted(set(shard_paths))

    payload, prefix = load_payload(args.data)
    examples = payload["examples"]
    n_merged = 0
    n_variants = 0
    seen_ids = set()

    for path in shard_paths:
        with open(path, encoding="utf-8") as fh:
            shard = json.load(fh)
        for eid, group_patch in shard.get("examples", {}).items():
            if eid not in examples:
                print(f"WARNING: shard has unknown example {eid} ({path})")
                continue
            seen_ids.add(eid)
            ex = examples[eid]
            for group_name, variant_patch in group_patch.items():
                block = ex.get(group_name)
                if not isinstance(block, dict):
                    print(f"WARNING: {eid}.{group_name} missing")
                    continue
                for variant, roll_patch in variant_patch.items():
                    roll = block.get(variant)
                    if not isinstance(roll, dict):
                        print(f"WARNING: {eid}.{group_name}.{variant} missing")
                        continue
                    beams = roll_patch.get("beams")
                    if not beams:
                        continue
                    roll["beams"] = beams
                    n_variants += 1
            n_merged += 1

    order = payload.get("example_order") or list(examples)
    missing = [eid for eid in order if eid not in seen_ids]
    print(f"Merged beams into {n_merged} windows / {n_variants} rollout variants "
          f"from {len(shard_paths)} shard(s)")
    if missing:
        print(f"WARNING: no shard data for {len(missing)} windows: {missing}")

    # Spot-check beam=1 ≈ existing greedy pred_score on first available window.
    for eid in order:
        ex = examples[eid]
        roll = (ex.get("rollouts") or {}).get("filtered")
        if not roll or "beams" not in roll:
            continue
        b1 = roll["beams"].get("1", {}).get("pred_score")
        greedy = roll.get("pred_score")
        if b1 and greedy:
            n = min(len(b1), len(greedy))
            mismatches = 0
            for i in range(n):
                a, b = b1[i], greedy[i]
                if (a is None) != (b is None):
                    mismatches += 1
                elif a is not None and (a.get("t") != b.get("t") or a.get("d") != b.get("d")
                                        or a.get("p") != b.get("p")):
                    mismatches += 1
            print(f"Verify {eid} filtered beam=1 vs greedy: "
                  f"{mismatches}/{n} slot mismatches "
                  f"{'(ok)' if mismatches == 0 else '(WARN)'}")
        break

    payload["beams_max"] = max(
        (
            max(int(k) for k in ((ex.get("rollouts") or {}).get("filtered") or {})
                .get("beams", {}) or [1])
            for ex in examples.values()
        ),
        default=1,
    )

    if args.dry_run:
        print("Dry run — not writing data.js")
        return

    out = Path(args.data)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(prefix)
        json.dump(payload, fh)
        fh.write(";\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
