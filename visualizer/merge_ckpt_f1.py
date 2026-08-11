#!/usr/bin/env python
"""Merge compare_ckpt_f1 shards and print macro F1 (piece-averaged)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_f1 import VARIANTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--output", default="visualizer/ckpt_f1_compare.json")
    args = ap.parse_args()

    merged = {}
    for path in args.shards:
        shard = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, block in shard.get("models", {}).items():
            dst = merged.setdefault(
                key,
                {
                    "label": block["label"],
                    "path": block["path"],
                    "kind": block["kind"],
                    "windows": {},
                },
            )
            dst["windows"].update(block.get("windows") or {})

    summary = {}
    print("Macro-mean F1 over pieces (raw unseeded AR; equal piece weight)")
    print(f"{'model':48s} {'n':>3s}  "
          + "  ".join(f"{v:>16s}" for v in VARIANTS))
    for key, block in merged.items():
        by_piece = defaultdict(list)
        for eid, scores in block["windows"].items():
            by_piece[scores.get("piece") or eid].append(scores)
        macros = {}
        for crit in VARIANTS:
            means = [
                sum(w[crit] for w in wins) / len(wins)
                for wins in by_piece.values()
                if wins
            ]
            macros[crit] = sum(means) / len(means) if means else None
        summary[key] = {
            "label": block["label"],
            "path": block["path"],
            "kind": block["kind"],
            "n_windows": len(block["windows"]),
            "n_pieces": len(by_piece),
            "macro_f1": macros,
            "windows": block["windows"],
        }
        line = f"{block['label'][:48]:48s} {len(by_piece):3d}  "
        for crit in VARIANTS:
            v = macros[crit]
            line += f"  {100 * v:15.2f}%" if v is not None else f"  {'–':>16s}"
        print(line)

    # Head-to-head
    print("\nHead-to-head (best-loss − viz/pitch), percentage points:")
    for kind, pitch_k, loss_k in (
        ("base", "base_pitch", "base_loss"),
        ("LoRA", "lora_pitch", "lora_loss"),
    ):
        a, b = summary[pitch_k]["macro_f1"], summary[loss_k]["macro_f1"]
        print(f"  {kind}:")
        for crit in VARIANTS:
            delta = 100 * (b[crit] - a[crit])
            winner = "best-loss" if delta > 0 else ("viz/pitch" if delta < 0 else "tie")
            print(f"    {crit:18s}  Δ={delta:+6.2f} pp  → {winner} better "
                  f"(viz {100*a[crit]:.2f}% vs loss {100*b[crit]:.2f}%)")

    out = Path(args.output)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
