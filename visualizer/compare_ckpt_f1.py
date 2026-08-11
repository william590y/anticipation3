#!/usr/bin/env python
"""Compare note-level F1 for viz (best AR-pitch) vs best-val-loss checkpoints.

Protocol matches the visualizer mean-F1 table: unfiltered (raw) unseeded greedy
AR rollouts on the paper-split windows in data.js, then macro-average F1 over
pieces (equal piece weight).

Checkpoints compared (from training logs at save time):
  base  viz/pitch : run_paper_split_v2/checkpoint-7500      (pitch 89.74%, loss 1.2568)
  base  best-loss : run_paper_split_v2/checkpoint-2500      (pitch 82.33%, loss 1.2180)
  lora  viz/pitch : run_paper_split_lora_r512/checkpoint-15000 (pitch 92.50%, loss 1.2041)
  lora  best-loss : run_paper_split_lora_r512/checkpoint-10000 (pitch 86.59%, loss 1.1950)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from anticipation.config import CONTEXT_SIZE  # noqa: E402
from evaluate_muster import load_model  # noqa: E402
from precompute_visualizer import LORA_BASE_MODEL, load_lora_model, tokens_from_controls  # noqa: E402
from precompute_beams import beam_search_score  # noqa: E402
from compute_sequence_ppl import load_payload  # noqa: E402
from compute_f1 import VARIANTS, score_notes  # noqa: E402

COMPARE = [
    {
        "key": "base_pitch",
        "label": "base viz (ckpt-7500, best AR pitch)",
        "kind": "base",
        "path": "run_paper_split_v2/checkpoint-7500",
    },
    {
        "key": "base_loss",
        "label": "base best-loss (ckpt-2500)",
        "kind": "base",
        "path": "run_paper_split_v2/checkpoint-2500",
    },
    {
        "key": "lora_pitch",
        "label": "LoRA viz (ckpt-15000, best AR pitch)",
        "kind": "lora",
        "path": "run_paper_split_lora_r512/checkpoint-15000",
    },
    {
        "key": "lora_loss",
        "label": "LoRA best-loss (ckpt-10000)",
        "kind": "lora",
        "path": "run_paper_split_lora_r512/checkpoint-10000",
    },
]


def clean_notes(notes):
    return [n for n in (notes or []) if n and n.get("p") is not None]


def macro_over_pieces(per_window, examples):
    """Mean over pieces of (mean over that piece's windows)."""
    by_piece = defaultdict(list)
    for eid, scores in per_window.items():
        piece = examples[eid].get("piece") or eid
        by_piece[piece].append(scores)
    out = {}
    for crit in VARIANTS:
        piece_means = []
        for windows in by_piece.values():
            vals = [w[crit] for w in windows if crit in w]
            if vals:
                piece_means.append(sum(vals) / len(vals))
        out[crit] = (sum(piece_means) / len(piece_means)) if piece_means else None
    return out, len(by_piece)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--output", required=True, help="Shard JSON path.")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--models", default="all",
        help="Comma-separated COMPARE keys, or 'all'.",
    )
    args = ap.parse_args()

    payload, _ = load_payload(args.data)
    examples = payload["examples"]
    order = list(payload.get("example_order") or list(examples))
    if args.num_shards < 1 or args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("invalid shard settings")
    order = order[args.shard_index :: args.num_shards]
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(order)} windows {order}")

    if args.models.strip() == "all":
        models = COMPARE
    else:
        want = {k.strip() for k in args.models.split(",") if k.strip()}
        models = [m for m in COMPARE if m["key"] in want]
        if not models:
            raise SystemExit(f"no models matched {want}")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Pre-build raw control streams once.
    window_inputs = {}
    for eid in order:
        ex = examples[eid]
        controls = ex.get("raw_notes") or ex.get("perf_notes") or []
        tokens = tokens_from_controls(controls, CONTEXT_SIZE - 4)
        gt = clean_notes(ex.get("gt_score") or [])
        window_inputs[eid] = {"tokens": tokens, "gt": gt, "piece": ex.get("piece") or eid}

    results = {"shard_index": args.shard_index, "num_shards": args.num_shards, "models": {}}

    for spec in models:
        print(f"\n=== Loading {spec['label']} ({spec['path']}) ===")
        if spec["kind"] == "base":
            model, loaded = load_model(spec["path"], config_source=None)
            if args.device is None:
                device = loaded if isinstance(loaded, torch.device) else torch.device(loaded)
            model.to(device)
        else:
            model = load_lora_model(spec["path"])
            model.to(device)
        model.eval()

        per_window = {}
        for eid in tqdm(order, desc=spec["key"]):
            inp = window_inputs[eid]
            pred = beam_search_score(
                model, device, inp["tokens"], num_beams=1, seed_note=None,
            )
            pred_notes = clean_notes(pred)
            scored = score_notes(pred_notes, inp["gt"])
            per_window[eid] = {
                "piece": inp["piece"],
                **{v: scored[v]["f1"] for v in VARIANTS},
                "n_pred": len(pred_notes),
                "n_gt": len(inp["gt"]),
            }
        results["models"][spec["key"]] = {
            "label": spec["label"],
            "path": spec["path"],
            "kind": spec["kind"],
            "windows": per_window,
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(results, fh)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
