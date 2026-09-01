#!/usr/bin/env python
"""Generated vs GT INTERLEAVED-sequence perplexity under the masked-40k model,
on the visualizer's 24 paper-split windows.

Model: ``run_paper_split_v2_masked_40k/checkpoint-40000`` -- the last
checkpoint of the paper-split run trained with
``--loss_mask_performance_tokens`` (performance tokens carry no loss).

Protocol per viz window (exact packed line recovered from
``data/{val,test}_paper.txt`` via the payload's ``source_line_index``):

1. Greedy-decode a score with the masked model in the packed format,
   ground-truth controls teacher-forced (`batched_autoregressive_generate_score`,
   ``ground_truth_score_tokens_to_feed=0`` -- the repo's standard rollout).
2. Compute teacher-forced NLL of BOTH interleaved sequences (GT packed line vs
   generated packed context) under the same model:
     * **score-token scope** (primary): NLL of the 414 body score tokens in
       their interleaved context, constrained and unconstrained -- this is the
       scope of the model's own training loss, since performance tokens were
       masked out of it;
     * **all-token scope** (secondary): NLL over every token from position 1,
       unconstrained. Reported for completeness -- the model never trained to
       predict performance tokens, and those tokens are identical between the
       two arms (controls are teacher-forced GT), so differences here come
       from score positions plus the controls' conditionals on differing
       score prefixes.

Usage:
  python eval_masked40k_viz_ppl.py --split validation --output out/val.json
  python eval_masked40k_viz_ppl.py --split test --output out/test.json
  python eval_masked40k_viz_ppl.py --merge 'masked40k_viz_ppl/*.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from anticipation.packed_sequence import ALTERNATING_START, iter_score_slot_positions
from eval_base_score_ppl import nll_at_positions, slot_logit_masks
from inference import batched_autoregressive_generate_score

DEFAULT_CKPT = "run_paper_split_v2_masked_40k/checkpoint-40000"
DEFAULT_PAYLOAD = "visualizer/data_slim.js"
SPLIT_FILES = {"validation": "data/val_paper.txt", "test": "data/test_paper.txt"}


def load_viz_windows(payload_path: str, split: str):
    raw = open(payload_path).read()
    raw = raw[raw.index("=") + 1:].strip().rstrip(";")
    payload = json.loads(raw)
    wanted = [(key, payload["examples"][key]) for key in payload["example_order"]
              if payload["examples"][key]["split"] == split]
    line_indices = {ex["source_line_index"] for _, ex in wanted}
    lines: dict[int, list[int]] = {}
    with open(SPLIT_FILES[split]) as f:
        for i, line in enumerate(f):
            if i in line_indices:
                lines[i] = [int(t) for t in line.split("|", 1)[0].split()]
                if len(lines) == len(line_indices):
                    break
    keys, windows = [], []
    for key, ex in wanted:
        keys.append(key)
        windows.append(torch.tensor(lines[ex["source_line_index"]],
                                    dtype=torch.long))
    return keys, windows


@torch.inference_mode()
def run_split(args):
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    from evaluate_muster import load_model
    print(f"Loading {args.checkpoint} ...")
    model, _model_device = load_model(args.checkpoint)
    model = model.to(device).eval()

    keys, windows = load_viz_windows(args.payload, args.split)
    print(f"{args.split}: {len(keys)} viz windows: {keys}")
    batch = torch.stack(windows).to(device)
    length = batch.shape[1]

    gen_ctx = batched_autoregressive_generate_score(
        model, batch, ALTERNATING_START, str(device),
        constrain_score_tokens=True, ground_truth_score_tokens_to_feed=0)

    masks = slot_logit_masks(device)
    positions = [p for p in iter_score_slot_positions(length) if p + 5 < length]
    flat_pos = torch.tensor([p + j for p in positions for j in range(3)],
                            device=device)
    slot_ids = torch.tensor([j for _ in positions for j in range(3)],
                            device=device)
    all_target = torch.arange(1, length, device=device)
    all_pred = all_target - 1
    all_slots = torch.zeros(length - 1, dtype=torch.long, device=device)

    rows = []
    for key, gt_seq, gen_seq in zip(keys, batch, gen_ctx):
        rows.append({"key": key})
        for arm, seq in (("gt", gt_seq), ("gen", gen_seq)):
            s = seq.unsqueeze(0)
            nll_u, nll_c = nll_at_positions(model, s, flat_pos - 1, flat_pos,
                                            slot_ids, masks, chunk=1)
            rows[-1][f"score_{arm}_c"] = float(nll_c.mean())
            rows[-1][f"score_{arm}_u"] = float(nll_u.mean())
            nll_all, _ = nll_at_positions(model, s, all_pred, all_target,
                                          all_slots, masks, chunk=1)
            rows[-1][f"all_{arm}_u"] = float(nll_all.mean())

    result = {"split": args.split, "checkpoint": args.checkpoint,
              "n_windows": len(rows), "windows": rows}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out}")
    print_stats([result])


def print_stats(shards):
    rows = [w for s in shards for w in s["windows"]]
    n = len(rows)
    ckpt = shards[0]["checkpoint"]
    splits = sorted({s["split"] for s in shards})
    print(f"\n=== {ckpt} | {n} viz windows ({', '.join(splits)}) ===")
    for metric, desc in (
            ("score", "interleaved, score-token scope (the model's loss scope)"),
            ("all", "interleaved, all tokens incl. performance (never in loss)")):
        variants = ("c", "u") if metric == "score" else ("u",)
        for v in variants:
            vd = "constrained" if v == "c" else "unconstrained"
            gt = [w[f"{metric}_gt_{v}"] for w in rows]
            gen = [w[f"{metric}_gen_{v}"] for w in rows]
            gt_m, gen_m = sum(gt) / n, sum(gen) / n
            gt_higher = sum(1 for a, b in zip(gt, gen) if a > b)
            print(f"\n--- {desc} [{vd}] ---")
            print(f"  mean NLL/token : GT {gt_m:.4f}   gen {gen_m:.4f}")
            print(f"  PPL            : GT {math.exp(gt_m):.3f}   "
                  f"gen {math.exp(gen_m):.3f}")
            print(f"  GT higher than gen in {gt_higher}/{n} windows")
    gt_m = sum(w["score_gt_c"] for w in rows) / n
    gen_m = sum(w["score_gen_c"] for w in rows) / n
    which = "GROUND-TRUTH" if gt_m > gen_m else "GENERATED"
    print(f"\n>>> Under the masked-40k model, the {which} interleaved "
          f"sequences have higher perplexity "
          f"(score-token scope, constrained: GT {math.exp(gt_m):.3f} vs "
          f"gen {math.exp(gen_m):.3f}).")
    per = sorted(rows, key=lambda w: w["key"])
    print("\nper-window (score-token, constrained): key  GT-PPL  gen-PPL")
    for w in per:
        print(f"  {w['key']}: {math.exp(w['score_gt_c']):9.3f}  "
              f"{math.exp(w['score_gen_c']):9.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=list(SPLIT_FILES))
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--payload", default=DEFAULT_PAYLOAD)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--merge", nargs="*", default=None)
    args = ap.parse_args()
    if args.merge is not None:
        paths = []
        for pat in args.merge:
            paths.extend(glob.glob(pat) if any(c in pat for c in "*?[") else [pat])
        shards = [json.load(open(p)) for p in sorted(set(paths))]
        print_stats(shards)
        return
    if not args.split or not args.output:
        raise SystemExit("need --split and --output (or --merge)")
    run_split(args)


if __name__ == "__main__":
    main()
