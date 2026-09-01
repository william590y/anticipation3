#!/usr/bin/env python
"""Double application of the alignment model on the visualizer windows.

Pass 1 greedily decodes the score from the window's performance controls (the
table's 'ours' rollout). Pass 2 then feeds pass 1's generated score BACK IN as
the performance stream: the 138 generated notes (sorted by onset) become the
controls of a fresh packed window via ``tokens_from_controls``, and the model
greedily decodes again.

With 138 controls the packed layout yields 32 prefix pairs + 106 body slots,
so pass 2 predicts 106 notes where pass 1 predicted 138. To separate the
fewer-slots handicap from the re-application effect, a control row scores
pass 1's first 106 notes (same onset-sorted truncation pass 2's controls see)
against the same ground truth.

All rows are scored against the window's full real GT note set under the F1
table's three criteria (``compute_f1.score_notes``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from anticipation.config import CONTEXT_SIZE  # noqa: E402
from evaluate_muster import load_model  # noqa: E402
from onpolicy_rollout import rollout_score_slots, score_token_positions  # noqa: E402
from precompute_beams import note_from_tokens  # noqa: E402
from precompute_visualizer import tokens_from_controls  # noqa: E402
from compute_f1 import score_notes  # noqa: E402
from compute_sequence_ppl import (control_notes_for_variant,  # noqa: E402
                                  gt_notes_for_variant, load_payload)


@torch.inference_mode()
def greedy_notes(ft, tokens, device, autocast):
    """Greedy constrained rollout of a packed window -> slot-ordered note dicts."""
    window = torch.tensor([tokens], dtype=torch.long, device=device)
    positions = score_token_positions(len(tokens), device=device)
    out = rollout_score_slots(
        ft, window, temperature=0.0, constrain=True, collect_logprobs=False,
        collect_gt_ce=False, autocast_ctx=autocast)
    toks = out["rolled"][0, positions].tolist()
    notes = []
    for k in range(len(toks) // 3):
        n = note_from_tokens(toks[3 * k], toks[3 * k + 1], toks[3 * k + 2])
        if n is not None:
            notes.append({"t": int(n["t"]), "d": int(n["d"]), "p": int(n["p"])})
    return notes


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--checkpoint", default="run_paper_split_v2/checkpoint-2500")
    ap.add_argument("--variant", default="raw", choices=["filtered", "raw"])
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast = lambda: torch.autocast(  # noqa: E731
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available())

    payload, _ = load_payload(args.data)
    keys = [k for i, k in enumerate(payload["example_order"])
            if i % args.num_shards == args.shard_index]

    ft, _ = load_model(args.checkpoint)
    ft = ft.to(device).eval()

    results = {"checkpoint": args.checkpoint, "variant": args.variant,
               "examples": {}}
    t0 = time.time()
    for ki, key in enumerate(keys):
        ex = payload["examples"][key]
        controls = control_notes_for_variant(ex, args.variant)
        tokens1 = tokens_from_controls(controls, CONTEXT_SIZE - 4)
        s1 = greedy_notes(ft, tokens1, device, autocast)

        # Same GT source as compute_f1.main scores the table with.
        gt = [{"t": int(n["t"]), "d": int(n["d"]), "p": int(n["p"])}
              for n in (ex.get("gt_score") or [])]

        # Pass 2: generated score as the performance stream.
        s1_sorted = sorted(s1, key=lambda n: (n["t"], n["p"]))
        tokens2 = tokens_from_controls(s1_sorted, CONTEXT_SIZE - 4)
        s2 = greedy_notes(ft, tokens2, device, autocast)
        s1_trunc = s1_sorted[:len(s2)]
        # Does pass 2 just copy its (score-shaped) input controls? Slot k of
        # pass 2 aligns with control k = s1_sorted[k].
        copies = sum(1 for a, b in zip(s2, s1_trunc)
                     if (a["t"], a["d"], a["p"]) == (b["t"], b["d"], b["p"]))
        copy_rate = copies / max(len(s2), 1)

        entry = {
            "copy_rate": copy_rate,
            "n_slots_pass1": len(s1), "n_slots_pass2": len(s2), "n_gt": len(gt),
            "f1_pass1": score_notes(s1, gt),
            "f1_pass1_trunc": score_notes(s1_trunc, gt),
            "f1_pass2": score_notes(s2, gt),
            "pred_pass1": s1, "pred_pass2": s2,
        }
        results["examples"][key] = entry
        c3 = lambda d: d["onset_pitch_tol1"]["f1"]  # noqa: E731
        print(f"{key}: col3 pass1={c3(entry['f1_pass1']):.3f} "
              f"pass1_trunc={c3(entry['f1_pass1_trunc']):.3f} "
              f"pass2={c3(entry['f1_pass2']):.3f} "
              f"copy={copy_rate:.2f} "
              f"(slots {len(s1)}->{len(s2)}, gt {len(gt)}) "
              f"[{ki + 1}/{len(keys)}  {time.time() - t0:.0f}s]", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
