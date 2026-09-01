#!/usr/bin/env python
"""fp32 vs bf16 vs fp16 greedy decode on the visualizer windows.

The benchmark agent measured bf16 costing ~9% relative F1 on 208 train/val
windows. That matters here because the F1 table mixes numerics: the 'ours'
greedy row comes from ``precompute_visualizer.py``, which uses NO autocast
(fp32), while every selection row's candidate pool comes from
``rerank_sample_viz.py`` / ``nbest/generate_nbest.py``, which decode under
bf16 autocast. This isolates the dtype effect on the SAME 24 windows with the
SAME metric, so the table can be read (and, for task 7, regenerated) knowing
what the numerics cost.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from anticipation.config import CONTEXT_SIZE  # noqa: E402
from compute_f1 import score_notes  # noqa: E402
from compute_sequence_ppl import control_notes_for_variant, load_payload  # noqa: E402
from evaluate_muster import load_model  # noqa: E402
from onpolicy_rollout import rollout_score_slots, score_token_positions  # noqa: E402
from precompute_beams import note_from_tokens  # noqa: E402
from precompute_visualizer import tokens_from_controls  # noqa: E402

CRITS = ("onset_pitch", "onset_pitch_dur", "onset_pitch_tol1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--checkpoint", default="run_paper_split_v2/checkpoint-2500")
    ap.add_argument("--variant", default="raw")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, _ = load_payload(a.data)
    keys = list(payload["example_order"])
    ft, _ = load_model(a.checkpoint)
    ft = ft.to(device).eval()

    modes = {
        "fp32 (no autocast)": lambda: contextlib.nullcontext(),
        "bf16 autocast": lambda: torch.autocast("cuda", dtype=torch.bfloat16),
        "fp16 autocast": lambda: torch.autocast("cuda", dtype=torch.float16),
    }
    results = {name: {c: [] for c in CRITS} for name in modes}
    same_as_fp32 = {name: 0 for name in modes}
    n_tok = 0

    with torch.inference_mode():
        for key in keys:
            ex = payload["examples"][key]
            controls = control_notes_for_variant(ex, a.variant)
            tokens = tokens_from_controls(controls, CONTEXT_SIZE - 4)
            window = torch.tensor([tokens], dtype=torch.long, device=device)
            positions = score_token_positions(len(tokens), device=device)
            gt = [{"t": int(n["t"]), "d": int(n["d"]), "p": int(n["p"])}
                  for n in (ex.get("gt_score") or [])]
            ref = None
            for name, ctx in modes.items():
                out = rollout_score_slots(
                    ft, window, temperature=0.0, constrain=True,
                    collect_logprobs=False, collect_gt_ce=False,
                    autocast_ctx=ctx)
                flat = out["rolled"][0, positions]
                if ref is None:
                    ref, n_tok = flat, n_tok + flat.numel()
                same_as_fp32[name] += int((flat == ref).sum())
                toks = flat.tolist()
                pred = [note_from_tokens(toks[3 * k], toks[3 * k + 1],
                                         toks[3 * k + 2])
                        for k in range(len(toks) // 3)]
                pred = [{"t": int(n["t"]), "d": int(n["d"]), "p": int(n["p"])}
                        for n in pred if n is not None]
                sc = score_notes(pred, gt)
                for c in CRITS:
                    results[name][c].append(sc[c]["f1"])
            print(f"{key}: " + "  ".join(
                f"{n.split()[0]}={100*results[n]['onset_pitch_tol1'][-1]:.1f}"
                for n in modes), flush=True)

    print(f"\n{len(keys)} windows, {n_tok} score tokens each mode")
    print(f"{'mode':22s} {'onset+pitch':>12s} {'+duration':>11s} "
          f"{'onset±1+pitch':>14s} {'tokens == fp32':>15s}")
    for name in modes:
        m = {c: 100 * sum(results[name][c]) / len(keys) for c in CRITS}
        print(f"{name:22s} {m['onset_pitch']:12.2f} {m['onset_pitch_dur']:11.2f} "
              f"{m['onset_pitch_tol1']:14.2f} "
              f"{100 * same_as_fp32[name] / max(n_tok, 1):14.1f}%")


if __name__ == "__main__":
    main()
