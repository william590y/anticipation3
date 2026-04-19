"""
Check that ASAP-native generation with max_notes=body_slots matches the prefix of
a full-piece run (same pred_score_triplets for the first body_slots steps), and
report onset monotonicity in generation order (non-REST rows only).

Usage (repo root):
  python scripts/verify_asap_native_prefix_parity.py --checkpoint checkpoint-20000
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from anticipation.vocab import REST, TIME_OFFSET
from evaluate_muster import load_model
from evaluate_muster_asap import autoregressive_generate_from_controls, preprocess_asap_piece
from inference import ALTERNATING_START


def _load_compare():
    path = ROOT / "scripts" / "compare_opening_rollout.py"
    spec = importlib.util.spec_from_file_location("cmp", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _onset_bins(triplet) -> int:
    return int(triplet[0]) - TIME_OFFSET


def _monotonic_non_rest_violations(pred) -> list[tuple[int, int, int]]:
    """(row_index, previous_non_rest_onset_bins, current_onset_bins) for each drop."""
    prev_onset = None
    out: list[tuple[int, int, int]] = []
    for i, t in enumerate(pred):
        if len(t) < 3 or int(t[2]) == REST:
            continue
        o = _onset_bins(t)
        if prev_onset is not None and o < prev_onset:
            out.append((i, prev_onset, o))
        prev_onset = o
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint-20000")
    parser.add_argument("--config-source", default="checkpoint-20000")
    parser.add_argument("--piece-index", type=int, default=10)
    parser.add_argument("--ground-truth-score-notes-to-feed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    cmp = _load_compare()
    df = pd.read_csv(cmp.ASAP_META_CSV)
    rows_ok = []
    for _, row in df.iterrows():
        fg = cmp.filegroup_for_row(row)
        if fg is not None:
            rows_ok.append((row, fg))
    rows_ok.sort(key=lambda x: str(x[0]["midi_performance"]))
    row, filegroup = rows_ok[args.piece_index]

    packed, err = cmp.build_opening_packed_tokens(filegroup)
    if packed is None:
        print(err)
        sys.exit(1)
    body_slots = (len(packed) - ALTERNATING_START) // 6

    piece_info = cmp.piece_info_for_row(row)
    pre = preprocess_asap_piece(piece_info, gt_score_source="midi")
    if pre.get("error"):
        print(pre["error"])
        sys.exit(1)
    controls = pre["control_triplets"]
    gt_full = pre["gt_score_triplets"]

    ckpt = ROOT / args.checkpoint
    print(f"Piece: {row['midi_performance']}")
    print(f"  body_slots (max_notes for truncated run): {body_slots}")
    print(f"  len(controls): {len(controls)}")

    model, device = load_model(str(ckpt), config_source=str(ROOT / args.config_source))

    pred_short, stats_s = autoregressive_generate_from_controls(
        model,
        controls,
        gt_full,
        device,
        temperature=args.temperature,
        ground_truth_score_notes_to_feed=args.ground_truth_score_notes_to_feed,
        max_notes=body_slots,
    )
    pred_full, stats_f = autoregressive_generate_from_controls(
        model,
        controls,
        gt_full,
        device,
        temperature=args.temperature,
        ground_truth_score_notes_to_feed=args.ground_truth_score_notes_to_feed,
    )

    n = len(pred_short)
    print(f"\nTruncated pred len: {n}, full pred len: {len(pred_full)}")
    print(f"Truncated num_window_resets: {stats_s['num_window_resets']}")
    print(f"Full num_window_resets: {stats_f['num_window_resets']}")

    prefix = pred_full[:n]
    mismatches = []
    for i in range(n):
        a, b = pred_short[i], prefix[i]
        if list(a) != list(b):
            mismatches.append((i, a, b))

    if not mismatches:
        print("\nOK: first-window pred_score_triplets are byte-identical (truncated vs full prefix).")
    else:
        print(f"\nMISMATCH: {len(mismatches)} triplet(s) differ in first {n} steps.")
        for i, a, b in mismatches[:8]:
            print(f"  idx {i}: short={a} full_prefix={b}")
        if len(mismatches) > 8:
            print(f"  ... and {len(mismatches) - 8} more")

    viol = _monotonic_non_rest_violations(pred_full)
    print(
        f"\nNon-REST onset monotonicity (generation order, full run): "
        f"{len(viol)} violation(s) where a non-REST onset is strictly less than the prior non-REST onset."
    )
    if viol:
        print("  First few (row_idx, prev_onset_bins, curr_onset_bins):")
        for i, prev_o, curr_o in viol[:12]:
            print(f"    row {i}: {prev_o} -> {curr_o} (backwards by {prev_o - curr_o})")


if __name__ == "__main__":
    main()
