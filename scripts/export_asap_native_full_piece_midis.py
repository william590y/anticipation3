"""
Two export folders (same layout idea as export_dual_conditioning_midis.py):

  1) 1_packed_inference_opening_first_window/
     Single packed window at sliding start_idx=0 (same object as
     export_dual_conditioning_midis.py → 1_packed_inference_opening/).
     inference.autoregressive_generate_score — not stitched multi-window packed.

  2) 2_asap_native_full_piece_rollout/
     preprocess_asap_piece + autoregressive_generate_from_controls with no max_notes
     (full piece; same stack as dual’s ASAP branch, without the opening-only cap).

Base directory (same as dual script):
  autoregressive_inference_results/dual_conditioning_midis/<safe>/

Usage (repo root):
  python scripts/export_asap_native_full_piece_midis.py --checkpoint checkpoint-20000
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

from anticipation.packed_sequence import extract_packed_components
from anticipation.vocab import REST

from evaluate_muster import load_model
from evaluate_muster_asap import autoregressive_generate_from_controls, preprocess_asap_piece
from inference import (
    ALTERNATING_START,
    autoregressive_generate_score,
    extract_components,
    normalize_triplet_times,
    raw_triplets_to_event_triplets,
    save_midi,
)


def _load_compare():
    path = ROOT / "scripts" / "compare_opening_rollout.py"
    spec = importlib.util.spec_from_file_location("cmp", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint-20000")
    parser.add_argument("--config-source", default="checkpoint-20000")
    parser.add_argument(
        "--piece-index",
        type=int,
        default=10,
        help="ASAP piece index (sorted list with all four files)",
    )
    parser.add_argument(
        "--ground-truth-score-tokens-to-feed",
        type=int,
        default=1,
        help=(
            "Packed / inference.py branch only; default 1 matches inference.py CLI. "
            "Use 0 to match export_dual_conditioning_midis.py default."
        ),
    )
    parser.add_argument(
        "--ground-truth-score-notes-to-feed",
        type=int,
        default=0,
        help="ASAP-native branch only",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--gt-score-source",
        choices=("midi", "xml", "auto"),
        default="midi",
        help="Same as preprocess_asap_piece / evaluate_muster_asap",
    )
    args = parser.parse_args()

    cmp = _load_compare()
    df = pd.read_csv(cmp.ASAP_META_CSV)
    rows_ok = []
    for _, row in df.iterrows():
        fg = cmp.filegroup_for_row(row)
        if fg is not None:
            rows_ok.append((row, fg))
    rows_ok.sort(key=lambda x: str(x[0]["midi_performance"]))
    if args.piece_index < 0 or args.piece_index >= len(rows_ok):
        print(f"--piece-index must be in 0..{len(rows_ok) - 1}")
        sys.exit(1)
    row, filegroup = rows_ok[args.piece_index]

    packed, err = cmp.build_opening_packed_tokens(filegroup)
    if packed is None:
        print(err)
        sys.exit(1)
    body_slots = (len(packed) - ALTERNATING_START) // 6

    piece_info = cmp.piece_info_for_row(row)
    pre = preprocess_asap_piece(piece_info, gt_score_source=args.gt_score_source)
    if pre.get("error"):
        print(pre["error"])
        sys.exit(1)
    controls = pre["control_triplets"]
    gt_full = pre["gt_score_triplets"]

    safe = row["midi_performance"].replace("/", "_").replace("\\", "_")
    base = ROOT / "autoregressive_inference_results" / "dual_conditioning_midis" / safe
    dir_a = base / "1_packed_inference_opening_first_window"
    dir_b = base / "2_asap_native_full_piece_rollout"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    ckpt = ROOT / args.checkpoint
    print(f"Piece: {row['midi_performance']}")
    print(
        f"  Packed branch: opening window only (start_idx=0), body perf cycles: {body_slots}"
    )
    print(f"  Performance controls (full): {len(controls)}")
    print(f"  Export base: {base}")

    print(f"\nLoading {ckpt} ...")
    model, device = load_model(str(ckpt), config_source=str(ROOT / args.config_source))

    # --- (1) Packed opening window only (start_idx=0; same packed line as dual branch 1) ---
    perf_raw, gt_packed = extract_components(packed, ALTERNATING_START)
    pred_ctx, _ = autoregressive_generate_score(
        model,
        packed,
        ALTERNATING_START,
        device,
        constrain_score_tokens=True,
        ground_truth_score_tokens_to_feed=args.ground_truth_score_tokens_to_feed,
    )
    _, pred_packed = extract_packed_components(
        pred_ctx, ALTERNATING_START, include_dummy_score=True
    )
    save_midi(
        normalize_triplet_times(raw_triplets_to_event_triplets(perf_raw)),
        dir_a / "input_performance.mid",
    )
    save_midi(normalize_triplet_times(gt_packed), dir_a / "ground_truth_score.mid")
    pred_notes = [t for t in pred_packed if t[2] != REST]
    if pred_notes:
        save_midi(normalize_triplet_times(pred_notes), dir_a / "output_score.mid")
    else:
        print(
            "  Warning: packed inference predicted only REST score slots; "
            "skipping 1_packed_inference_opening_first_window/output_score.mid "
            "(events_to_midi would receive no notes after unpad)."
        )

    # --- (2) ASAP preprocess + full-piece autoregressive_generate_from_controls ---
    pred_asap, stats = autoregressive_generate_from_controls(
        model,
        controls,
        gt_full,
        device,
        temperature=args.temperature,
        ground_truth_score_notes_to_feed=args.ground_truth_score_notes_to_feed,
    )
    gt_export = [t for t in gt_full if t[2] != REST]

    save_midi(
        normalize_triplet_times(controls),
        dir_b / "input_performance.mid",
    )
    save_midi(normalize_triplet_times(gt_export), dir_b / "ground_truth_score.mid")
    save_midi(normalize_triplet_times(pred_asap), dir_b / "output_score.mid")

    readme = base / "README_full_piece_midis.txt"
    readme.write_text(
        "\n".join(
            [
                "Two branches (compare generators side by side):",
                "",
                "1) 1_packed_inference_opening_first_window/",
                "   Same packed line as export_dual_conditioning_midis.py → 1_packed_inference_opening/",
                "   (tokenize-asap-sliding start_idx=0 only; one window, not full-piece packed).",
                "   inference.autoregressive_generate_score; GT is window-aligned score from the pack.",
                "   If the model predicts only REST score tokens, output_score.mid is omitted (MIDI",
                "   export would otherwise be empty after unpad).",
                "",
                "2) 2_asap_native_full_piece_rollout/",
                "   Same ASAP stack as dual branch 2, but full piece (no max_notes).",
                "   input_performance.mid = all controls; GT = full score (REST omitted in MIDI).",
                "",
                "GT MIDI differs between (1) and (2) on purpose (packed window vs full score).",
                "Temporal span of (1) is the opening context only; (2) is the whole performance.",
                "",
                f"piece_index: {args.piece_index}",
                f"opening_body_slots: {body_slots}",
                f"gt_score_source: {args.gt_score_source}",
                f"ground_truth_score_tokens_to_feed: {args.ground_truth_score_tokens_to_feed}",
                f"ground_truth_score_notes_to_feed: {args.ground_truth_score_notes_to_feed}",
                f"temperature: {args.temperature}",
                f"num_controls_used: {stats.get('num_controls_used')}",
                f"num_window_resets: {stats.get('num_window_resets')}",
            ]
        ),
        encoding="utf-8",
    )

    print("\n--- (1) Packed opening first window + inference.py ---")
    for name in ("input_performance.mid", "ground_truth_score.mid", "output_score.mid"):
        p = dir_a / name
        if p.exists():
            print(f"  {p}")
        else:
            print(f"  (missing) {p}")
    print("\n--- (2) ASAP native full piece ---")
    for name in ("input_performance.mid", "ground_truth_score.mid", "output_score.mid"):
        print(f"  {dir_b / name}")
    print(f"\n{readme}")


if __name__ == "__main__":
    main()
