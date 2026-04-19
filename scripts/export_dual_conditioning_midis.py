"""
Export two MIDI triplets (input_performance, ground_truth_score, output_score)
for the *same* ASAP piece and the *same* opening-window length (body note count):

  1) packed_inference_opening/
     Conditioning: opening packed token line (tokenize-asap-sliding start_idx=0).
     Generation: inference.autoregressive_generate_score (same as inference.py).

  2) asap_native_opening_first_window/
     Conditioning: preprocess_asap_piece (performance controls + full-score GT
     from MIDI, as in evaluate_muster_asap.py). Generation is truncated to the
     same number of perf-note cycles as the opening packed window (max_notes).

  Ground-truth MIDI differs between folders on purpose:
    - Folder 1: aligned score triplets embedded in the packed window (training target).
    - Folder 2: full normalized score from build_full_normalized_score_triplets (MUSTER-style GT).

Usage (repo root):
  python scripts/export_dual_conditioning_midis.py --checkpoint checkpoint-20000
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

from anticipation.packed_sequence import PREFIX_CONTROLS, extract_packed_components
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
        help="ASAP piece index (sorted list with all four files); default 10 for a non-opening-Bach piece",
    )
    parser.add_argument(
        "--ground-truth-score-tokens-to-feed",
        type=int,
        default=0,
        help="Packed / inference branch only",
    )
    parser.add_argument(
        "--ground-truth-score-notes-to-feed",
        type=int,
        default=0,
        help="ASAP branch only",
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
    row, filegroup = rows_ok[args.piece_index]

    packed, err = cmp.build_opening_packed_tokens(filegroup)
    if packed is None:
        print(err)
        sys.exit(1)
    body_slots = (len(packed) - ALTERNATING_START) // 6
    opening_control_count = PREFIX_CONTROLS + body_slots

    piece_info = cmp.piece_info_for_row(row)
    pre = preprocess_asap_piece(piece_info, gt_score_source="midi")
    if pre.get("error"):
        print(pre["error"])
        sys.exit(1)
    controls = pre["control_triplets"]
    gt_full = pre["gt_score_triplets"]

    safe = row["midi_performance"].replace("/", "_").replace("\\", "_")
    base = ROOT / "autoregressive_inference_results" / "dual_conditioning_midis" / safe
    dir_a = base / "1_packed_inference_opening"
    dir_b = base / "2_asap_native_opening_first_window"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    ckpt = ROOT / args.checkpoint
    print(f"Piece: {row['midi_performance']}")
    print(f"  Body perf-note cycles (both branches): {body_slots}")
    print(f"  Export base: {base}")

    print(f"\nLoading {ckpt} ...")
    model, device = load_model(str(ckpt), config_source=str(ROOT / args.config_source))

    # --- Branch 1: packed line + inference.py generator ---
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
    save_midi(normalize_triplet_times(pred_packed), dir_a / "output_score.mid")

    # --- Branch 2: MIDI preprocess + evaluate_muster_asap generator (first window only) ---
    pred_asap, _ = autoregressive_generate_from_controls(
        model,
        controls,
        gt_full,
        device,
        temperature=0.0,
        ground_truth_score_notes_to_feed=args.ground_truth_score_notes_to_feed,
        max_notes=body_slots,
    )
    controls_opening = controls[: min(len(controls), opening_control_count)]
    gt_export = [t for t in gt_full if t[2] != REST]

    save_midi(
        normalize_triplet_times(controls_opening),
        dir_b / "input_performance.mid",
    )
    save_midi(normalize_triplet_times(gt_export), dir_b / "ground_truth_score.mid")
    save_midi(normalize_triplet_times(pred_asap), dir_b / "output_score.mid")

    readme = base / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Two conditioning / generation branches for the same piece.",
                "",
                "1) 1_packed_inference_opening/",
                "   Input + GT score: extracted from the opening PACKED token window",
                "   (same object as tokenize-asap-sliding.py start_idx==0).",
                "   Output: inference.autoregressive_generate_score (inference.py path).",
                "",
                "2) 2_asap_native_opening_first_window/",
                "   Input: first N performance control triplets from preprocess_asap_piece,",
                f"   N = min(len(controls), {opening_control_count}) (opening window control span).",
                "   GT score: full normalized score from MIDI (REST notes removed for export),",
                "   as used by evaluate_muster_asap.py / MUSTER-style evaluation.",
                "   Output: autoregressive_generate_from_controls, same as evaluate_muster_asap,",
                f"   stopped after max_notes={body_slots} (same body length as opening packed window).",
                "",
                "GT MIDI intentionally differs: (1) aligned window score vs (2) full score grid.",
            ]
        ),
        encoding="utf-8",
    )

    print("\n--- (1) Packed opening + inference.py ---")
    for name in ("input_performance.mid", "ground_truth_score.mid", "output_score.mid"):
        p = dir_a / name
        print(f"  {p}")
    print("\n--- (2) ASAP preprocess + evaluate_muster_asap generator (first window) ---")
    for name in ("input_performance.mid", "ground_truth_score.mid", "output_score.mid"):
        p = dir_b / name
        print(f"  {p}")
    print(f"\n{readme}")


if __name__ == "__main__":
    main()
