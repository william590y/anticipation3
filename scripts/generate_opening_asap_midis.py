"""
Same piece + opening horizon as generate_opening_inference_midi.py, but generation
uses evaluate_muster_asap.autoregressive_generate_from_controls (same as full
evaluate_muster_asap.py, with max_notes set to one packed-window body length).

Writes input_performance.mid, ground_truth_score.mid, output_score.mid using the
same packing-derived perf/GT as inference (so the three inputs match that script),
and ASAP predictions for the opening body. Prints midi_to_events previews like
generate_opening_inference_midi.py.

Usage (repo root):
  python scripts/generate_opening_asap_midis.py --checkpoint checkpoint-20000
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_muster import load_model
from evaluate_muster_asap import autoregressive_generate_from_controls, preprocess_asap_piece
from inference import (
    ALTERNATING_START,
    extract_components,
    normalize_triplet_times,
    raw_triplets_to_event_triplets,
    save_midi,
)

from anticipation.convert import midi_to_events

import pandas as pd


def _print_midi_preview(title: str, mid_path: Path, max_triplets: int = 12) -> None:
    print(f"\n{title}")
    print(f"  path: {mid_path}")
    if not mid_path.exists():
        print("  (file missing)")
        return
    ev = midi_to_events(str(mid_path), quantize=False)
    n_tok = min(len(ev), max_triplets * 3)
    print(f"  {len(ev)} event tokens; first {n_tok // 3} triplets (time, dur, pitch):")
    for i in range(0, n_tok, 3):
        if i + 2 < len(ev):
            print(f"    {ev[i]}, {ev[i + 1]}, {ev[i + 2]}")


def _load_compare_module():
    path = ROOT / "scripts" / "compare_opening_rollout.py"
    spec = importlib.util.spec_from_file_location("compare_opening_rollout", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint-20000")
    parser.add_argument("--config-source", default="checkpoint-20000")
    parser.add_argument("--piece-index", type=int, default=0)
    parser.add_argument(
        "--ground-truth-score-notes-to-feed",
        type=int,
        default=0,
        help="Same flag semantics as evaluate_muster_asap.py",
    )
    args = parser.parse_args()

    cmp = _load_compare_module()
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
    perf_raw, gt_score_packed = extract_components(packed, ALTERNATING_START)

    piece_info = cmp.piece_info_for_row(row)
    pre = preprocess_asap_piece(piece_info, gt_score_source="midi")
    if pre.get("error"):
        print(pre["error"])
        sys.exit(1)
    controls = pre["control_triplets"]
    gt_scores = pre["gt_score_triplets"]

    ckpt = ROOT / args.checkpoint
    print(f"Piece: {row['midi_performance']}")
    print(f"  Opening body perf-note cycles: {body_slots} (packed len {len(packed)})")
    print(f"  Controls in piece: {len(controls)}, GT score triplets: {len(gt_scores)}")

    print(f"Loading model {ckpt} ...")
    model, device = load_model(str(ckpt), config_source=str(ROOT / args.config_source))

    pred_asap, _stats = autoregressive_generate_from_controls(
        model,
        controls,
        gt_scores,
        device,
        temperature=0.0,
        ground_truth_score_notes_to_feed=args.ground_truth_score_notes_to_feed,
        max_notes=body_slots,
    )

    safe = row["midi_performance"].replace("/", "_").replace("\\", "_")
    out_dir = (
        ROOT
        / "autoregressive_inference_results"
        / "asap_opening_window_only"
        / Path(args.checkpoint).name
        / safe
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    perf_midi_triplets = normalize_triplet_times(raw_triplets_to_event_triplets(perf_raw))
    gt_midi_triplets = normalize_triplet_times(gt_score_packed)
    pred_midi_triplets = normalize_triplet_times(pred_asap)

    inp_path = out_dir / "input_performance.mid"
    gt_path = out_dir / "ground_truth_score.mid"
    pred_path = out_dir / "output_score.mid"

    save_midi(perf_midi_triplets, inp_path)
    save_midi(gt_midi_triplets, gt_path)
    save_midi(pred_midi_triplets, pred_path)

    print("\nMIDI outputs (same layout as opening inference script):")
    print(f"  input performance: {inp_path}")
    print(f"  ground-truth score: {gt_path}")
    print(f"  predicted score:    {pred_path}")

    _print_midi_preview("Input performance (packed opening window)", inp_path)
    _print_midi_preview("Ground-truth score (packed opening window)", gt_path)
    _print_midi_preview("Predicted score (evaluate_muster_asap path)", pred_path)


if __name__ == "__main__":
    main()
