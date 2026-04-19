"""
Full-piece generation + MUSTER (same path as evaluate_muster_asap.py) for exactly
one ASAP piece selected by --piece-index (sorted list of pieces with all four
tokenization files).

Writes under muster_evaluation_results/single_full_piece_rollout/<checkpoint>_<safe_name>/:
  - input is not written by MUSTER path; see README — perf/score MIDIs from
    evaluate_triplet_slice_with_muster: ground_truth_score.mid, output_score.mid
  - muster_metrics.json, XMLs, muster_work/

Usage:
  python scripts/run_full_asap_muster_one_piece.py --checkpoint checkpoint-20000 --piece-index 10
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

from evaluate_muster import check_muster_installation
from evaluate_muster_asap import evaluate_asap_muster, preprocess_asap_piece


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
        help="Index into sorted list of ASAP pieces with all four annotation files (default: 10, not 0)",
    )
    parser.add_argument(
        "--gt-score-source",
        choices=("midi", "xml", "auto"),
        default="midi",
    )
    parser.add_argument(
        "--ground-truth-score-notes-to-feed",
        type=int,
        default=0,
    )
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
    if args.piece_index < 0 or args.piece_index >= len(rows_ok):
        print(f"--piece-index must be in 0..{len(rows_ok) - 1}")
        sys.exit(1)

    row, _ = rows_ok[args.piece_index]
    piece_info = cmp.piece_info_for_row(row)
    print(f"Piece [{args.piece_index}]: {piece_info['perf_path']}")

    check_muster_installation()

    pre = preprocess_asap_piece(piece_info, gt_score_source=args.gt_score_source)
    if pre.get("error"):
        print(f"preprocess failed: {pre['error']}")
        sys.exit(1)

    safe = piece_info["perf_path"].replace("/", "_").replace("\\", "_")
    ck_name = Path(args.checkpoint).name
    # evaluate_asap_muster writes to output_dir / safe_name — do not bake safe into output_dir.
    output_dir = (
        ROOT
        / "muster_evaluation_results"
        / "single_full_piece_rollout"
        / f"{ck_name}_idx{args.piece_index}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    readme = output_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Full-piece evaluate_muster_asap.py rollout (all performance notes,",
                "KV window resets as in training eval).",
                "",
                f"piece_index: {args.piece_index}",
                f"perf_path: {piece_info['perf_path']}",
                f"checkpoint: {args.checkpoint}",
                f"gt_score_source: {args.gt_score_source}",
                f"ground_truth_score_notes_to_feed: {args.ground_truth_score_notes_to_feed}",
                f"temperature: {args.temperature}",
                "",
                "Per-piece outputs: subdirectory named after this piece (safe_name).",
                "MIDIs: ground_truth_score.mid, output_score.mid; MUSTER: muster_metrics.json, XMLs.",
            ]
        ),
        encoding="utf-8",
    )

    evaluate_asap_muster(
        str(ROOT / args.checkpoint),
        [pre],
        str(output_dir),
        config_source=str(ROOT / args.config_source),
        temperature=args.temperature,
        ground_truth_score_notes_to_feed=args.ground_truth_score_notes_to_feed,
        requested_gt_score_source=args.gt_score_source,
    )

    seq_dir = output_dir / safe
    print(f"\nDone. Output directory:\n  {output_dir}")
    print(f"Piece folder (MIDIs, MUSTER, metrics):\n  {seq_dir}")
    for name in (
        "ground_truth_score.mid",
        "output_score.mid",
        "muster_metrics.json",
        "ground_truth_score.xml",
        "output_score.xml",
    ):
        p = seq_dir / name
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    main()
