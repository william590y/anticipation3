"""
Write a single opening-window packed line, run inference.py on it, print paths
and short event previews from input_performance.mid, ground_truth_score.mid,
and output_score.mid (same decode as anticipation.convert.midi_to_events).

Usage (repo root):
  python scripts/generate_opening_inference_midi.py --checkpoint checkpoint-20000
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from anticipation.convert import midi_to_events


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
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--ground-truth-score-tokens-to-feed",
        type=int,
        default=0,
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

    line_path = ROOT / "data" / "single_opening_window_for_inference.txt"
    line_path.parent.mkdir(parents=True, exist_ok=True)
    token_str = " ".join(str(t) for t in packed)
    line_path.write_text(f"{token_str} | \n", encoding="utf-8")
    print(f"Wrote opening window line ({len(packed)} tokens): {line_path}")
    print(f"  piece: {row['midi_performance']}")

    out_base = ROOT / "autoregressive_inference_results" / "opening_window_only"
    cmd = [
        sys.executable,
        str(ROOT / "inference.py"),
        "--checkpoint",
        str(ROOT / args.checkpoint),
        "--config-source",
        str(ROOT / args.config_source),
        "--test-file",
        str(line_path),
        "--num-examples",
        "1",
        "--seed",
        str(args.seed),
        "--output-base",
        str(out_base),
        "--ground-truth-score-tokens-to-feed",
        str(args.ground_truth_score_tokens_to_feed),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    from inference import checkpoint_label

    ck_label = checkpoint_label(str(ROOT / args.checkpoint))
    run_dir = out_base / ck_label / f"sample1_seed{args.seed}"
    seq_dirs = sorted(p for p in run_dir.glob("sequence_*") if p.is_dir())
    if not seq_dirs:
        print(f"No sequence_* under {run_dir}")
        sys.exit(1)
    seq_dir = seq_dirs[0]
    pred_mid = seq_dir / "output_score.mid"
    perf_mid = seq_dir / "input_performance.mid"
    gt_mid = seq_dir / "ground_truth_score.mid"
    print("\nMIDI outputs (open these in a DAW / MuseScore):")
    print(f"  input performance: {perf_mid}")
    print(f"  ground-truth score: {gt_mid}")
    print(f"  predicted score:    {pred_mid}")

    _print_midi_preview("Input performance (from packed window)", perf_mid)
    _print_midi_preview("Ground-truth score (from packed window)", gt_mid)
    _print_midi_preview("Predicted score (model)", pred_mid)


if __name__ == "__main__":
    main()
