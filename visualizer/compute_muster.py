"""
Post-process visualizer/data.js: compute MUSTER (predicted-score vs ground-truth
score) similarity for every rollout variant already stored in the file.

Single-window scope, unlike evaluate_muster.py/inference.py's
``normalize_triplet_times`` (built for the multi-window/full-piece evaluators):
no independent re-anchoring is applied here. For one packed window, GT and
predicted score notes already share a single, non-negative time origin fixed
once at tokenization (tokenize-asap-sliding.py's per-window
``min_score_time_units``) -- the model predicts tokens directly on that same
axis, it never establishes its own. Re-deriving a second anchor per side (as
the multi-window evaluators do) is unnecessary here and is exactly what causes
that bug, so this script skips it entirely: notes are exported as-is.

GT and predicted notes are fed to MUSTER as plain note lists (not index-aligned
slot by slot) -- MUSTER's own matcher does the note-to-note alignment, so a
raw-rollout prediction attempting to "explain" a performer mistake with no true
score counterpart is correctly counted as an extra note rather than requiring
any manual slot bookkeeping here.

Run:
  python visualizer/compute_muster.py --data visualizer/data.js
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET  # noqa: E402
from evaluate_muster import (  # noqa: E402
    check_muster_installation,
    run_muster_evaluation,
    triplets_to_musicxml,
)

TARGET_BEAT_INTERVAL = 0.5  # matches the score's 0.5s beat grid (packed_sequence)
MODEL_KEYS = ("rollouts", "rollouts_lora")
VARIANTS = ("filtered", "filtered_seeded", "raw", "raw_seeded")


def notes_to_triplets(notes):
    return [
        [TIME_OFFSET + int(n["t"]), DUR_OFFSET + int(n["d"]), NOTE_OFFSET + int(n["p"])]
        for n in notes
        if n is not None
    ]


def compute_muster_for_rollout(gt_notes, pred_notes, work_dir):
    gt_triplets = notes_to_triplets(gt_notes)
    pred_triplets = notes_to_triplets(pred_notes)
    if len(gt_triplets) < 2 or len(pred_triplets) < 2:
        return None

    gt_xml = work_dir / "gt_score.xml"
    pred_xml = work_dir / "pred_score.xml"
    if not triplets_to_musicxml(gt_triplets, str(gt_xml), beat_seconds=TARGET_BEAT_INTERVAL):
        return None
    if not triplets_to_musicxml(pred_triplets, str(pred_xml), beat_seconds=TARGET_BEAT_INTERVAL):
        return None

    return run_muster_evaluation(gt_xml, pred_xml, "muster", work_dir)


def load_payload(path):
    text = Path(path).read_text(encoding="utf-8")
    prefix = "window.VISUALIZER_DATA = "
    if not text.startswith(prefix):
        raise ValueError(f"unexpected data.js format (missing '{prefix}' prefix): {path}")
    body = text[len(prefix):].rstrip()
    if body.endswith(";"):
        body = body[:-1]
    return json.loads(body)


def write_payload(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("window.VISUALIZER_DATA = ")
        json.dump(payload, handle)
        handle.write(";\n")


def main():
    global TARGET_BEAT_INTERVAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="visualizer/data.js")
    parser.add_argument(
        "--beat-seconds", type=float, default=TARGET_BEAT_INTERVAL,
        help="score beat grid in seconds (0.5 default; 0.48 for the b048 tokenization). "
             "Used as beat_seconds when engraving pred/gt triplets for MUSTER scoring, so "
             "it must match how --data was tokenized.",
    )
    args = parser.parse_args()
    TARGET_BEAT_INTERVAL = args.beat_seconds

    check_muster_installation()

    t_start = time.perf_counter()
    payload = load_payload(args.data)
    examples = payload["examples"]

    n_computed = n_skipped = 0
    with tempfile.TemporaryDirectory(prefix="muster_viz_") as tmp:
        tmp_path = Path(tmp)
        for wid, ex in examples.items():
            gt_notes = ex["gt_score"]
            for model_key in MODEL_KEYS:
                roots = ex.get(model_key)
                if not roots:
                    continue
                for variant in VARIANTS:
                    rollout = roots.get(variant)
                    if not rollout:
                        continue
                    work_dir = tmp_path / wid / model_key / variant
                    work_dir.mkdir(parents=True, exist_ok=True)
                    metrics = compute_muster_for_rollout(gt_notes, rollout["pred_score"], work_dir)
                    rollout["muster"] = metrics
                    if metrics is None:
                        n_skipped += 1
                        print(f"  {wid}/{model_key}/{variant}: MUSTER failed/skipped", flush=True)
                    else:
                        n_computed += 1
                        print(
                            f"  {wid}/{model_key}/{variant}: "
                            f"MER={metrics['mean_error_rate']:.2f}%  "
                            f"PER={metrics['pitch_error_rate']:.2f}%  "
                            f"OTER={metrics['onset_time_error_rate']:.2f}%  "
                            f"OFTER={metrics['offset_time_error_rate']:.2f}%",
                            flush=True,
                        )

    print(f"Computed {n_computed}, skipped {n_skipped} ({time.perf_counter() - t_start:.1f}s)")
    write_payload(args.data, payload)
    print(f"Updated {args.data}")


if __name__ == "__main__":
    main()
