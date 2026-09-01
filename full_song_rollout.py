#!/usr/bin/env python
"""Full-song sliding-window rollouts over the paper-split test/val performances.

Runs evaluate_muster_asap.autoregressive_generate_from_controls in
window_mode="slide" (see its docstring for the three subtleties) over every
performance in the requested split, conditioning on the RAW performance MIDI --
the same regime the papers transcribe -- and writes one JSON per performance:

  {perf_path, split, n_controls, gen_stats,
   pred: [{t,d,p}...],   # onset/dur in 10ms units on the stitched grid
   gt:   [{t,d,p}...]}   # the full aligned GT score, same units

File-gated and shardable: --shard-index/--num-shards, skips outputs that
already exist, so a preempted array task resumes at the next piece.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_muster_asap as ema                      # noqa: E402
from anticipation.vocab import (DUR_OFFSET, NOTE_OFFSET, REST,  # noqa: E402
                                TIME_OFFSET)


def split_perfs(split_file: str, want: str) -> set:
    tag = {"test": "=== TEST PERFORMANCES ===",
           "val": "=== VALIDATION PERFORMANCES ==="}[want]
    out, inside = set(), False
    for line in open(split_file, encoding="utf-8"):
        line = line.strip()
        if line == tag:
            inside = True
            continue
        if line.startswith("==="):
            inside = False
            continue
        if inside and line and not line.startswith("#"):
            out.add(line.lstrip("./"))
    if not out:
        raise SystemExit(f"no {want} performances found in {split_file}")
    return out


def units(trip):
    t, d, p = int(trip[0]), int(trip[1]), int(trip[2])
    return {"t": max(0, t - TIME_OFFSET), "d": max(0, d - DUR_OFFSET),
            "p": p - NOTE_OFFSET}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="run_paper_split_v2/checkpoint-2500")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--split-file", default="data/paper_split.txt")
    ap.add_argument("--window-mode", choices=["reset", "slide"], default="slide")
    ap.add_argument("--window-overlap", type=int, default=69)
    ap.add_argument("--overlap-source", choices=["pred", "gt", "gt_time"],
                    default="pred")
    ap.add_argument("--save-trace", action="store_true",
                    help="capture per-note entropy + top-5 alternatives")
    ap.add_argument("--pitch-force", action="store_true",
                    help="oracle diagnostic: substitute the aligned GT pitch "
                         "whenever the argmax pitch is wrong (outputs to "
                         "<mode>_pforce/)")
    ap.add_argument("--out-dir", default="fullsong_rollouts")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-notes", type=int, default=None)
    a = ap.parse_args()

    want = split_perfs(a.split_file, a.split)
    pieces = [p for p in ema.load_asap_metadata()
              if p["perf_path"].lstrip("./") in want]
    pieces.sort(key=lambda p: p["perf_path"])
    mine = pieces[a.shard_index::a.num_shards]
    mode_dir = a.window_mode + ("_pforce" if a.pitch_force else "") \
        + {"gt": "_gtctx", "gt_time": "_gttime"}.get(a.overlap_source, "")
    out_dir = Path(a.out_dir) / a.split / mode_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{a.split}: {len(pieces)} performances, shard "
          f"{a.shard_index}/{a.num_shards} -> {len(mine)}", flush=True)

    model, device = ema.load_model(a.checkpoint, config_source="auto")
    for pi in mine:
        name = pi["perf_path"].lstrip("./").replace("/", "__")[:-4] + ".json"
        out = out_dir / name
        if out.exists():
            print(f"  skip (exists): {name}", flush=True)
            continue
        info = ema.preprocess_asap_piece(pi, gt_score_source="midi")
        ctl, gt = info["control_triplets"], info["gt_score_triplets"]
        if len(ctl) < 1 or len(gt) < 5:
            print(f"  skip (too short): {name}", flush=True)
            continue
        pred, stats = ema.autoregressive_generate_from_controls(
            model, ctl, gt, device, temperature=0.0,
            ground_truth_score_notes_to_feed=0, max_notes=a.max_notes,
            window_mode=a.window_mode, window_overlap=a.window_overlap,
            pitch_force=a.pitch_force, overlap_source=a.overlap_source,
            capture_trace=a.save_trace)
        _trace = stats.pop("score_token_perplexity_trace", {})
        rec = {"perf_path": pi["perf_path"], "split": a.split,
               "pitch_force": a.pitch_force,
               "checkpoint": a.checkpoint, "window_mode": a.window_mode,
               "window_overlap": a.window_overlap,
               "n_controls": len(ctl), "gen_stats": stats,
               "pred": [units(t) for t in pred],
               "trace": ({k: _trace.get(k)
                          for k in ("time", "dur", "pitch", "H_time", "H_dur",
                                    "H_pitch", "alt_time", "alt_dur",
                                    "alt_pitch")}
                         if a.save_trace else None),
               "gt": [units(t) for t in gt if int(t[2]) != REST]}
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        tmp.rename(out)
        print(f"  wrote {name}: {len(rec['pred'])} pred / {len(rec['gt'])} gt "
              f"notes, {stats['num_window_resets']} boundaries", flush=True)
    print("SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
