"""Fixed-MUSTER on full-song rollouts — producer/consumer version.

Stage 1 (producers): one task per song builds the GT-side MUSTER artifacts
(gt.xml -> gt_hmm.txt, gt_fmt3x.txt) into a shared cache, once.
Stage 2 (consumers): one task per (song, system) pair pulls from the queue,
symlinks the cached GT artifacts, and runs only the prediction-side steps
(est fmt3x/spr -> match -> realign -> ScoreMatchEvaluation_VoicePlus).
Tasks are ordered largest-song-first so the Ballades never straggle.

Sources: everything in nbest/eval_fullsong_regimes.SOURCES plus the GT
self-comparison floor. Same debugged binaries as the windowed study.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_muster import get_muster_exe, triplets_to_musicxml   # noqa: E402
from nbest.eval_muster_windows import KEYS, dicts_to_triplets      # noqa: E402
from nbest.eval_fullsong_regimes import SOURCES                    # noqa: E402
from nbest.eval_muster_fullsong import quarters_to_dicts           # noqa: E402

CACHE = ROOT / "nbest_data/muster_gt_cache"


def run(cmd, cwd):
    r = subprocess.run([get_muster_exe(cmd[0]), *cmd[1:]], cwd=str(cwd),
                       capture_output=True, text=True)
    return r if r.returncode == 0 else None


def produce_gt(job):
    stem, gt_trips = job
    d = CACHE / stem
    d.mkdir(parents=True, exist_ok=True)
    if (d / "gt_fmt3x.txt").exists() and (d / "gt_hmm.txt").exists():
        return stem, True
    if not triplets_to_musicxml(gt_trips, str(d / "gt.xml"), beat_seconds=0.5):
        return stem, False
    ok = (run(("MusicXMLToHMM", "gt.xml", "gt_hmm.txt"), d)
          and run(("MusicXMLToFmt3x", "gt.xml", "gt_fmt3x.txt"), d))
    return stem, bool(ok)


def consume_pair(job):
    stem, name, trips = job
    out = {"key": stem, "name": name}
    gt_dir = CACHE / stem
    with tempfile.TemporaryDirectory(prefix="muster_pc_") as td:
        wd = Path(td)
        for f in ("gt_hmm.txt", "gt_fmt3x.txt"):
            os.symlink(gt_dir / f, wd / f)
        if not trips or not triplets_to_musicxml(trips, str(wd / "est.xml"),
                                                 beat_seconds=0.5):
            out["metrics"] = None
            return out
        steps = [
            ("MusicXMLToFmt3x", "est.xml", "est_fmt3x.txt"),
            ("Fmt3xToSpr", "est_fmt3x.txt", "est_spr.txt"),
            ("ScorePerfmMatcher", "gt_hmm.txt", "est_spr.txt",
             "est_pre_match.txt", "0.01"),
            ("ErrorDetection", "gt_fmt3x.txt", "gt_hmm.txt",
             "est_pre_match.txt", "est_err_match.txt"),
            ("RealignmentMOHMM", "gt_fmt3x.txt", "gt_hmm.txt",
             "est_err_match.txt", "est_auto_match.txt", "0.3"),
        ]
        for c in steps:
            if run(c, wd) is None:
                out["metrics"] = None
                return out
        r = run(("ScoreMatchEvaluation_VoicePlus", "gt_fmt3x.txt",
                 "est_fmt3x.txt", "est_auto_match.txt",
                 "est_err_detail.txt", "-1"), wd)
        if r is None or ":" not in r.stdout.strip():
            out["metrics"] = None
            return out
        v = r.stdout.strip().split(":")[-1].strip().split()
        if len(v) < 8:
            out["metrics"] = None
            return out
        out["metrics"] = dict(zip(
            ("pitch_error_rate", "missing_note_rate", "extra_note_rate",
             "onset_time_error_rate", "offset_time_error_rate",
             "mean_error_rate", "voice_error_rate",
             "mean_error_rate_with_voice"), map(float, v[:8])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    ap.add_argument("--out", default="nbest_data/muster_fixed_fullsong_test.json")
    a = ap.parse_args()

    gt_dir = ROOT / "fullsong_rollouts/test/slide"
    songs, pairs = [], []
    for f in sorted(gt_dir.glob("*.json")):
        gt_notes = json.load(open(f))["gt"]
        gt_trips = dicts_to_triplets(gt_notes)
        songs.append((f.stem, gt_trips))
        pairs.append((f.stem, "self", gt_trips, len(gt_trips)))
        for name, kind, dd in SOURCES:
            pf = ROOT / dd / f.name
            if not pf.exists():
                continue
            r = json.load(open(pf))
            notes = (r["pred"] if kind == "units"
                     else quarters_to_dicts(r["notes"], float(r["qpb"])))
            pairs.append((f.stem, name, dicts_to_triplets(notes),
                          len(gt_trips)))
    pairs.sort(key=lambda p: -p[3])
    pairs = [(s, n, t) for s, n, t, _ in pairs]
    print(f"{len(songs)} songs -> GT producers; {len(pairs)} consumer pairs; "
          f"procs={a.procs}", flush=True)

    with Pool(a.procs) as pool:
        ok = dict(pool.imap_unordered(produce_gt, songs, chunksize=1))
        bad = [s for s, v in ok.items() if not v]
        if bad:
            print(f"GT production failed for {len(bad)}: {bad[:5]}")
        pairs = [p for p in pairs if ok.get(p[0])]
        recs, done = [], 0
        for r in pool.imap_unordered(consume_pair, pairs, chunksize=1):
            recs.append(r)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(pairs)}", flush=True)

    by = defaultdict(list)
    for r in recs:
        if r["metrics"]:
            by[r["name"]].append(r["metrics"])
    names = ["self"] + [s[0] for s in SOURCES]
    print("\nsource     " + "".join(f"{k.split('_')[0]:>9}" for k in KEYS))
    for name in names:
        vals = by.get(name, [])
        if not vals:
            print(f"{name:10} (no results)")
            continue
        print(f"{name:10} "
              + "".join(f"{np.mean([v[k] for v in vals]):>9.3f}" for k in KEYS)
              + f"   (n={len(vals)})")
    json.dump({"records": recs}, open(ROOT / a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
