#!/usr/bin/env python
"""MUSTER (fixed) over the windowed test set: ours vs Beyer vs Zeng + the
self-comparison floor.

Runs the repaired MUSTER pipeline (offset-interpolation fix, alignment-rescue
pass, unison symmetry, claim-swap round -- see
MUSTER/Code/ScoreMatchEvaluation_VoicePlus_v220118.cpp FIX comments) on every
stride-150 test window, four comparisons per window: GT-vs-GT (the residual
floor), and GT vs each system. XML export uses the exact-grid
triplets_to_musicxml path -- the same converter the GT goes through, so both
sides share one representation (papers' bin dicts get vocab offsets added, the
only conversion step).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, REST, TIME_OFFSET  # noqa: E402
from evaluate_muster import run_muster_evaluation, triplets_to_musicxml    # noqa: E402
from onpolicy_rollout import score_token_positions                         # noqa: E402

KEYS = ("pitch_error_rate", "missing_note_rate", "extra_note_rate",
        "onset_time_error_rate", "offset_time_error_rate", "mean_error_rate")


def toks_to_triplets(flat):
    t = [int(x) for x in flat]
    return [(t[3*i], t[3*i+1], t[3*i+2]) for i in range(len(t)//3)
            if t[3*i+2] != REST and NOTE_OFFSET <= t[3*i+2]]


def dicts_to_triplets(notes):
    out = []
    for n in notes:
        if not n:
            continue
        # pitch 0 is paper2's UNSHIFTED <PAD> index leaking through the
        # window slicer (paper1's pads unbucket to -1 and are dropped there;
        # paper2's argmax has no shift). MIDI 0 also segfaults MUSTER's
        # MusicXMLToFmt3x via a negative octave -- it took out 830/1181
        # Beyer windows. Drop it as their own decoder's pad mask would.
        if int(n["p"]) < 1:
            continue
        out.append((TIME_OFFSET + max(0, int(n["t"])),
                    DUR_OFFSET + max(1, int(n["d"])),
                    NOTE_OFFSET + int(n["p"])))
    return out


def eval_window(job):
    key, gt_trips, systems = job
    out = {"key": key}
    with tempfile.TemporaryDirectory(prefix="muster_") as td:
        wd = Path(td)
        gt_xml = wd / "gt_side.xml"
        if not triplets_to_musicxml(gt_trips, str(gt_xml), beat_seconds=0.5):
            out["error"] = "gt xml export failed"
            return out
        for name, trips in [("self", gt_trips)] + systems:
            sd = wd / name
            sd.mkdir()
            px = sd / "pred.xml"
            if not trips or not triplets_to_musicxml(trips, str(px), beat_seconds=0.5):
                out[name] = None
                continue
            m = run_muster_evaluation(gt_xml, px, name, sd)
            out[name] = ({k: m[k] for k in KEYS} if m else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--out", default="nbest_data/muster_fixed_windowed_test.json")
    a = ap.parse_args()
    d = torch.load(ROOT / "nbest_data/test9_stride150.pt", map_location="cpu",
                   weights_only=False)
    W = d["window_tokens"].long()
    flat = score_token_positions(W.shape[1])
    row = {int(l): i for i, l in enumerate(d["window_line_idx"].tolist())}
    fc = {}
    for ci, l in enumerate(d["cand_line_idx"].tolist()):
        fc.setdefault(row[int(l)], ci)
    t = open(ROOT / "visualizer/data_testset.js", encoding="utf-8").read()
    pj = json.loads(t[t.index("{"): t.rindex("}") + 1])["examples"]
    sel = json.load(open(ROOT / "nbest_data/test_set_selector_eval.json"))
    pieces = sel["pieces"]

    jobs = []
    for wi in range(W.shape[0]):
        key = f"test-{wi:05d}"
        ex = pj.get(key)
        if not ex:
            continue
        gt_trips = toks_to_triplets(W[wi][flat])
        systems = [("ours", toks_to_triplets(d["cand_tokens"][fc[wi]].long()))]
        for short, grp in (("zeng", "rollouts_paper1"), ("beyer", "rollouts_paper2")):
            g = ex.get(grp)
            v = next((vv for vv in g.values()
                      if isinstance(vv, dict) and "pred_score" in vv), None) if g else None
            systems.append((short, dicts_to_triplets(v["pred_score"]) if v else None))
        jobs.append((key, gt_trips, systems))
    print(f"{len(jobs)} windows x 4 comparisons", flush=True)

    with Pool(a.procs) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(eval_window, jobs, chunksize=4)):
            recs.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(jobs)}", flush=True)
    import numpy as np
    print("\nname      " + "".join(f"{k.split('_')[0]:>9}" for k in KEYS))
    for name in ("self", "ours", "beyer", "zeng"):
        vals = {k: [r[name][k] for r in recs if r.get(name)] for k in KEYS}
        n = len(vals[KEYS[0]])
        print(f"{name:8} " + "".join(f"{np.mean(vals[k]):>9.3f}" for k in KEYS)
              + f"   (n={n})")
    for r in recs:
        wi = int(r["key"].split("-")[1])
        w = pieces.get(str(wi), pieces.get(wi))
        r["work"] = w.rsplit("/", 1)[0] if w else None
    json.dump({"records": recs}, open(ROOT / a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
