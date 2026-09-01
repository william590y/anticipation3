"""Fixed-MUSTER on FULL-SONG rollouts across models x regimes + papers.

Same debugged MUSTER pipeline as nbest/eval_muster_windows.py (see the FIX
comments in MUSTER/Code/ScoreMatchEvaluation_VoicePlus_v220118.cpp), applied
to whole-song predictions: every rollout source of
nbest/eval_fullsong_regimes.py plus Beyer (paper2) / Zeng (paper1) outputs,
with the GT self-comparison floor. MUSTER's own HMM aligner does temporal
matching, so this is an independent check on the clock-vs-content split.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nbest.eval_muster_windows import KEYS, dicts_to_triplets, eval_window  # noqa: E402
from nbest.eval_fullsong_regimes import SOURCES                             # noqa: E402


def quarters_to_dicts(notes, qpb):
    f = 50.0 / qpb
    return [{"t": int(round(n["on_q"] * f)),
             "d": max(1, int(round(n["dur_q"] * f))), "p": n["p"]}
            for n in notes]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out", default="nbest_data/muster_fixed_fullsong_test.json")
    a = ap.parse_args()

    gt_dir = ROOT / "fullsong_rollouts/test/slide"
    jobs = []
    for f in sorted(gt_dir.glob("*.json")):
        gt = dicts_to_triplets(json.load(open(f))["gt"])
        systems = []
        for name, kind, dd in SOURCES:
            pf = ROOT / dd / f.name
            if not pf.exists():
                systems.append((name, None))
                continue
            r = json.load(open(pf))
            notes = (r["pred"] if kind == "units"
                     else quarters_to_dicts(r["notes"], float(r["qpb"])))
            systems.append((name, dicts_to_triplets(notes)))
        jobs.append((f.stem, gt, systems))
    print(f"{len(jobs)} songs x {len(SOURCES)+1} pipelines", flush=True)

    with Pool(a.procs) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(eval_window, jobs, chunksize=1)):
            recs.append(r)
            print(f"  {i+1}/{len(jobs)} {r['key'][:40]}", flush=True)

    names = ["self"] + [s[0] for s in SOURCES]
    print("\nsource     " + "".join(f"{k.split('_')[0]:>9}" for k in KEYS))
    for name in names:
        vals = {k: [r[name][k] for r in recs if r.get(name)] for k in KEYS}
        n = len(vals[KEYS[0]])
        if n == 0:
            print(f"{name:10} (no results)")
            continue
        print(f"{name:10} " + "".join(f"{np.mean(vals[k]):>9.3f}" for k in KEYS)
              + f"   (n={n})")
    json.dump({"records": recs}, open(ROOT / a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
