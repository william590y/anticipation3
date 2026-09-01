#!/usr/bin/env python
"""Whole-song F1: ours (sliding-window rollouts) vs Beyer vs Zeng.

Reads fullsong_rollouts/<split>/slide/*.json (ours + GT, 10ms units) and
fullsong_papers/<split>/{paper1,paper2}/*.json (quarter units + qpb), scores
each performance under the four metric families of the window study --
tol1, scale-max, interarrival, IOI scale-max -- and aggregates per
performance, per work, per split.

Anchoring: pred and GT are each re-anchored to their OWN min onset (the
normalize_triplet_times convention -- over a whole song neither side owns the
other's origin). Papers rebinned exactly from quarters: t = round(on_q*50/qpb*r).

Output: nbest_data/fullsong_eval.json (per-perf records, feeds the histogram
task) + printed tables. CPU-parallel (--procs).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nbest.eval_scalemax_f1 import RATIOS, to_ioi, tol1   # noqa: E402


def norm(notes):
    if not notes:
        return notes
    m = min(n["t"] for n in notes)
    return [{"t": n["t"] - m, "d": n["d"], "p": n["p"]} for n in notes]


def eval_perf(job):
    key, gt, models = job
    gt = norm(gt)
    gt_ioi = to_ioi(gt)
    out = {"key": key}
    for name, (kind, payload) in models.items():
        best = ibest = -1.0
        best_r = ibest_r = None
        for r in RATIOS:
            r = float(r)
            if kind == "units":
                pred = [{"t": int(round(n["t"] * r)),
                         "d": max(1, int(round(n["d"] * r))), "p": n["p"]}
                        for n in payload]
            else:                      # quarters
                notes, qpb = payload
                f = 50.0 / qpb * r
                pred = [{"t": int(round(n["on_q"] * f)),
                         "d": max(1, int(round(n["dur_q"] * f))), "p": n["p"]}
                        for n in notes]
            pred = norm(pred)
            fv = tol1(pred, gt)
            if fv > best:
                best, best_r = fv, r
            fi = tol1(to_ioi(pred), gt_ioi)
            if fi > ibest:
                ibest, ibest_r = fi, r
            if r == 1.0:
                out[f"{name}_base"] = fv
                out[f"{name}_ioi"] = fi
        out[f"{name}_smax"] = best
        out[f"{name}_smax_r"] = best_r
        out[f"{name}_ioi_smax"] = ibest
        out[f"{name}_ioi_smax_r"] = ibest_r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--splits", default="test,val")
    ap.add_argument("--out", default="nbest_data/fullsong_eval.json")
    a = ap.parse_args()

    jobs, meta = [], {}
    for split in a.splits.split(","):
        ours_dir = ROOT / "fullsong_rollouts" / split / "slide"
        for f in sorted(ours_dir.glob("*.json")):
            r = json.load(open(f))
            key = f"{split}/{r['perf_path'].lstrip('./')}"
            models = {"ours": ("units", r["pred"])}
            stem = f.stem + ".json"
            for short, kind in (("zeng", "paper1"), ("beyer", "paper2")):
                pf = ROOT / "fullsong_papers" / split / kind / stem
                if pf.exists():
                    pr = json.load(open(pf))
                    models[short] = ("quarters", (pr["notes"], float(pr["qpb"])))
            if len(models) == 3:
                jobs.append((key, r["gt"], models))
                meta[key] = {"split": split,
                             "work": "/".join(r["perf_path"].lstrip("./")
                                              .split("/")[:-1]),
                             "n_gt": len(r["gt"]), "n_pred": len(r["pred"]),
                             "boundaries": r["gen_stats"]["num_window_resets"]}
            else:
                print(f"  incomplete ({sorted(models)}): {key}")
    print(f"{len(jobs)} performances with all three models", flush=True)
    if not jobs:
        raise SystemExit("nothing to score yet")

    with Pool(a.procs) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(eval_perf, jobs, chunksize=1)):
            recs.append(r)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(jobs)}", flush=True)
    for r in recs:
        r.update(meta[r["key"]])

    def sf(dv, n=200000, seed=0):
        dv = dv[~np.isnan(dv)]
        rng = np.random.default_rng(seed)
        null = np.abs((rng.choice((-1., 1.), size=(n, dv.size)) * dv).mean(1))
        return float((np.count_nonzero(null >= abs(dv.mean()) - 1e-15) + 1) / (n + 1))

    METS = ["base", "smax", "ioi", "ioi_smax"]
    NAMES = ["ours", "beyer", "zeng"]
    for scope, sel in (("test", lambda r: r["split"] == "test"),
                       ("val", lambda r: r["split"] == "val"),
                       ("test+val", lambda r: True)):
        rs = [r for r in recs if sel(r)]
        if not rs:
            continue
        byw = defaultdict(list)
        for r in rs:
            byw[r["work"]].append(r)
        print(f"\n=== {scope}: {len(rs)} performances, {len(byw)} works ===")
        print(f"{'metric':10}" + "".join(f"{n:>9}" for n in NAMES)
              + "   | per-work" + "".join(f"{n:>9}" for n in NAMES)
              + f"{'p(B-O)':>9}{'p(Z-O)':>9}")
        for m in METS:
            perf = [100 * np.mean([r[f"{n}_{m}"] for r in rs]) for n in NAMES]
            pw = {n: np.array([np.mean([r[f"{n}_{m}"] for r in g])
                               for g in byw.values()]) for n in NAMES}
            print(f"{m:10}" + "".join(f"{v:>8.2f}%" for v in perf)
                  + "   |         "
                  + "".join(f"{100*pw[n].mean():>8.2f}%" for n in NAMES)
                  + f"{sf(pw['beyer']-pw['ours']):>9.3f}"
                  + f"{sf(pw['zeng']-pw['ours']):>9.3f}")
    json.dump({"records": recs}, open(ROOT / a.out, "w"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
