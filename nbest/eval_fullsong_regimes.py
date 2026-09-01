"""Full-song F1 tables across models x window regimes x metric families.

Scores every rollout source (baseline + mask fine-tunes x slide-69 /
slide-by-1 / reset / gt-ctx, plus Beyer paper2 and Zeng paper1 outputs)
against the shared GT under the four whole-song metric families of
nbest/eval_fullsong.py: base tol1, scale-max, IOI, and IOI scale-max.
Prints the requested tables (IOI scale-max; then scale-max and IOI alone).
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nbest.eval_fullsong import eval_perf  # noqa: E402

SOURCES = [
    ("base s69",  "units",    "fullsong_rollouts/test/slide"),
    ("base s1",   "units",    "fullsong_slide1_base/test/slide"),
    ("base rst",  "units",    "fullsong_rollouts/test/reset"),
    ("base gtc",  "units",    "fullsong_rollouts/test/slide_gtctx"),
    ("m25 s69",   "units",    "fullsong_rollouts_maskft25/test/slide"),
    ("m25 s1",    "units",    "fullsong_slide1_mask25/test/slide"),
    ("m25 rst",   "units",    "fullsong_rollouts_maskft25/test/reset"),
    ("m50 s69",   "units",    "fullsong_rollouts_maskft50/test/slide"),
    ("m50 s1",    "units",    "fullsong_slide1_mask50/test/slide"),
    ("m50 rst",   "units",    "fullsong_rollouts_maskft50/test/reset"),
    ("m75 s69",   "units",    "fullsong_rollouts_maskft/test/slide"),
    ("m75 s1",    "units",    "fullsong_slide1_mask75/test/slide"),
    ("m75 rst",   "units",    "fullsong_rollouts_maskft/test/reset"),
    ("m75 gtc",   "units",    "fullsong_rollouts_maskft/test/slide_gtctx"),
    ("base pf",   "units",    "fullsong_rollouts/test/slide_pforce"),
    ("m25 pf",    "units",    "fullsong_rollouts_maskft25/test/slide_pforce"),
    ("m50 pf",    "units",    "fullsong_rollouts_maskft50/test/slide_pforce"),
    ("m75 pf",    "units",    "fullsong_rollouts_maskft/test/slide_pforce"),
    ("dag1 s69",  "units",    "fullsong_rollouts_dagger75v2/test/slide"),
    ("dag1 rst",  "units",    "fullsong_rollouts_dagger75v2/test/reset"),
    ("dag2 s69",  "units",    "fullsong_rollouts_dagger75v3/test/slide"),
    ("dag2 rst",  "units",    "fullsong_rollouts_dagger75v3/test/reset"),
    ("dag3 s69",  "units",    "fullsong_rollouts_dagger75v4/test/slide"),
    ("dag3 rst",  "units",    "fullsong_rollouts_dagger75v4/test/reset"),
    ("m25dag3 s69","units",   "fullsong_mask25_dagv3/test/slide"),
    ("m50dag3 s69","units",   "fullsong_mask50_dagv3/test/slide"),
    ("m75dag3 s69","units",   "fullsong_mask75_dagv3/test/slide"),
    ("m75dag3 rst","units",   "fullsong_mask75_dagv3/test/reset"),
    ("rr15 s69",   "units",   "fullsong_rollouts_randrep15/test/slide"),
    ("beyer",     "quarters", "fullsong_papers/test/paper2"),
    ("zeng",      "quarters", "fullsong_papers/test/paper1"),
]


def main() -> None:
    gt_dir = ROOT / "fullsong_rollouts/test/slide"
    jobs = []
    for f in sorted(gt_dir.glob("*.json")):
        gt = json.load(open(f))["gt"]
        models = {}
        for name, kind, dd in SOURCES:
            pf = ROOT / dd / f.name
            if not pf.exists():
                continue
            r = json.load(open(pf))
            if kind == "units":
                if r["pred"]:
                    models[name] = ("units", r["pred"])
            else:
                models[name] = ("quarters", (r["notes"], float(r["qpb"])))
        jobs.append((f.stem, gt, models))
    print(f"{len(jobs)} performances", flush=True)

    with Pool(16) as pool:
        recs = list(pool.imap_unordered(eval_perf, jobs, chunksize=1))

    persong = {r.pop("key"): r for r in recs}
    agg = {}
    for name, _, _ in SOURCES:
        for met in ("base", "smax", "ioi", "ioi_smax"):
            vals = [r[f"{name}_{met}"] for r in recs if f"{name}_{met}" in r]
            agg[(name, met)] = (len(vals), 100 * float(np.mean(vals)))

    def table(title, mets):
        print(f"\n=== {title} ===")
        print(f"{'source':10s} {'n':>3s}" + "".join(f"{m:>10s}" for m in mets))
        for name, _, _ in SOURCES:
            n = agg[(name, mets[0])][0]
            if n == 0:
                continue
            cells = "".join(f"{agg[(name, m)][1]:9.2f}%" for m in mets)
            print(f"{name:10s} {n:3d}{cells}")

    table("full-song F1, IOI scale-max", ["ioi_smax"])
    table("full-song F1, scale-max and IOI separately", ["smax", "ioi"])
    table("reference: base tol1", ["base"])
    json.dump({f"{k[0]}|{k[1]}": v for k, v in agg.items()},
              open(ROOT / "nbest_data/fullsong_regimes_eval.json", "w"))
    json.dump(persong, open(ROOT / "nbest_data/fullsong_regimes_persong.json", "w"))


if __name__ == "__main__":
    main()
