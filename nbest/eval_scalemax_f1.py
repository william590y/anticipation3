#!/usr/bin/env python
"""Scale-max and interarrival F1 on the test set: ours vs the two papers.

MOTIVATION. A transcription can be internally right but on the wrong BEAT
UNIT: ASAP annotates 6/8 at the dotted quarter, 2/2 at the half, etc., and a
model that detected beats at a different unit renders every onset scaled by a
rational ratio. The Ballade evidence says this is live: Beyer scored 4.7% on
it while its run log shows `span ratio 2.00` and `meter changes; using modal
qpb=3` on exactly that piece. So:

  scale-max F1:      max over r in RATIOS of tol1-F1 with onsets t->round(t*r)
                     (papers rebinned exactly from their stored quarter-valued
                     output: t = round(on * 50/qpb * r); ours scaled in float).
                     RATIOS = all ratios of ASAP beat units {8th, quarter,
                     dotted quarter, half, dotted half}.
  interarrival F1:   the same tol1 matcher run on (IOI, pitch) events, where
                     IOI_i = t_i - t_{i-1} after sorting by onset (first note
                     0). Rhythm-shape credit without global-alignment credit.
                     NOTE the matcher is content-based, so this matches a
                     (IOI, pitch) pair anywhere in the window -- it is a
                     deliberately forgiving upper-bound diagnostic.

HONESTY CAVEAT baked into the report: max over 13 scales is 13 chances to
match, so scale-max INFLATES everyone -- ours included, and ours is trained on
the GT grid so its own gain under scale-max estimates the pure inflation
floor. Read the papers' gain RELATIVE to ours' gain, not absolutely.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from f1_reward import score_triplet_to_note            # noqa: E402
from onpolicy_rollout import score_token_positions     # noqa: E402

BEATS = [Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3)]
RATIOS = sorted({a / b for a in BEATS for b in BEATS})   # 13 rationals, 1/6..6


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_cf1 = None
def tol1(pred, gt):
    global _cf1
    if _cf1 is None:
        _cf1 = _load("compute_f1", "visualizer/compute_f1.py")
    return _cf1.score_notes(pred, gt)["onset_pitch_tol1"]["f1"]


def to_ioi(notes):
    s = sorted(notes, key=lambda n: (n["t"], n["p"]))
    out, prev = [], None
    for n in s:
        out.append({"t": 0 if prev is None else n["t"] - prev, "d": n["d"],
                    "p": n["p"]})
        prev = n["t"]
    return out


def scaled_ours(notes, r):
    return [{"t": int(round(n["t"] * r)), "d": max(1, int(round(n["d"] * r))),
             "p": n["p"]} for n in notes]


def scaled_paper(quarters, qpb, r):
    f = 50.0 / qpb * r
    return [{"t": int(round(q["on"] * f)), "d": max(1, int(round(q["dur"] * f))),
             "p": q["p"]} for q in quarters]


def eval_window(job):
    # job carries PLAIN DATA only -- multiprocessing pickles it, and closures
    # don't pickle (the first submission died on exactly that).
    key, gt, ours, beyer_q, zeng_q = job
    models = {
        "ours": lambda r: scaled_ours(ours, r),
        "beyer": lambda r: scaled_paper(beyer_q[0], beyer_q[1], r),
        "zeng": lambda r: scaled_paper(zeng_q[0], zeng_q[1], r),
    }
    out = {"key": key}
    for name, notes_fn in models.items():
        best, best_r, ibest, ibest_r = -1.0, None, -1.0, None
        for r in RATIOS:
            pred = notes_fn(float(r))
            f = tol1(pred, gt)
            if f > best:
                best, best_r = f, float(r)
            fi = tol1(to_ioi(pred), to_ioi(gt))
            if fi > ibest:
                ibest, ibest_r = fi, float(r)
            if r == 1:
                out[f"{name}_base"] = f
                out[f"{name}_ioi"] = fi
        out[f"{name}_smax"] = best
        out[f"{name}_smax_r"] = best_r
        out[f"{name}_ioi_smax"] = ibest
        out[f"{name}_ioi_smax_r"] = ibest_r
    return out


def build_jobs(shard="nbest_data/test9_stride150.pt",
               payload_js="visualizer/data_testset.js"):
    d = torch.load(ROOT / shard, map_location="cpu", weights_only=False)
    flat = score_token_positions(d["window_tokens"].shape[1])
    row = {int(l): i for i, l in enumerate(d["window_line_idx"].tolist())}
    first_cand = {}
    for ci, l in enumerate(d["cand_line_idx"].tolist()):
        first_cand.setdefault(row[int(l)], ci)          # greedy = candidate 0

    t = open(ROOT / payload_js, encoding="utf-8").read()
    p = json.loads(t[t.index("{"): t.rindex("}") + 1])

    def notes_of(tok):
        o = []
        for k in range(len(tok) // 3):
            n = score_triplet_to_note(tok[3 * k], tok[3 * k + 1], tok[3 * k + 2])
            if n is not None:
                o.append({"t": int(n[0]), "d": int(n[1]), "p": int(n[2])})
        return o

    # Match the payload's own key prefix (test-*/val-*).
    kprefix = (p.get("example_order") or ["test-0"])[0].split("-")[0]
    jobs, skipped = [], Counter()
    for wi in range(d["window_tokens"].shape[0]):
        key = f"{kprefix}-{wi:05d}"
        ex = p["examples"].get(key)
        if ex is None:
            skipped["no_example"] += 1
            continue
        pq = {}
        for gname, short in (("rollouts_paper1", "zeng"),
                             ("rollouts_paper2", "beyer")):
            g = ex.get(gname)
            v = next((vv for vv in g.values()
                      if isinstance(vv, dict) and "pred_quarters" in vv), None) if g else None
            if v is None:
                pq = None
                break
            pq[short] = (v["pred_quarters"], float(v["quarters_per_beat"]))
        if pq is None:
            skipped["missing_paper"] += 1
            continue
        gt = notes_of(d["window_tokens"][wi][flat].tolist())
        ours = notes_of(d["cand_tokens"][first_cand[wi]].tolist())
        jobs.append((key, gt, ours, pq["beyer"], pq["zeng"]))
    print(f"jobs: {len(jobs)}  skipped: {dict(skipped)}", flush=True)
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--shard", default="nbest_data/test9_stride150.pt")
    ap.add_argument("--payload", default="visualizer/data_testset.js")
    ap.add_argument("--pieces-from-payload", action="store_true",
                    help="take each window's piece from the payload's own "
                         "`piece` field (val windows have no selector-eval "
                         "json to map through)")
    ap.add_argument("--out", default="nbest_data/test_set_scalemax_f1.json")
    a = ap.parse_args()
    jobs = build_jobs(a.shard, a.payload)
    with Pool(a.procs) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(eval_window, jobs, chunksize=8)):
            recs.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(jobs)}", flush=True)

    if a.pieces_from_payload:
        t = open(ROOT / a.payload, encoding="utf-8").read()
        pj = json.loads(t[t.index("{"): t.rindex("}") + 1])["examples"]
        def work_of(key):
            e = pj.get(key)
            return e["piece"].rsplit("/", 1)[0] if e and e.get("piece") else None
    else:
        sel = json.load(open(ROOT / "nbest_data/test_set_selector_eval.json"))
        pieces = sel["pieces"]
        def work_of(key):
            wi = int(key.split("-")[1])
            w = pieces.get(str(wi), pieces.get(wi))
            return w.rsplit("/", 1)[0] if w else None

    def sf(dv, n=200000, seed=0):
        dv = dv[~np.isnan(dv)]
        rng = np.random.default_rng(seed)
        null = np.abs((rng.choice((-1., 1.), size=(n, dv.size)) * dv).mean(1))
        return float((np.count_nonzero(null >= abs(dv.mean()) - 1e-15) + 1) / (n + 1))

    metrics = ["base", "smax", "ioi", "ioi_smax"]
    names = ["ours", "beyer", "zeng"]
    print("\n" + "=" * 84)
    print(f"WINDOW LEVEL (n={len(recs)})")
    print(f"{'metric':12}" + "".join(f"{m:>12}" for m in names))
    for met in metrics:
        print(f"{met:12}" + "".join(
            f"{100*np.mean([r[f'{n}_{met}'] for r in recs]):>11.2f}%" for n in names))

    byw = defaultdict(list)
    for r in recs:
        w = work_of(r["key"])
        if w:
            byw[w].append(r)
    print(f"\nPIECE LEVEL (n={len(byw)} works)")
    pmeans = {}
    for met in metrics:
        for n in names:
            pmeans[(n, met)] = np.array(
                [np.mean([r[f"{n}_{met}"] for r in g]) for g in byw.values()])
    print(f"{'metric':12}" + "".join(f"{m:>12}" for m in names)
          + f"{'beyer-ours p':>14}{'zeng-ours p':>13}")
    for met in metrics:
        line = f"{met:12}" + "".join(f"{100*pmeans[(n,met)].mean():>11.2f}%" for n in names)
        line += f"{sf(pmeans[('beyer',met)]-pmeans[('ours',met)]):>14.4f}"
        line += f"{sf(pmeans[('zeng',met)]-pmeans[('ours',met)]):>13.4f}"
        print(line)

    print("\nargmax-scale histogram (scale-max, absolute onsets):")
    for n in names:
        c = Counter(round(r[f"{n}_smax_r"], 4) for r in recs)
        top = "  ".join(f"r={k}:{v}" for k, v in c.most_common(6))
        print(f"  {n:6} {top}")

    print("\nper-work scale-max gain (smax - base), tol1 pts:")
    print(f"{'work':30}{'n':>5}" + "".join(f"{n+'_gain':>12}" for n in names)
          + f"{'beyer r*':>9}")
    for w in sorted(byw, key=lambda x: -len(byw[x])):
        g = byw[w]
        nm = w.split("/")[-2] if w.count("/") >= 2 else w
        line = f"{nm[:28]:30}{len(g):>5}"
        for n in names:
            gain = 100*(np.mean([r[f"{n}_smax"] for r in g])
                        - np.mean([r[f"{n}_base"] for r in g]))
            line += f"{gain:>+12.1f}"
        rmode = Counter(round(r["beyer_smax_r"], 3) for r in g).most_common(1)[0][0]
        print(line + f"{rmode:>9}")

    json.dump({"records": recs, "ratios": [float(r) for r in RATIOS]},
              open(ROOT / a.out, "w"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
