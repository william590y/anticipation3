#!/usr/bin/env python
"""Add scale-max and interarrival F1 to every rollout in data.js.

Merges three entries into each rollout blob's existing `f1` dict, alongside
the standard criteria, so the viz's F1 panel can display them by key:

  tol1_scalemax     {f1, r}  max over ASAP beat-unit ratios of onset±1+pitch
                             F1 with onsets scaled t -> round(t*r); paper rows
                             are rebinned exactly from their stored
                             pred_quarters (t = round(on*50/qpb*r)).
  tol1_ioi          {f1}     the same matcher on (interarrival, pitch) events.
  tol1_ioi_scalemax {f1, r}  max over ratios of the interarrival variant.

CPU-only; ~1 min. Rerun after any merge that adds rollout groups, then rerun
split_visualizer_payload.py. Caveat: max over 13 scales inflates every row;
compare rows to each other, not to the standard columns.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nbest.eval_scalemax_f1 import (RATIOS, scaled_ours, scaled_paper,  # noqa: E402
                                    to_ioi, tol1)


def iter_rollouts(ex):
    for g, v in ex.items():
        if not (isinstance(g, str) and g.startswith("rollouts")):
            continue
        if not isinstance(v, dict):
            continue
        for variant, roll in v.items():
            if isinstance(roll, dict) and "pred_score" in roll:
                yield g, variant, roll


def main() -> None:
    path = ROOT / "visualizer" / "data.js"
    txt = path.read_text(encoding="utf-8")
    prefix = txt[: txt.index("{")]
    payload = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    n_blobs = 0
    t0 = time.time()
    for key, ex in payload["examples"].items():
        gt = [n for n in (ex.get("gt_score") or []) if n]
        if not gt:
            continue
        gt_ioi = to_ioi(gt)
        for g, variant, roll in iter_rollouts(ex):
            pred_of = (
                (lambda r, q=roll["pred_quarters"], b=float(roll["quarters_per_beat"]):
                 scaled_paper(q, b, r))
                if roll.get("pred_quarters") and roll.get("quarters_per_beat")
                else (lambda r, nn=[n for n in roll["pred_score"] if n]:
                      scaled_ours(nn, r)))
            best = ibest = -1.0
            best_r = ibest_r = None
            ioi1 = None
            for r in RATIOS:
                pred = pred_of(float(r))
                f = tol1(pred, gt)
                if f > best:
                    best, best_r = f, float(r)
                fi = tol1(to_ioi(pred), gt_ioi)
                if fi > ibest:
                    ibest, ibest_r = fi, float(r)
                if r == 1:
                    ioi1 = fi
            f1d = roll.setdefault("f1", {})
            f1d["tol1_scalemax"] = {"f1": best, "r": best_r}
            f1d["tol1_ioi"] = {"f1": ioi1}
            f1d["tol1_ioi_scalemax"] = {"f1": ibest, "r": ibest_r}
            n_blobs += 1
        print(f"  {key}: done ({time.time()-t0:.0f}s)", flush=True)
    path.write_text(prefix + json.dumps(payload) + ";", encoding="utf-8")
    print(f"wrote {path} ({n_blobs} rollout blobs updated)")


if __name__ == "__main__":
    main()
