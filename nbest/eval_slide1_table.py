"""Whole-song + segment tol1 F1 table across models x window regimes.

Covers the slide-by-one-note regime (overlap 137, fullsong_slide1_*) next to
the standard slide (overlap 69), reset, and GT-context oracle rollouts, for
the unmasked baseline and the finished mask-dropout fine-tunes.
"""
import json
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("cf1", "visualizer/compute_f1.py")
cf1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf1)

SEG = 138


def tol1(p, g):
    return cf1.score_notes(p, g)["onset_pitch_tol1"]["f1"]


def norm(ns):
    m = min(n["t"] for n in ns)
    return [{"t": n["t"] - m, "d": n["d"], "p": n["p"]} for n in ns]


def eval_dir(dd):
    files = sorted(Path(dd).glob("*.json"))
    if not files:
        return None
    whole, curve = [], {}
    for f in files:
        r = json.load(open(f))
        if not r["pred"] or not r["gt"]:
            continue
        pred, gt = norm(r["pred"]), norm(r["gt"])
        n = min(len(pred), len(gt))
        whole.append(tol1(pred[:n], gt[:n]))
        for si, s in enumerate(range(0, n - SEG + 1, SEG)):
            if si > 6:
                break
            curve.setdefault(si, []).append(
                tol1(norm(pred[s:s + SEG]), norm(gt[s:s + SEG])))
    segs = "  ".join(f"{si}:{100 * np.mean(v):4.1f}"
                     for si, v in sorted(curve.items()))
    return len(files), 100 * np.mean(whole), segs


ROWS = [
    ("baseline  slide-69", "fullsong_rollouts/test/slide"),
    ("baseline  slide-by-1", "fullsong_slide1_base/test/slide"),
    ("baseline  reset", "fullsong_rollouts/test/reset"),
    ("baseline  gt-ctx oracle", "fullsong_rollouts/test/slide_gtctx"),
    ("mask25    slide-69", "fullsong_rollouts_maskft25/test/slide"),
    ("mask25    slide-by-1", "fullsong_slide1_mask25/test/slide"),
    ("mask25    reset", "fullsong_rollouts_maskft25/test/reset"),
    ("mask75    slide-69", "fullsong_rollouts_maskft/test/slide"),
    ("mask75    slide-by-1", "fullsong_slide1_mask75/test/slide"),
    ("mask75    reset", "fullsong_rollouts_maskft/test/reset"),
    ("mask75    gt-ctx oracle", "fullsong_rollouts_maskft/test/slide_gtctx"),
]

print(f"{'model/regime':26s} {'n':>3s} {'whole F1':>9s}  per-138-note segment F1")
for name, dd in ROWS:
    r = eval_dir(dd)
    if r is None:
        print(f"{name:26s}   -   (no files)")
        continue
    n, w, segs = r
    print(f"{name:26s} {n:3d} {w:8.2f}%  {segs}")
