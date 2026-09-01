"""DTW drift analysis of full-song sliding-window rollouts.

Aligns each rollout (pred) to its GT score with DTW over chroma piano-roll
frames on the shared score grid (50 units/beat), then measures:
  - warp factor: d(pred time)/d(gt time) along the alignment path (1.0 = clock
    in sync; sustained !=1 = internal-clock drift that DTW absorbs)
  - per-note |onset error| and match rate, unwarped vs after warping pred
    onsets through the DTW path
aggregated by GT score-note index, where window boundaries sit at the fixed
indices 138 + 69k for every song (window 138, overlap 69).

Writes nbest_data/dtw_drift.json and results/dtw_drift.png.
"""
import json
import glob
import sys

import numpy as np

FRAME = 10          # grid units per DTW frame (0.2 beat)
SUSTAIN_W = 0.3     # weight of held notes vs onsets in the chroma feature
MATCH_CAP = 100     # max |onset error| in units for a pitch match (2 beats)
WIN, OVL = 138, 69  # window size / overlap in score notes


def chroma_frames(notes, n_frames):
    on = np.zeros((n_frames, 12), dtype=np.float32)
    sus = np.zeros((n_frames, 12), dtype=np.float32)
    for n in notes:
        f0 = min(n_frames - 1, n["t"] // FRAME)
        f1 = min(n_frames - 1, (n["t"] + max(1, n["d"])) // FRAME)
        c = n["p"] % 12
        on[f0, c] += 1.0
        sus[f0:f1 + 1, c] += 1.0
    feat = on + SUSTAIN_W * sus
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    feat /= np.maximum(norm, 1e-6)
    feat[norm[:, 0] < 1e-6] = 1.0 / np.sqrt(12)  # silence: uniform vector
    return feat


def dtw_path(A, B):
    """Full DTW, steps (1,1),(1,0),(0,1); returns path arrays (i_gt, j_pred)."""
    n, m = len(A), len(B)
    cost = 1.0 - A @ B.T
    D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        prev, cur = D[i - 1], D[i]
        # cur[j] = cost[i-1,j-1] + min(prev[j-1], prev[j], cur[j-1]); the
        # cur[j-1] term is sequential, so run the recurrence in a fused loop.
        c = cost[i - 1]
        run = np.minimum(prev[1:], prev[:-1])
        acc = np.inf
        for j in range(m):
            acc = c[j] + min(run[j], acc)
            cur[j + 1] = acc
    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        opts = [(D[i - 1, j - 1], i - 1, j - 1),
                (D[i - 1, j], i - 1, j),
                (D[i, j - 1], i, j - 1)]
        _, i, j = min(opts)
        path.append((max(0, i - 1), max(0, j - 1)))
    path.reverse()
    return np.array(path)


def analyze(pred, gt):
    pred = sorted(pred, key=lambda n: (n["t"], n["p"]))
    gt = sorted(gt, key=lambda n: (n["t"], n["p"]))
    T = max(max(n["t"] + n["d"] for n in gt), max(n["t"] + n["d"] for n in pred))
    nf = T // FRAME + 2
    path = dtw_path(chroma_frames(gt, nf), chroma_frames(pred, nf))

    # monotone frame maps: gt frame -> mean pred frame on path, and inverse
    gi, pj = path[:, 0].astype(float), path[:, 1].astype(float)
    ug = np.unique(gi.astype(int))
    g2p = np.array([pj[gi == g].mean() for g in ug])
    up = np.unique(pj.astype(int))
    p2g = np.array([gi[pj == p].mean() for p in up])

    # warp factor per gt frame, smoothed over ~2 beats (10 frames)
    slope = np.gradient(g2p, ug)
    k = 11
    kern = np.ones(k) / k
    slope_s = np.convolve(np.pad(slope, k // 2, mode="edge"), kern, "valid")

    gt_t = np.array([n["t"] for n in gt], dtype=float)
    gt_p = np.array([n["p"] for n in gt])
    pr_t = np.array([n["t"] for n in pred], dtype=float)
    pr_p = np.array([n["p"] for n in pred])
    pr_t_warp = np.interp(pr_t / FRAME, up, p2g) * FRAME
    warp_at_note = np.interp(gt_t / FRAME, ug, slope_s)

    def note_errors(pred_times):
        err = np.full(len(gt), np.nan)
        for i in range(len(gt)):
            cand = pred_times[pr_p == gt_p[i]]
            if len(cand):
                d = np.abs(cand - gt_t[i]).min()
                if d <= MATCH_CAP:
                    err[i] = d
        return err

    return {
        "warp": warp_at_note,
        "err_raw": note_errors(pr_t),
        "err_warp": note_errors(pr_t_warp),
    }


def main():
    src = "fullsong_rollouts/test/slide"
    dst = "nbest_data/dtw_drift.json"
    for a in sys.argv[1:]:
        if a.startswith("--dir="): src = a.split("=", 1)[1]
        if a.startswith("--out="): dst = a.split("=", 1)[1]
    paper = None
    for a in sys.argv[1:]:
        if a.startswith("--paper="):
            paper = a.split("=", 1)[1]
    files = sorted(glob.glob(f"{src}/*.json"))
    if "--selftest" in sys.argv:
        d = json.load(open(files[0]))
        r = analyze(d["gt"], d["gt"])
        ok = (abs(np.nanmedian(r["warp"]) - 1) < 1e-3
              and np.nanmax(r["err_warp"]) == 0 and np.nanmax(r["err_raw"]) == 0)
        print("selftest warp mean", np.nanmean(r["warp"]),
              "max err", np.nanmax(r["err_raw"]), "->", "OK" if ok else "FAIL")
        return
    out = []
    for f in files:
        d = json.load(open(f))
        if paper is not None:
            pf = f"{paper}/" + f.split("/")[-1]
            try:
                pr = json.load(open(pf))
            except FileNotFoundError:
                continue
            fac = 50.0 / float(pr["qpb"])
            d["pred"] = [{"t": int(round(n["on_q"] * fac)),
                          "d": max(1, int(round(n["dur_q"] * fac))),
                          "p": n["p"]} for n in pr["notes"] if n["p"] >= 1]
        if not d["pred"] or not d["gt"]:
            continue
        r = analyze(d["pred"], d["gt"])
        out.append({"file": f.split("/")[-1],
                    "n_gt": len(d["gt"]),
                    "warp": np.round(r["warp"], 4).tolist(),
                    "err_raw": r["err_raw"].tolist(),
                    "err_warp": r["err_warp"].tolist()})
        print(f"{f.split('/')[-1][:48]:48s} n={len(d['gt']):5d} "
              f"warp med={np.nanmedian(r['warp']):.3f} "
              f"err raw={np.nanmean(r['err_raw']):.1f} "
              f"warped={np.nanmean(r['err_warp']):.1f}", flush=True)
    json.dump(out, open(dst, "w"))
    print(f"wrote {dst} ({len(out)} songs)")


if __name__ == "__main__":
    main()
