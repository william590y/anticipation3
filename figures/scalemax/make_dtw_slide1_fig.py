"""DTW small-multiples for slide-by-one rollouts: models x (warp, error, match).

Reads nbest_data/dtw_slide1_{base,mask75,mask25}.json (+ optional mask50) and
overlays the baseline overlap-69 warp median (nbest_data/dtw_drift.json) for
reference. One boundary matters in this regime: note 138, where the context
window fills and per-note sliding begins. Writes results/dtw_slide1.png.
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

XMAX, SMOOTH, FILL = 1200, 21, 138
MODELS = [("base", "baseline (no mask)"), ("mask25", "mask 0.25"),
          ("mask75", "mask 0.75")]
if os.path.exists("nbest_data/dtw_slide1_mask50.json"):
    MODELS.insert(2, ("mask50", "mask 0.50"))

C_RAW, C_WARP, C_NEU, C_REF = "#c4531f", "#1f5fa8", "#444444", "#999999"


def load(path):
    songs = json.load(open(path))
    L = XMAX
    arr = {k: np.full((len(songs), L), np.nan)
           for k in ("warp", "eraw", "ewrp", "mraw", "mwrp")}
    for s, d in enumerate(songs):
        n = min(L, d["n_gt"])
        arr["warp"][s, :n] = d["warp"][:n]
        er = np.array(d["err_raw"][:n], dtype=float)
        ew = np.array(d["err_warp"][:n], dtype=float)
        arr["eraw"][s, :n] = er
        arr["ewrp"][s, :n] = ew
        arr["mraw"][s, :n] = ~np.isnan(er)
        arr["mwrp"][s, :n] = ~np.isnan(ew)
    return arr


def roll(a, k=SMOOTH):
    pad = k // 2
    v = np.pad(a, pad, mode="edge")
    return np.convolve(np.nan_to_num(v), np.ones(k), "valid") / np.convolve(
        (~np.isnan(v)).astype(float), np.ones(k), "valid").clip(1e-9)


ref = load("nbest_data/dtw_drift.json")
ref_warp = np.nanmedian(ref["warp"], axis=0)

ncol = len(MODELS)
fig, axes = plt.subplots(3, ncol, figsize=(4.6 * ncol + 1, 8.6),
                         sharex=True, sharey="row")
x = np.arange(XMAX)
stats = {}
for c, (key, label) in enumerate(MODELS):
    a = load(f"nbest_data/dtw_slide1_{key}.json")
    show = (~np.isnan(a["warp"])).sum(axis=0) >= 10
    w_med = np.nanmedian(a["warp"], axis=0)
    w_lo = np.nanpercentile(a["warp"], 25, axis=0)
    w_hi = np.nanpercentile(a["warp"], 75, axis=0)
    er = roll(np.nanmean(a["eraw"], axis=0)) / 50.0
    ew = roll(np.nanmean(a["ewrp"], axis=0)) / 50.0
    mr = roll(np.nanmean(a["mraw"], axis=0)) * 100
    mw = roll(np.nanmean(a["mwrp"], axis=0)) * 100
    stats[key] = (np.nanmean(a["eraw"]) / 50, np.nanmean(a["ewrp"]) / 50,
                  np.nanmean(a["mraw"]) * 100, np.nanmean(a["mwrp"]) * 100)

    ax = axes[0, c]
    ax.fill_between(x[show], w_lo[show], w_hi[show], color=C_NEU, alpha=0.16,
                    lw=0)
    ax.plot(x[show], w_med[show], color=C_NEU, lw=1.7, label="median warp")
    ax.plot(x[: len(ref_warp)], ref_warp, color=C_REF, lw=1.1, ls="--",
            label="baseline slide-69 ref")
    ax.axhline(1.0, color="#2a7d4f", lw=0.9, ls="--")
    ax.set_ylim(0, 2.05)
    ax.set_title(label)
    if c == 0:
        ax.set_ylabel("warp factor\nd(pred)/d(gt)")
        ax.legend(fontsize=7, frameon=False, loc="upper right")

    ax = axes[1, c]
    ax.plot(x[show], er[show], color=C_RAW, lw=1.7, label="unwarped")
    ax.plot(x[show], ew[show], color=C_WARP, lw=1.7, label="after DTW warp")
    ax.set_ylim(bottom=0)
    if c == 0:
        ax.set_ylabel("mean |onset error|\n(beats, matched)")
        ax.legend(fontsize=7, frameon=False, loc="upper left")

    ax = axes[2, c]
    ax.plot(x[show], mr[show], color=C_RAW, lw=1.7)
    ax.plot(x[show], mw[show], color=C_WARP, lw=1.7)
    ax.set_ylim(0, 100)
    if c == 0:
        ax.set_ylabel("pitch matched\nwithin 2 beats (%)")
    ax.set_xlabel("GT score-note index")

for ax in axes.ravel():
    ax.axvline(FILL, color="#7a5aa0", lw=1.0, ls=":")
    ax.grid(alpha=0.22, lw=0.5)
    ax.set_xlim(0, XMAX)
axes[0, 0].annotate("context fills;\nslide-by-1 begins", xy=(FILL, 1.85),
                    fontsize=7.5, color="#7a5aa0", ha="left",
                    xytext=(FILL + 25, 1.72))

fig.suptitle("Slide-by-one-note rollouts (overlap 137): DTW warp and onset "
             "error vs song position, 59 test songs", y=0.995)
fig.tight_layout()
fig.savefig("results/dtw_slide1.png", dpi=130, bbox_inches="tight")
print("wrote results/dtw_slide1.png")
for k, (a, b, c_, d) in stats.items():
    print(f"{k:8s} err raw {a:.2f} -> warped {b:.2f} beats | "
          f"match {c_:.1f}% -> {d:.1f}%")
