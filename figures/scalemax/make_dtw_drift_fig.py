"""Figure: DTW warp factor + unwarped/warped onset error vs song position.

Reads nbest_data/dtw_drift.json (from nbest/dtw_drift_analysis.py).
X-axis is GT score-note index, so window boundaries (138 + 69k) line up
across all songs and are drawn as vertical lines. Writes results/dtw_drift.png.
"""
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIN, OVL = 138, 69
XMAX = 1200          # note-index range to show (>= half the songs reach ~1200)
SMOOTH = 21          # rolling window (notes) for the error/match curves

songs = json.load(open("nbest_data/dtw_drift.json"))
L = XMAX
warp = np.full((len(songs), L), np.nan)
eraw = np.full((len(songs), L), np.nan)
ewrp = np.full((len(songs), L), np.nan)
mraw = np.full((len(songs), L), np.nan)
mwrp = np.full((len(songs), L), np.nan)
for s, d in enumerate(songs):
    n = min(L, d["n_gt"])
    warp[s, :n] = d["warp"][:n]
    er = np.array(d["err_raw"][:n], dtype=float)
    ew = np.array(d["err_warp"][:n], dtype=float)
    eraw[s, :n] = er
    ewrp[s, :n] = ew
    mraw[s, :n] = ~np.isnan(er)
    mwrp[s, :n] = ~np.isnan(ew)

nsongs = (~np.isnan(warp)).sum(axis=0)
x = np.arange(L)
show = nsongs >= 10


def roll(a, k=SMOOTH):
    pad = k // 2
    v = np.pad(a, pad, mode="edge")
    return np.convolve(np.nan_to_num(v), np.ones(k), "valid") / np.convolve(
        (~np.isnan(v)).astype(float), np.ones(k), "valid").clip(1e-9)


w_med = np.nanmedian(warp, axis=0)
w_lo = np.nanpercentile(warp, 25, axis=0)
w_hi = np.nanpercentile(warp, 75, axis=0)
er_m = roll(np.nanmean(eraw, axis=0)) / 50.0   # units -> beats
ew_m = roll(np.nanmean(ewrp, axis=0)) / 50.0
mr_m = roll(np.nanmean(mraw, axis=0)) * 100
mw_m = roll(np.nanmean(mwrp, axis=0)) * 100

C_RAW, C_WARP, C_NEU = "#c4531f", "#1f5fa8", "#444444"
bounds = np.arange(WIN, XMAX, OVL)

fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True,
                         gridspec_kw={"height_ratios": [1.2, 1.2, 0.8]})
for ax in axes:
    for b in bounds:
        ax.axvline(b, color="#999999", lw=0.6, ls=":", zorder=0)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

ax = axes[0]
ax.fill_between(x[show], w_lo[show], w_hi[show], color=C_NEU, alpha=0.18,
                lw=0, label="IQR across songs")
ax.plot(x[show], w_med[show], color=C_NEU, lw=1.8, label="median warp factor")
ax.axhline(1.0, color="#2a7d4f", lw=1.0, ls="--")
ax.text(XMAX * 0.99, 1.02, "in sync (1.0)", ha="right", fontsize=8,
        color="#2a7d4f")
ax.set_ylabel("warp factor\nd(pred time)/d(gt time)")
ax.set_ylim(0, 2.05)
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax.set_title("DTW alignment of slide rollouts vs GT score (59 test songs); "
             "dotted verticals = window boundaries (138+69k)")

ax = axes[1]
ax.plot(x[show], er_m[show], color=C_RAW, lw=1.8, label="unwarped |onset error|")
ax.plot(x[show], ew_m[show], color=C_WARP, lw=1.8, label="after DTW warp")
ax.set_ylabel("mean |onset error|\n(beats, matched notes)")
ax.set_ylim(bottom=0)
ax.legend(loc="upper left", fontsize=8, frameon=False)

ax = axes[2]
ax.plot(x[show], mr_m[show], color=C_RAW, lw=1.8, label="unwarped")
ax.plot(x[show], mw_m[show], color=C_WARP, lw=1.8, label="warped")
ax.set_ylabel("pitch matched\nwithin 2 beats (%)")
ax.set_ylim(0, 100)
ax.set_xlabel("GT score-note index into song")
ax.legend(loc="lower left", fontsize=8, frameon=False)
ax.set_xlim(0, XMAX)

fig.align_ylabels(axes)
fig.tight_layout()
fig.savefig("results/dtw_drift.png", dpi=130, bbox_inches="tight")
print("wrote results/dtw_drift.png")
print(f"songs contributing at idx0/600/1199: {nsongs[0]}/{nsongs[600]}/{nsongs[1199]}")
print(f"overall mean err: raw {np.nanmean(eraw)/50:.2f} beats, "
      f"warped {np.nanmean(ewrp)/50:.2f} beats; "
      f"match rate raw {np.nanmean(mraw)*100:.1f}%, warped {np.nanmean(mwrp)*100:.1f}%")
