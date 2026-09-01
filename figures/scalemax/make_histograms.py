"""Tasks 4+5: shared per-window/per-piece F1 histograms, 3 systems overlaid.

16 figures: 4 metrics x {windowed, rollout} x {test, test+val}. Each figure has
two panels: the setting's primary granularity (windowed -> per WINDOW; rollout
-> per PERFORMANCE) and per MUSICAL WORK (means). Step histograms with a light
fill, identical fixed colors per system everywhere (color follows the entity):
ours #2a78d6, Beyer #eb6834, Zeng #1baf7a -- the validated first-three
categorical slots of the reference palette, on its light surface.

Sources (whichever exist -- reruns pick up the rest):
  windowed test: nbest_data/test_set_scalemax_f1.json  (+ selector-eval pieces)
  windowed val : nbest_data/val_set_scalemax_f1.json   (pieces in the payload)
  rollout      : nbest_data/fullsong_eval.json
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "hists"
OUT.mkdir(exist_ok=True)
SYS = [("ours", "Ours (FT)", "#2a78d6"), ("beyer", "Beyer & Dai", "#eb6834"),
       ("zeng", "Zeng+", "#1baf7a")]
METS = [("base", "onset±1+pitch F1"), ("smax", "scale-max F1"),
        ("ioi", "interarrival F1"), ("ioi_smax", "IOI scale-max F1")]
SURF, INK, MUT = "#fcfcfb", "#0b0b0b", "#52514e"

def windowed_records():
    """[(split, work, {sys_met: f1})], from the windowed evals."""
    out = []
    f = ROOT / "nbest_data/test_set_scalemax_f1.json"
    if f.exists():
        sel = json.load(open(ROOT / "nbest_data/test_set_selector_eval.json"))
        pieces = sel["pieces"]
        for r in json.load(open(f))["records"]:
            wi = int(r["key"].split("-")[1])
            w = pieces.get(str(wi), pieces.get(wi))
            if w:
                out.append(("test", w.rsplit("/", 1)[0], r))
    f = ROOT / "nbest_data/val_set_scalemax_f1.json"
    if f.exists():
        t = open(ROOT / "visualizer/data_valset.js", encoding="utf-8").read()
        pj = json.loads(t[t.index("{"): t.rindex("}") + 1])["examples"]
        for r in json.load(open(f))["records"]:
            e = pj.get(r["key"])
            if e and e.get("piece"):
                out.append(("val", e["piece"].rsplit("/", 1)[0], r))
    return out

def rollout_records():
    f = ROOT / "nbest_data/fullsong_eval.json"
    if not f.exists():
        return []
    return [(r["split"], r["work"], r) for r in json.load(open(f))["records"]]

def panel(ax, v, color, label, title, xlab, ymax, show_x):
    """ONE system per panel -- small multiples. Three overlaid steps on a
    bimodal distribution were unreadable; identity now comes from the row,
    color is reinforcement, and a shared y-scale makes shapes comparable."""
    bins = np.linspace(0, 100, 21)
    if v is not None and len(v):
        ax.hist(v, bins=bins, histtype="stepfilled", alpha=0.25, color=color)
        ax.hist(v, bins=bins, histtype="step", lw=1.8, color=color)
        m = float(np.mean(v))
        ax.axvline(m, color=INK, lw=1.1, ls=":")
        ax.text(m + 1.5, ymax * 0.92, f"mean {m:.1f}", color=INK, fontsize=8.5,
                va="top")
    ax.text(0.99, 0.95, label, transform=ax.transAxes, ha="right", va="top",
            color=color, fontsize=10, fontweight="bold")
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    if show_x:
        ax.set_xlabel(xlab, color=MUT)
    else:
        ax.tick_params(labelbottom=False)
    ax.set_ylabel("count", color=MUT, fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", color="#e8e7e4", lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d5d4d0")
    ax.tick_params(colors=MUT)

def figure(recs, met, mlab, setting, scope, unit):
    vals_p = {k: [100 * r[f"{k}_{met}"] for _, _, r in recs
                  if f"{k}_{met}" in r] for k, _, _ in SYS}
    byw = defaultdict(lambda: defaultdict(list))
    for _, w, r in recs:
        for k, _, _ in SYS:
            if f"{k}_{met}" in r:
                byw[w][k].append(100 * r[f"{k}_{met}"])
    vals_w = {k: [np.mean(v[k]) for v in byw.values() if v[k]] for k, _, _ in SYS}
    fig, axes = plt.subplots(3, 2, figsize=(11, 6.4), facecolor=SURF,
                             sharex="col")
    n = len(recs)
    bins = np.linspace(0, 100, 21)
    # shared y per column so the three shapes are directly comparable
    ymax_p = max((np.histogram(v, bins=bins)[0].max()
                  for v in vals_p.values() if len(v)), default=1) * 1.15
    ymax_w = max((np.histogram(v, bins=bins)[0].max()
                  for v in vals_w.values() if len(v)), default=1) * 1.3
    for i, (key, label, color) in enumerate(SYS):
        panel(axes[i][0], vals_p.get(key), color, label,
              f"per {unit} (n={n})" if i == 0 else None,
              f"{mlab} (%)", ymax_p, show_x=(i == 2))
        panel(axes[i][1], vals_w.get(key), color, label,
              f"per musical work (n={len(byw)}, means)" if i == 0 else None,
              f"{mlab} (%)", ymax_w, show_x=(i == 2))
        for ax in axes[i]:
            ax.set_facecolor(SURF)
    fig.suptitle(f"{mlab} — {setting}, {scope}", color=INK, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    name = f"hist_{met}_{setting}_{scope.replace('+', '_')}.png"
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    return name

def main():
    made = []
    for setting, loader, unit in (("windowed", windowed_records, "window"),
                                  ("rollout", rollout_records, "performance")):
        rows = loader()
        if not rows:
            print(f"{setting}: no data yet, skipped")
            continue
        for scope in ("test", "test+val"):
            recs = [r for r in rows if scope == "test+val" or r[0] == "test"]
            if scope == "test+val" and not any(s == "val" for s, _, _ in rows):
                print(f"{setting}/{scope}: val data missing, skipped")
                continue
            for met, mlab in METS:
                made.append(figure(recs, met, mlab, setting, scope, unit))
    print(f"wrote {len(made)} figures to {OUT}:")
    for m in made:
        print("  " + m)

if __name__ == "__main__":
    main()
