"""Curiosity (1): the window histogram remade at the NOTE level.

At note granularity the F1 outcome per note is binary, so the informative
distribution is the SIGNED ONSET ERROR per pitch-matched note (10ms bins):
its mass inside [-1, +1] is exactly what tol1-F1 counts. Matching mirrors the
table's matcher (per pitch, onset-sorted, greedy nearest, one-to-one) with the
tolerance removed; GT notes with no same-pitch partner are the "no pitch
match" bar. Small multiples, one row per system, same palette as the window
histograms; error axis clipped to +-25 bins with overflow gutters.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from f1_reward import score_triplet_to_note   # noqa: E402
import torch                                   # noqa: E402

SYS = [("ours", "Ours (FT)", "#2a78d6"), ("beyer", "Beyer & Dai", "#eb6834"),
       ("zeng", "Zeng+", "#1baf7a")]
SURF, INK, MUT = "#fcfcfb", "#0b0b0b", "#52514e"
CLIP = 25

def match_errors(pred, gt):
    """Signed onset errors for one-to-one same-pitch matches (tol=inf), plus
    the count of GT notes left unmatched (wrong/absent pitch)."""
    errs, un = [], 0
    bp = defaultdict(list)
    for n in pred:
        bp[n["p"]].append(n["t"])
    for p in bp:
        bp[p].sort()
    for p, gts in sorted(
            defaultdict(list, {p: sorted(n["t"] for n in gt if n["p"] == p)
                               for p in {n["p"] for n in gt}}).items()):
        cand = bp.get(p, [])
        used = [False] * len(cand)
        for t in gts:
            best, bi = None, -1
            for i, ct in enumerate(cand):
                if used[i]:
                    continue
                d = abs(ct - t)
                if best is None or d < best:
                    best, bi = d, i
            if bi >= 0:
                used[bi] = True
                errs.append(cand[bi] - t)
            else:
                un += 1
    return errs, un

def main():
    d = torch.load(ROOT / "nbest_data/test9_stride150.pt", map_location="cpu",
                   weights_only=False)
    from onpolicy_rollout import score_token_positions
    flat = score_token_positions(d["window_tokens"].shape[1])
    row = {int(l): i for i, l in enumerate(d["window_line_idx"].tolist())}
    fc = {}
    for ci, l in enumerate(d["cand_line_idx"].tolist()):
        fc.setdefault(row[int(l)], ci)
    t = open(ROOT / "visualizer/data_testset.js", encoding="utf-8").read()
    pj = json.loads(t[t.index("{"): t.rindex("}") + 1])["examples"]

    def notes_of(flat_toks):
        toks = [int(x) for x in flat_toks]
        out = []
        for k in range(len(toks) // 3):
            n = score_triplet_to_note(toks[3*k], toks[3*k+1], toks[3*k+2])
            if n is not None:
                out.append({"t": int(n[0]), "d": int(n[1]), "p": int(n[2])})
        return out

    errs = {k: [] for k, _, _ in SYS}
    unmatched = {k: 0 for k, _, _ in SYS}
    total_gt = 0
    for wi in range(d["window_tokens"].shape[0]):
        key = f"test-{wi:05d}"
        ex = pj.get(key)
        if not ex:
            continue
        gt = notes_of(d["window_tokens"][wi][flat])
        total_gt += len(gt)
        preds = {"ours": notes_of(d["cand_tokens"][fc[wi]].long())}
        for short, grp in (("zeng", "rollouts_paper1"), ("beyer", "rollouts_paper2")):
            g = ex.get(grp)
            v = next((vv for vv in g.values()
                      if isinstance(vv, dict) and "pred_score" in vv), None) if g else None
            preds[short] = ([n for n in v["pred_score"] if n] if v else None)
        for k, _, _ in SYS:
            if preds.get(k) is None:
                continue
            e, un = match_errors(preds[k], gt)
            errs[k].extend(e)
            unmatched[k] += un
        if (wi + 1) % 300 == 0:
            print(f"  {wi+1} windows", flush=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), facecolor=SURF, sharex=True)
    bins = np.arange(-CLIP - 1.5, CLIP + 2.5, 1.0)
    for i, (k, label, color) in enumerate(SYS):
        ax = axes[i]
        e = np.clip(np.array(errs[k], float), -CLIP - 1, CLIP + 1)
        n_match = len(errs[k])
        within = np.mean(np.abs(np.array(errs[k])) <= 1) if n_match else 0
        ax.hist(e, bins=bins, color=color, alpha=0.3)
        ax.hist(e, bins=bins, histtype="step", lw=1.6, color=color)
        ax.axvspan(-1.5, 1.5, color=INK, alpha=0.05)
        un_frac = unmatched[k] / max(1, total_gt)
        ax.text(0.99, 0.9, f"{label}\n|err|<=1: {within:.1%} of matched\n"
                           f"no pitch match: {un_frac:.1%} of GT notes",
                transform=ax.transAxes, ha="right", va="top", color=color,
                fontsize=10, fontweight="bold")
        ax.set_facecolor(SURF)
        ax.set_ylabel("notes", color=MUT, fontsize=9)
        ax.grid(True, axis="y", color="#e8e7e4", lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUT)
    axes[-1].set_xlabel("signed onset error, 10 ms bins "
                        f"(clipped at ±{CLIP}; shaded band = tol1)", color=MUT)
    fig.suptitle("Per-note onset error — windowed, test (pitch-matched notes; "
                 f"{total_gt:,} GT notes)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(__file__).parent / "hists" / "hist_notelevel_onset_err_test.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
