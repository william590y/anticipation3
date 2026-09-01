"""DAgger vs Beyer & Dai vs Zeng: whole-song comparison tables.

Reads the aggregates written by nbest/eval_fullsong_regimes.py
(nbest_data/fullsong_regimes_eval.json) and the fixed-MUSTER records
(nbest_data/muster_fixed_fullsong_test.json), and writes

  results/dagger_comparison.txt   -- plain text, both tables
  results/dagger_comparison.tex   -- standalone LaTeX, booktabs

F1 families (higher better): tol1 = onset+/-1 bin & pitch on the absolute
grid; scale-max = max over the 13 beat-unit rescalings; IOI = shift-invariant
inter-onset intervals; IOI scale-max = invariant to global scale AND offset.
MUSTER (fixed, lower better): the four debugged C++ fixes, GT self-floor
included for calibration.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

ROWS = [
    ("base s69", "baseline, slide-69"),
    ("m75 s69", "mask-dropout 0.75, slide-69"),
    ("dag1 s69", "DAgger round 1, slide-69"),
    ("dag2 s69", "DAgger round 2, slide-69"),
    ("dag3 s69", "DAgger round 3, slide-69"),
    ("dag1 rst", "DAgger round 1, reset"),
    ("dag2 rst", "DAgger round 2, reset"),
    ("dag3 rst", "DAgger round 3, reset"),
    ("beyer", "Beyer \\& Dai (ISMIR 2024)"),
    ("zeng", "Zeng+ (ICLR 2026)"),
]
METS = [("base", "tol1"), ("smax", "scale-max"), ("ioi", "IOI"),
        ("ioi_smax", "IOI scale-max")]
MKEYS = ("pitch_error_rate", "missing_note_rate", "extra_note_rate",
         "onset_time_error_rate", "offset_time_error_rate", "mean_error_rate")
MHDR = ("pitch", "missing", "extra", "onset", "offset", "mean")


def load():
    agg = json.load(open(ROOT / "nbest_data/fullsong_regimes_eval.json"))
    mus_raw = json.load(open(ROOT / "nbest_data/muster_fixed_fullsong_test.json"))
    mus: dict = {}
    for r in mus_raw.get("records", []):
        if r.get("metrics"):
            mus.setdefault(r["name"], []).append(r["metrics"])
    mus_mean = {k: {m: float(np.mean([v[m] for v in rows])) for m in MKEYS}
                | {"n": len(rows)} for k, rows in mus.items()}
    return agg, mus_mean


def txt(agg, mus) -> str:
    L = []
    L.append("Whole-song F1 on the 59-performance paper test split "
             "(higher is better, %)")
    L.append("")
    L.append(f"{'system':32s} {'n':>3s}" + "".join(f"{h:>14s}"
                                                   for _, h in METS))
    for key, label in ROWS:
        n = agg.get(f"{key}|base", [0, 0])[0]
        if not n:
            continue
        cells = "".join(f"{agg[f'{key}|{m}'][1]:13.2f}%" for m, _ in METS)
        L.append(f"{label.replace(chr(92)+'&', '&'):32s} {n:3d}{cells}")
    L.append("")
    L.append("Fixed-MUSTER error rates, whole-song (LOWER is better, %); "
             "7 Ondine performances")
    L.append("excluded -- the GT score itself fails MUSTER's XML conversion, "
             "so all systems lose")
    L.append("the same 7 and the comparison stays paired.")
    L.append("")
    L.append(f"{'system':32s} {'n':>3s}" + "".join(f"{h:>10s}" for h in MHDR))
    if "self" in mus:
        r = mus["self"]
        L.append(f"{'GT self-comparison (floor)':32s} {r['n']:3d}"
                 + "".join(f"{r[m]:10.2f}" for m in MKEYS))
    for key, label in ROWS:
        r = mus.get(key)
        if not r:
            continue
        L.append(f"{label.replace(chr(92)+'&', '&'):32s} {r['n']:3d}"
                 + "".join(f"{r[m]:10.2f}" for m in MKEYS))
    return "\n".join(L) + "\n"


def tex(agg, mus) -> str:
    def f1rows():
        out = []
        for key, label in ROWS:
            n = agg.get(f"{key}|base", [0, 0])[0]
            if not n:
                continue
            paper = key in ("beyer", "zeng")
            lab = f"\\textit{{{label}}}" if paper else label
            cells = " & ".join(f"{agg[f'{key}|{m}'][1]:.2f}" for m, _ in METS)
            out.append(f"{lab} & {cells} \\\\")
            if key == "dag3 rst":
                out.append("\\midrule")
        return "\n".join(out)

    def musrows():
        out = []
        if "self" in mus:
            r = mus["self"]
            out.append("GT self-comparison (floor) & "
                       + " & ".join(f"{r[m]:.2f}" for m in MKEYS) + " \\\\")
            out.append("\\midrule")
        for key, label in ROWS:
            r = mus.get(key)
            if not r:
                continue
            paper = key in ("beyer", "zeng")
            lab = f"\\textit{{{label}}}" if paper else label
            out.append(f"{lab} & "
                       + " & ".join(f"{r[m]:.2f}" for m in MKEYS) + " \\\\")
            if key == "dag3 rst":
                out.append("\\midrule")
        return "\n".join(out)

    return f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage[margin=1in]{{geometry}}
\\begin{{document}}

\\begin{{table}}[h]
\\centering
\\caption{{Whole-song F1 on the 59-performance paper test split (\\%, higher is
better). \\textsc{{tol1}} scores onset\\,$\\pm$\\,1 grid unit and pitch on the
absolute grid; \\textsc{{scale-max}} takes the best of 13 beat-unit
rescalings; \\textsc{{IOI}} scores shift-invariant inter-onset intervals; and
\\textsc{{IOI scale-max}} is invariant to both global time-scale and offset.
DAgger leads both reference systems on the two shift-invariant families.}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
system & tol1 & scale-max & IOI & IOI scale-max \\\\
\\midrule
{f1rows()}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[h]
\\centering
\\caption{{Fixed-MUSTER error rates on whole-song transcriptions (\\%, lower is
better), on the 52 performances whose ground-truth score survives MUSTER's
MusicXML conversion. The onset column is where the residual gap to Beyer \\&
Dai lives; our systems emit more extra notes while theirs miss more.}}
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
system & pitch & missing & extra & onset & offset & mean \\\\
\\midrule
{musrows()}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\end{{document}}
"""


def main() -> None:
    agg, mus = load()
    RES.mkdir(exist_ok=True)
    (RES / "dagger_comparison.txt").write_text(txt(agg, mus))
    (RES / "dagger_comparison.tex").write_text(tex(agg, mus))
    print(txt(agg, mus))
    print(f"wrote {RES}/dagger_comparison.txt and .tex")


if __name__ == "__main__":
    main()
