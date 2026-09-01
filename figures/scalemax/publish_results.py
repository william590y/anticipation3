#!/usr/bin/env python
"""Route every table/plot artifact into results/ with LaTeX wrappers.

Idempotent -- run after any stage lands. For each known artifact it
  * copies the file into results/,
  * for plots, writes results/latex/fig_<stem>.tex -- a \\begin{figure}
    snippet with \\includegraphics + caption, ready to \\input into a paper,
  * regenerates results/latex/all_figures.tex (every figure + both tables)
    and compiles it with tectonic to results/all_figures.pdf.
Tables keep their existing standalone .tex (copied alongside).
"""
from __future__ import annotations
import shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
LTX = RES / "latex"
SRC = ROOT / "figures" / "scalemax"

# (path, caption) -- captions say what the figure IS, including its caveat.
PLOTS = [
    (SRC/"hists/hist_base_windowed_test.png",
     "Distribution of onset$\\pm$1+pitch F1, windowed test set: per window "
     "(left, $n{=}1180$) and per musical work (right, $n{=}14$ means), one row "
     "per system."),
    (SRC/"hists/hist_smax_windowed_test.png",
     "Scale-max F1 (max over ASAP beat-unit ratios); the max over 13 scales "
     "inflates every system by a comparable amount, so compare rows, not modes."),
    (SRC/"hists/hist_ioi_windowed_test.png",
     "Interarrival F1: the tol1 matcher on (inter-onset interval, pitch) "
     "events -- rhythm-shape credit without absolute alignment."),
    (SRC/"hists/hist_ioi_smax_windowed_test.png",
     "Interarrival scale-max F1."),
    (SRC/"hists/hist_notelevel_onset_err_test.png",
     "Per-note signed onset error for pitch-matched notes (162{,}978 GT "
     "notes); edge bars aggregate all $|$err$|>25$ bins."),
    (SRC/"hists/hist_notelevel_onset_err_log_test.png",
     "Per-note $|$onset error$|$ on a log axis: the apparent side modes of the "
     "linear view are one broad derailed-regime tail (median 175 bins)."),
    # --- future artifacts; skipped until they exist -------------------------
    (SRC/"hists/hist_base_windowed_test_val.png",
     "Onset$\\pm$1+pitch F1 histograms, windowed, test+val."),
    (SRC/"hists/hist_smax_windowed_test_val.png", "Scale-max F1, windowed, test+val."),
    (SRC/"hists/hist_ioi_windowed_test_val.png", "Interarrival F1, windowed, test+val."),
    (SRC/"hists/hist_ioi_smax_windowed_test_val.png", "IOI scale-max F1, windowed, test+val."),
    (SRC/"hists/hist_base_rollout_test.png",
     "Onset$\\pm$1+pitch F1, full-song sliding-window rollouts, test: per "
     "performance and per work."),
    (SRC/"hists/hist_smax_rollout_test.png", "Scale-max F1, rollouts, test."),
    (SRC/"hists/hist_ioi_rollout_test.png", "Interarrival F1, rollouts, test."),
    (SRC/"hists/hist_ioi_smax_rollout_test.png", "IOI scale-max F1, rollouts, test."),
    (SRC/"hists/hist_base_rollout_test_val.png", "Onset$\\pm$1+pitch F1, rollouts, test+val."),
    (SRC/"hists/hist_smax_rollout_test_val.png", "Scale-max F1, rollouts, test+val."),
    (SRC/"hists/hist_ioi_rollout_test_val.png", "Interarrival F1, rollouts, test+val."),
    (SRC/"hists/hist_ioi_smax_rollout_test_val.png", "IOI scale-max F1, rollouts, test+val."),
]
TABLES = [
    (SRC/"table_aggregate", "Windowed test set: four metric families, three "
     "systems, window and piece level, sign-flip permutation $p$ vs ours."),
    (SRC/"table_perpiece", "Per-work breakdown of all four metrics."),
    (SRC/"table_fullsong", "Whole-song sliding-window rollouts vs the two "
     "reference systems."),
    (SRC/"table_pitch_forcing", "Oracle pitch-forcing diagnostic."),
    (SRC/"table_muster", "MUSTER (fixed) on the windowed test set: error rates "
     "(lower is better) after repairing the metric's identity failures; the "
     "floor row is GT scored against itself."),
]

def main() -> None:
    RES.mkdir(exist_ok=True)
    LTX.mkdir(exist_ok=True)
    body = []
    for path, cap in PLOTS:
        if not path.exists():
            continue
        shutil.copy2(path, RES / path.name)
        stem = path.stem
        snip = (f"\\begin{{figure}}[t]\n  \\centering\n"
                f"  \\includegraphics[width=\\linewidth]{{{path.name}}}\n"
                f"  \\caption{{{cap}}}\n  \\label{{fig:{stem}}}\n"
                f"\\end{{figure}}\n")
        (LTX / f"fig_{stem}.tex").write_text(snip)
        body.append(snip)
    for base, cap in TABLES:
        if not base.with_suffix(".tex").exists():
            continue
        for ext in (".tex", ".png", ".pdf"):
            f = base.with_suffix(ext)
            if f.exists():
                shutil.copy2(f, RES / f.name)
        inner = base.with_suffix(".tex").read_text()
        inner = inner[inner.index("\\begin{tabular}"):]
        inner = inner[: inner.index("\\end{document}")]
        body.append(f"\\begin{{table}}[t]\n  \\centering\n  \\small\n"
                    f"  \\resizebox{{\\linewidth}}{{!}}{{%\n{inner}}}\n"
                    f"  \\caption{{{cap}}}\n  \\label{{tab:{base.stem}}}\n"
                    f"\\end{{table}}\n")
    doc = ("\\documentclass[11pt]{article}\n"
           "\\usepackage[margin=1in]{geometry}\n"
           "\\usepackage{graphicx}\\usepackage{booktabs}\\usepackage{amsmath}\n"
           "\\graphicspath{{../}}\n"
           "\\begin{document}\n"
           "\\section*{Transcription evaluation --- collected figures and tables}\n"
           + "\n\\clearpage\n".join(body) + "\n\\end{document}\n")
    (LTX / "all_figures.tex").write_text(doc)
    r = subprocess.run(["tectonic", "all_figures.tex"], cwd=LTX,
                       capture_output=True, text=True)
    if r.returncode == 0:
        shutil.move(str(LTX / "all_figures.pdf"), RES / "all_figures.pdf")
        print("compiled results/all_figures.pdf")
    else:
        print("TECTONIC FAILED:\n" + r.stderr[-800:])
    n = len(list(RES.glob("*.png"))) + len(list(RES.glob("*.tex")))
    print(f"results/: {sorted(p.name for p in RES.iterdir() if p.is_file())}")

if __name__ == "__main__":
    main()
