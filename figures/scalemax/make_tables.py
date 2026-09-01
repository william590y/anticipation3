"""Emit LaTeX for the two scale-max/IOI tables and render PNGs via tectonic.

Numbers come from nbest_data/test_set_scalemax_f1.json + the piece map --
regenerated, never transcribed, so the .tex can't drift from the data.
"""
import json, subprocess, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
d = json.load(open(ROOT / "nbest_data/test_set_scalemax_f1.json"))
sel = json.load(open(ROOT / "nbest_data/test_set_selector_eval.json"))
pieces = sel["pieces"]
byw = defaultdict(list)
for r in d["records"]:
    wi = int(r["key"].split("-")[1])
    w = pieces.get(str(wi), pieces.get(wi))
    if w: byw[w.rsplit("/", 1)[0]].append(r)
METS = [("base", r"tol.\ $\pm1$"), ("smax", "scale-max"),
        ("ioi", "interarrival"), ("ioi_smax", "IOI scale-max")]
MODELS = [("ours", "Ours (FT)"), ("beyer", "Beyer \\& Dai"), ("zeng", "Zeng+")]

def sf(dv, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice((-1., 1.), size=(n, dv.size)) * dv).mean(1))
    return float((np.count_nonzero(null >= abs(dv.mean()) - 1e-15) + 1) / (n + 1))

recs = d["records"]
pm = {(m, k): np.array([np.mean([r[f"{m}_{k}"] for r in g]) for g in byw.values()])
      for m, _ in MODELS for k, _ in METS}

# ---------------- table 1: aggregate ----------------------------------------
rows1 = []
for k, lab in METS:
    w = [100*np.mean([r[f"{m}_{k}"] for r in recs]) for m, _ in MODELS]
    p = [100*pm[(m, k)].mean() for m, _ in MODELS]
    pb = sf(pm[("beyer", k)] - pm[("ours", k)])
    pz = sf(pm[("zeng", k)] - pm[("ours", k)])
    def bold(vals):
        i = int(np.argmax(vals))
        return [f"\\textbf{{{v:.1f}}}" if j == i else f"{v:.1f}"
                for j, v in enumerate(vals)]
    rows1.append((lab, bold(w), bold(p), pb, pz))
t1 = [r"\begin{tabular}{l ccc ccc cc}", r"\toprule",
      r" & \multicolumn{3}{c}{window level ($n{=}1180$)} & \multicolumn{3}{c}{piece level ($n{=}14$ works)} & & \\",
      r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
      "metric & " + " & ".join(l for _, l in MODELS) + " & "
      + " & ".join(l for _, l in MODELS)
      + r" & $p_{\mathrm{B-O}}$ & $p_{\mathrm{Z-O}}$ \\", r"\midrule"]
for lab, w, p, pb, pz in rows1:
    t1.append(f"{lab} & " + " & ".join(w) + " & " + " & ".join(p)
              + f" & {pb:.3f} & {pz:.3f} \\\\")
t1 += [r"\bottomrule", r"\end{tabular}"]

# ---------------- table 2: per-piece ----------------------------------------
t2 = [r"\begin{tabular}{l r " + "ccc " * 4 + "}", r"\toprule",
      r"work & $n$ & " + " & ".join(
          f"\\multicolumn{{3}}{{c}}{{{lab}}}" for _, lab in METS) + r" \\",
      "".join(f"\\cmidrule(lr){{{3+3*i}-{5+3*i}}}" for i in range(4)),
      r" & & " + " & ".join("O & B & Z" for _ in METS) + r" \\", r"\midrule"]
for w in sorted(byw, key=lambda x: -len(byw[x])):
    g = byw[w]
    nm = (w.split("/")[-2] if w.count("/") >= 2 else w).replace("_", r"\_")
    cells = []
    for k, _ in METS:
        vals = [100*np.mean([r[f"{m}_{k}"] for r in g]) for m, _ in MODELS]
        i = int(np.argmax(vals))
        cells += [f"\\textbf{{{v:.0f}}}" if j == i else f"{v:.0f}"
                  for j, v in enumerate(vals)]
    t2.append(f"{nm} & {len(g)} & " + " & ".join(cells) + r" \\")
t2.append(r"\midrule")
cells = []
for k, _ in METS:
    vals = [100*pm[(m, k)].mean() for m, _ in MODELS]
    i = int(np.argmax(vals))
    cells += [f"\\textbf{{{v:.1f}}}" if j == i else f"{v:.1f}"
              for j, v in enumerate(vals)]
t2.append(r"mean over pieces & 14 & " + " & ".join(cells) + r" \\")
t2 += [r"\bottomrule", r"\end{tabular}"]

DOC = r"""\documentclass[border=6pt]{standalone}
\usepackage{booktabs}\usepackage{amsmath}
\begin{document}%s\end{document}"""
here = Path(__file__).parent
for name, body in (("table_aggregate", "\n".join(t1)),
                   ("table_perpiece", "\n".join(t2))):
    tex = here / f"{name}.tex"
    tex.write_text(DOC % body)
    subprocess.run(["tectonic", str(tex)], cwd=here, check=True,
                   capture_output=True)
    subprocess.run(["convert", "-density", "300", str(here / f"{name}.pdf"),
                    "-background", "white", "-alpha", "remove",
                    str(here / f"{name}.png")], check=True)
    print(f"wrote {name}.tex / .pdf / .png")
