"""MUSTER (fixed) table: windowed test set, ours vs Beyer vs Zeng.

Error rates (LOWER is better). Includes the self-comparison floor row --
GT scored against itself under the fixed metric -- and piece-level sign-flip
p-values on the mean error rate vs ours.
"""
import json, subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
recs = json.load(open(ROOT / "nbest_data/muster_fixed_windowed_test.json"))["records"]
KEYS = [("pitch_error_rate", "pitch"), ("missing_note_rate", "miss"),
        ("extra_note_rate", "extra"), ("onset_time_error_rate", "onset"),
        ("offset_time_error_rate", "offset"), ("mean_error_rate", "MER")]
SYS = [("ours", "Ours (FT)"), ("beyer", "Beyer \\& Dai"), ("zeng", "Zeng+")]

def sf(dv, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice((-1., 1.), size=(n, dv.size)) * dv).mean(1))
    return float((np.count_nonzero(null >= abs(dv.mean()) - 1e-15) + 1) / (n + 1))

rows = []
for name, label in SYS + [("self", "GT vs itself (floor)")]:
    vals = {k: [r[name][k] for r in recs if r.get(name)] for k, _ in KEYS}
    rows.append((name, label, [np.mean(vals[k]) for k, _ in KEYS],
                 len(vals[KEYS[0][0]])))
# piece-level MER + p
byw = defaultdict(list)
for r in recs:
    if r.get("work"):
        byw[r["work"]].append(r)
pw = {name: np.array([np.mean([r[name]["mean_error_rate"] for r in g if r.get(name)])
                      for g in byw.values()]) for name, _ in SYS}
p_b = sf(pw["beyer"] - pw["ours"]); p_z = sf(pw["zeng"] - pw["ours"])

body = [r"\begin{tabular}{l rrrrrr r r}", r"\toprule",
        "system & " + " & ".join(l for _, l in KEYS)
        + r" & MER (piece) & $n$ \\", r"\midrule"]
best = {i: min(rows[j][2][i] for j in range(3)) for i in range(len(KEYS))}
for j, (name, label, vals, n) in enumerate(rows):
    cells = []
    for i, v in enumerate(vals):
        s = f"{v:.2f}"
        if j < 3 and abs(v - best[i]) < 1e-9:
            s = f"\\textbf{{{s}}}"
        cells.append(s)
    pm = f"{pw[name].mean():.2f}" if name in pw else "--"
    if name in pw and pw[name].mean() == min(x.mean() for x in pw.values()):
        pm = f"\\textbf{{{pm}}}"
    if name == "self":
        body.append(r"\midrule")
    body.append(f"{label} & " + " & ".join(cells) + f" & {pm} & {n} \\\\")
body += [r"\midrule",
         (r"\multicolumn{9}{l}{piece-level ($n{=}14$ works) sign-flip $p$ vs ours: "
          f"Beyer $p={p_b:.3f}$, Zeng $p={p_z:.3f}$}}\\\\"),
         r"\bottomrule", r"\end{tabular}"]
DOC = ("\\documentclass[border=6pt]{standalone}\n\\usepackage{booktabs}"
       "\\usepackage{amsmath}\n\\begin{document}%s\\end{document}")
(HERE / "table_muster.tex").write_text(DOC % "\n".join(body))
subprocess.run(["tectonic", "table_muster.tex"], cwd=HERE, check=True,
               capture_output=True)
subprocess.run(["convert", "-density", "300", str(HERE / "table_muster.pdf"),
                "-background", "white", "-alpha", "remove",
                str(HERE / "table_muster.png")], check=True)
print(f"wrote table_muster.*   piece-level MER: "
      + "  ".join(f"{n}={pw[n].mean():.2f}" for n in pw)
      + f"   p_B={p_b:.4f} p_Z={p_z:.4f}")
